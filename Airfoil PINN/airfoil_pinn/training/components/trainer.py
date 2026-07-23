# Исправленный файл: trainer.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль trainer для проекта PINN Learning.
Содержит класс Trainer, отвечающий за прямое распространение,
шаги оптимизации (Adam и L-BFGS), RAR и адаптивную балансировку весов,
а также сбор диагностических данных.

Исправленная версия:
- В RAR новые точки инициализируются нулевой невязкой, чтобы избежать дублирования высоких значений.
- Метод get_profile_at_x помечен как устаревший (для диагностики использовать data_handler.interpolate_profile с предсказаниями).
- Незначительные улучшения.

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная)
"""

import logging
import os
from collections import deque
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
from scipy.spatial import cKDTree

from airfoil_pinn.app.paths import PICTURES_DIR
from airfoil_pinn.utils import safe_grad
from airfoil_pinn.data.sdf_diagnostics import (
    compute_sdf_summary,
    estimate_boundary_layer_delta,
    format_sdf_summary,
)
from airfoil_pinn.training.components.rar import RARManager

logger = logging.getLogger(__name__)

__all__ = ["Trainer"]


class Trainer:
    """
    Класс, реализующий логику обучения PINN: прямой проход, шаги оптимизации,
    адаптивную балансировку весов, RAR и сбор диагностических метрик.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_computer,
        data_handler,
        config: Dict,
        device: torch.device,
        dtype: torch.dtype,
        sa_weights: Optional[nn.Module] = None,
        scaler: Optional[GradScaler] = None,
        using_lbfgs: bool = False,
        diagnostics_config: Optional[Dict] = None,
        use_amp: Optional[bool] = None,
        plots_dir: Optional[str] = None,
    ):
        """
        Args:
            model: нейросетевая модель
            optimizer: оптимизатор (Adam или L-BFGS)
            loss_computer: объект для вычисления компонентов функции потерь
            data_handler: обработчик данных (предоставляет DataLoader и метаданные)
            config: словарь конфигурации (раздел 'training')
            device: устройство для вычислений
            dtype: тип данных (torch.float32, torch.float64 и т.д.)
            sa_weights: обучаемые веса для SA-PINN (опционально)
            scaler: скалер для AMP (опционально)
            using_lbfgs: флаг, указывающий, что используется L-BFGS
            diagnostics_config: словарь с настройками диагностики (интервалы, флаги и т.п.)
            plots_dir: директория для сохранения диагностических графиков
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.data_handler = data_handler
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sa_weights = sa_weights
        self.scaler = scaler
        self.using_lbfgs = using_lbfgs
        self.plots_dir = plots_dir

        # Параметры обучения
        tr = config
        self.grad_clip = tr.get('grad_clip')
        self.accum_steps = tr.get('gradient_accumulation_steps', 1)
        self.reset_interval = tr.get('reset_interval', 0)
        self.reset_noise = tr.get('reset_noise', 0.01)
        self.rar_enable = tr.get('rar_enable', False)
        self.rar_interval = tr.get('rar_interval', 100)
        self.rar_num_points = tr.get('rar_num_points', 200)
        self.rar_mode = str(tr.get('rar_mode', 'adaptive')).lower()
        self.rar_score_mode = str(tr.get('rar_score_mode', 'total_loss')).lower()
        self.rar_error_quantile = float(tr.get('rar_error_quantile', 0.75))
        self.rar_residual_power = float(tr.get('rar_residual_power', 1.0))
        self.rar_candidate_multiplier = max(4, int(tr.get('rar_candidate_multiplier', 12)))
        self.rar_jitter_scale = float(tr.get('rar_jitter_scale', 0.85))
        self.rar_min_distance_factor = float(tr.get('rar_min_distance_factor', 0.35))
        self.rar_min_boundary_distance_factor = float(
            tr.get('rar_min_boundary_distance_factor', 0.05)
        )
        self.rar_rebuild_fvm = bool(tr.get('rar_rebuild_fvm', True))
        self.rar_fvm_rebuild_every = max(1, int(tr.get('rar_fvm_rebuild_every', 1)))
        self.rar_fvm_min_new_points = max(1, int(tr.get('rar_fvm_min_new_points', 1)))
        self.rar_save_mesh_volume_plot = bool(tr.get('rar_save_mesh_volume_plot', True))
        self.rar_mesh_plot_dpi = int(tr.get('rar_mesh_plot_dpi', 150))
        self.use_amp = bool(use_amp) if use_amp is not None else tr.get('use_amp', False)

        # Адаптивная балансировка весов BC (только для soft граничных условий)
        # Флаг adaptive_bc читается из LossComputer, который берёт его из config['loss_weights']
        self.adaptive_weights = self.loss_computer.adaptive_bc and self.loss_computer.bc_type == 'soft'
        self.adapt_frequency = 50
        self.adapt_window = 10
        self.loss_ma = {}
        for comp in self.loss_computer.pde_components:
            self.loss_ma[comp] = deque(maxlen=self.adapt_window)
        if self.loss_computer.bc_type == 'soft':
            for name in self.data_handler.boundary_type_names:
                self.loss_ma[name] = deque(maxlen=self.adapt_window)

        # Хранилище score для RAR и диагностики (на CPU).
        # point_residual сейчас означает именно RAR score: local total-loss
        # при rar_score_mode='total_loss' или PDE-score при rar_score_mode='pde'.
        self.point_residual = torch.zeros(
            self.data_handler.n_points,
            device='cpu',
            dtype=self.dtype
        )
        self.max_residual = torch.zeros_like(self.point_residual)
        # Отдельная карта чистой PDE-невязки для привычного residual_map.
        self.point_pde_residual = torch.zeros_like(self.point_residual)
        self.max_pde_residual = torch.zeros_like(self.point_residual)

        # История RAR (для статистики)
        self.rar_history = []  # список словарей с типами границ добавленных точек

        # Флаг изменения SA-весов (для сигнала solver'у о необходимости пересоздания оптимизатора)
        self.sa_weights_changed = False

        # FVM-потеря: раз в fvm_interval эпох; внутри эпохи — раз в fvm_batch_interval батчей
        self.fvm_interval = config.get('fvm_interval', 1)
        self.fvm_batch_interval: int = getattr(loss_computer, 'fvm_batch_interval', 1)
        self._fvm_epoch_counter = 0
        self._last_fvm_loss = 0.0
        self._last_integral_loss = 0.0
        self._rar_fvm_rebuild_counter = 0
        self._rar_new_points_since_fvm = 0
        self._rar_step_counter = 0

        # Параметры диагностики
        self.diag_config = diagnostics_config or {}
        self.diag_interval = self.diag_config.get('interval', 50)
        self.profile_sections = self.diag_config.get('profile_sections', [0.2, 0.5, 0.8])
        self.spectrum_section_x = self.diag_config.get('spectrum_section_x', 0.5)
        self.compute_grad_cosine = self.diag_config.get('compute_grad_cosine', False)
        self.strict_autograd = bool(self.diag_config.get('strict_autograd', True))
        self._fvm_tensor_cache = None
        self.rar_manager = RARManager(self)

    def _grad(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        create_graph: bool = True,
        silent: bool = False,
    ) -> torch.Tensor:
        """Вычисляет dy/dx с учетом diagnostics.strict_autograd."""
        return safe_grad(
            y,
            x,
            create_graph=create_graph,
            silent=silent,
            strict=self.strict_autograd,
        )

    def _autocast_context(self):
        if self.use_amp and self.scaler is not None and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    # ------------------------------------------------------------------
    # Прямой проход (вычисление всех компонент потерь)
    # ------------------------------------------------------------------
    def forward_pass(
        self,
        batch: Tuple,
        pde_mult: float = 1.0,
        add_fvm: bool = False,
        fvm_scale: float = 1.0,
        add_integral: bool = False,
        integral_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Выполняет прямой проход для одного батча, вычисляет все компоненты потерь
        и возвращает словарь с результатами, включая отдельные тензоры потерь.
        """
        idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch

        idx = idx.to(self.device)
        x = x.to(self.device, dtype=self.dtype).requires_grad_(True)
        y = y.to(self.device, dtype=self.dtype).requires_grad_(True)
        sdf = sdf.to(self.device, dtype=self.dtype)

        if bt.shape[1] > 0:
            bt = bt.to(self.device, dtype=self.dtype)
        else:
            bt = None

        sdf, sdf_pde_mask = self.loss_computer.prepare_sdf_for_pde(
            x,
            y,
            sdf,
            bt,
        )

        mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
        mask_v = mask_v.to(self.device, dtype=torch.bool).view(-1, 1)
        mask_p = mask_p.to(self.device, dtype=torch.bool).view(-1, 1)
        u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)
        v_bc = v_bc.to(self.device, dtype=self.dtype).view(-1, 1)
        p_bc = p_bc.to(self.device, dtype=self.dtype).view(-1, 1)

        # Формируем входной тензор для модели
        input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)

        # Прямой проход модели
        out = self.model(input_tensor)
        u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # Применение жёстких граничных условий (если необходимо)
        if self.loss_computer.bc_type == 'hard':
            u, v, p = self.loss_computer.apply_hard_constraints(
                u_raw, v_raw, p_raw, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc
            )
        else:
            u, v, p = u_raw, v_raw, p_raw
        u, v = self.loss_computer.apply_velocity_dirichlet_constraint(u, v, bt)

        # Вычисление невязок PDE
        res = self.loss_computer.compute_pde_residuals(u, v, p, x, y)
        cont = res['cont']
        mom1 = res[self.loss_computer.pde_components[1]]
        mom2 = res[self.loss_computer.pde_components[2]]
        pde_mask = self._pde_loss_mask(bt, cont) & sdf_pde_mask
        pde_score = cont + mom1 + mom2
        self._update_pde_point_residual(
            idx,
            torch.where(pde_mask, pde_score, torch.zeros_like(pde_score)),
        )

        # Средние и максимальные значения для отчёта
        pde_raw = {
            'cont': self._masked_mean(cont, pde_mask).item(),
            self.loss_computer.pde_components[1]: self._masked_mean(mom1, pde_mask).item(),
            self.loss_computer.pde_components[2]: self._masked_mean(mom2, pde_mask).item()
        }
        pde_max = {
            'cont': self._masked_max(cont, pde_mask).item(),
            self.loss_computer.pde_components[1]: self._masked_max(mom1, pde_mask).item(),
            self.loss_computer.pde_components[2]: self._masked_max(mom2, pde_mask).item()
        }

        # Взвешивание PDE с учётом SA-весов (если используются)
        w_pde = None
        w_bc = None
        if self.sa_weights is not None:
            w = self.sa_weights()[idx]
            if self.loss_computer.sa_apply_to_bc:
                w_pde = w[:, :3]
                w_bc = w[:, 3:]
            else:
                w_pde = w
                w_bc = None
            loss_c = self._masked_mean(w_pde[:, 0].view(-1, 1) * cont, pde_mask)
            loss_1 = self._masked_mean(w_pde[:, 1].view(-1, 1) * mom1, pde_mask)
            loss_2 = self._masked_mean(w_pde[:, 2].view(-1, 1) * mom2, pde_mask)
        else:
            loss_c = self._masked_mean(cont, pde_mask)
            loss_1 = self._masked_mean(mom1, pde_mask)
            loss_2 = self._masked_mean(mom2, pde_mask)
            w_bc = None

        # PDE потеря с учётом весов из конфигурации
        pde_loss = pde_mult * (
            self.loss_computer.loss_weights['cont'] * loss_c +
            self.loss_computer.loss_weights[self.loss_computer.pde_components[1]] * loss_1 +
            self.loss_computer.loss_weights[self.loss_computer.pde_components[2]] * loss_2
        )

        # Граничные условия (только для soft)
        if self.loss_computer.bc_type == 'soft':
            bc_comp, extra_grad = self.loss_computer.compute_bc_loss(u, v, p, bt, x, y)
            # Если SA-веса применяются к BC, используем только их; иначе — фиксированные веса
            if self.sa_weights is not None and self.loss_computer.sa_apply_to_bc and w_bc is not None:
                bc_loss = self._compute_sa_bc_loss(
                    u=u,
                    v=v,
                    p=p,
                    boundary_types=bt,
                    w_bc=w_bc,
                ) + extra_grad
            else:
                bc_loss = sum(
                    self.loss_computer.loss_weights.get(name, 0.0) * bc_comp[name]
                    for name in bc_comp
                ) + extra_grad
        else:
            bc_comp = {}
            # Создаём нулевой тензор, который требует градиент (чтобы избежать ошибок в диагностике)
            bc_loss = torch.zeros(1, device=self.device, dtype=self.dtype, requires_grad=True).squeeze()

        # RAR использует локальный аналог Total Loss: взвешенная PDE-часть
        # плюс точечные BC-вклады. Глобальные integral/FVM не имеют координатной
        # локализации и не влияют на ранжирование точек.
        rar_point_score = self._compute_rar_point_score(
            cont=cont,
            mom1=mom1,
            mom2=mom2,
            u=u,
            v=v,
            p=p,
            x=x,
            y=y,
            boundary_types=bt,
            pde_mask=pde_mask,
            pde_mult=pde_mult,
            w_pde=w_pde,
            w_bc=w_bc,
        )
        self._update_rar_point_score(idx, rar_point_score)

        # Глобальные интегральные/FVM потери не считаются по случайному mini-batch.
        # Trainer включает их только на запланированных batch-ах и нормирует
        # вклад так, чтобы пользовательский вес не умножался на число запусков.
        integral_raw_t = torch.zeros(1, device=self.device, dtype=self.dtype).squeeze()
        integral_t = torch.zeros(1, device=self.device, dtype=self.dtype).squeeze()
        fvm_loss_val = 0.0
        fvm_t = torch.zeros(1, device=self.device, dtype=self.dtype).squeeze()

        needs_fvm_forward = (
            self.loss_computer.fvm is not None
            and (
                (add_fvm and self.loss_computer.fvm_weight > 0.0)
                or (add_integral and self.loss_computer.integral_condition)
            )
        )
        out_fvm = None
        x_fvm = None
        if needs_fvm_forward:
            if hasattr(self.data_handler, 'fvm_x'):
                x_fvm, y_fvm, sdf_fvm, bt_fvm = self._get_fvm_tensors()
                inp_fvm  = self.loss_computer.build_input_tensor(x_fvm, y_fvm, sdf_fvm, bt_fvm)
                out_fvm  = self.model(inp_fvm)
                u_fvm, v_fvm, p_fvm = out_fvm[:, 0:1], out_fvm[:, 1:2], out_fvm[:, 2:3]
                if self.loss_computer.bc_type == 'hard' and bt_fvm is not None:
                    u_fvm, v_fvm, p_fvm = self._apply_hard_constraints_from_boundary_types(
                        u_fvm,
                        v_fvm,
                        p_fvm,
                        bt_fvm,
                    )
                u_fvm, v_fvm = self.loss_computer.apply_velocity_dirichlet_constraint(
                    u_fvm, v_fvm, bt_fvm
                )
                out_fvm = torch.cat([u_fvm, v_fvm, p_fvm], dim=1)
            else:
                logger.warning("FVM/integral: fvm-тензоры не подготовлены, глобальный сеточный проход пропущен.")

        if add_integral and self.loss_computer.integral_condition:
            if out_fvm is not None and self.loss_computer.fvm is not None:
                integral_raw_t = self.loss_computer.compute_mesh_integral_loss(
                    out_fvm[:, :2],
                    self.loss_computer.fvm.bnd_node,
                    self.loss_computer.fvm.bnd_S,
                    self.loss_computer.fvm.bnd_type,
                    node_x=x_fvm,
                )
            else:
                integral_raw_t = self._compute_global_integral_loss_from_dataset()
            integral_t = integral_raw_t * self.loss_computer.integral_weight * float(integral_scale)

        if add_fvm and out_fvm is not None and self.loss_computer.fvm_weight > 0.0:
            fvm_weighted_t = self.loss_computer.compute_fvm_loss(out_fvm[:, :2])
            fvm_loss_val = fvm_weighted_t.item()
            fvm_t = fvm_weighted_t * float(fvm_scale)

        # Полная потеря — единый граф для backward, градиенты PDE/BC/FVM не конкурируют
        total_loss = pde_loss + bc_loss + integral_t + fvm_t

        # Средняя скорость на стенке (для мониторинга)
        wall_speed = None
        if self.loss_computer.bc_type == 'soft' and 'wall' in bc_comp and bc_comp['wall'].item() > 0 and bt is not None:
            wall_idx = self.data_handler.boundary_type_names.index('wall')
            u_wall = u[bt[:, wall_idx] > 0.5]
            v_wall = v[bt[:, wall_idx] > 0.5]
            if len(u_wall) > 0:
                wall_speed = torch.sqrt(u_wall**2 + v_wall**2).mean().item()

        return {
            'total': total_loss,
            'pde_raw': pde_raw,
            'pde_max': pde_max,
            'bc_comp': {k: v.item() for k, v in bc_comp.items()} if self.loss_computer.bc_type == 'soft' else {},
            'extra_grad': extra_grad.item() if self.loss_computer.bc_type == 'soft' else 0.0,
            'integral_loss': integral_raw_t.item() if torch.is_tensor(integral_raw_t) else integral_raw_t,
            'fvm_loss': fvm_loss_val,
            'wall_speed': wall_speed,
            # Возвращаем отдельные тензоры потерь для градиентного анализа
            'loss_components': {
                'cont': loss_c,
                'mom1': loss_1,
                'mom2': loss_2,
                'bc': bc_loss,          # теперь всегда тензор
                'integral': integral_t
            },
            'u': u.detach(),  # для профилей и спектров
            'v': v.detach(),
            'x': x.detach(),
            'y': y.detach(),
            'idx': idx.cpu()
        }

    def _pde_loss_mask(
        self,
        boundary_types: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Возвращает mask точек, участвующих в PDE-loss."""
        if getattr(self.loss_computer, 'pde_on_boundary', False) or boundary_types is None:
            return torch.ones_like(reference, dtype=torch.bool)

        interior_idx = getattr(self.data_handler.dataset, 'type_to_idx', {}).get('interior')
        if interior_idx is None or interior_idx >= boundary_types.shape[1]:
            return torch.ones_like(reference, dtype=torch.bool)

        mask = boundary_types[:, interior_idx].view(-1, 1) > 0.5
        if not bool(mask.any()):
            logger.debug("PDE-loss mask is empty for this boundary-only batch.")
        return mask

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Среднее по mask с защитой от пустой маски."""
        if mask is None or mask.shape != values.shape:
            return values.mean()
        if not bool(mask.any()):
            return values.sum() * 0.0
        return values[mask].mean()

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Максимум по mask с защитой от пустой маски."""
        if mask is None or mask.shape != values.shape:
            return values.max()
        if not bool(mask.any()):
            return values.sum() * 0.0
        return values[mask].max()

    def _compute_sa_bc_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        w_bc: torch.Tensor,
    ) -> torch.Tensor:
        """Pointwise BC-loss с self-adaptive весами по активным boundary-точкам."""
        if boundary_types is None or boundary_types.shape[1] == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)

        terms = []
        for i, name in enumerate(self.data_handler.boundary_type_names):
            if i >= boundary_types.shape[1] or i >= w_bc.shape[1]:
                continue
            if self.loss_computer.loss_weights.get(name, 0.0) <= 0.0:
                continue

            mask = boundary_types[:, i].view(-1, 1) > 0.5
            if not bool(mask.any()):
                continue

            if name in ('inlet', 'farfield', 'velocity_dirichlet'):
                point_loss = (
                    (u - self.data_handler.inlet_u_norm) ** 2
                    + (v - self.data_handler.inlet_v_norm) ** 2
                )
            elif name == 'outlet':
                point_loss = p ** 2
            elif name == 'wall':
                point_loss = u ** 2 + v ** 2
            elif name == 'slip':
                point_loss = (v - self.data_handler.inlet_v_norm) ** 2
            elif name == 'symmetry':
                point_loss = v ** 2
            elif name == 'axis_symmetry':
                point_loss = u ** 2
            else:
                continue

            weighted = point_loss * w_bc[:, i].view(-1, 1)
            terms.append(weighted[mask].mean())

        if not terms:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return torch.stack(terms).sum()

    def _apply_hard_constraints_from_boundary_types(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        p_raw: torch.Tensor,
        boundary_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Применяет hard constraints к произвольному набору точек по one-hot ГУ."""
        u = u_raw
        v = v_raw
        p = p_raw

        for idx, name in enumerate(self.data_handler.boundary_type_names):
            if idx >= boundary_types.shape[1]:
                continue
            mask = boundary_types[:, idx].view(-1, 1) > 0.5
            if not bool(mask.any()):
                continue

            if name in ('inlet', 'velocity_dirichlet'):
                u = torch.where(mask, torch.ones_like(u) * self.data_handler.inlet_u_norm, u)
                v = torch.where(mask, torch.ones_like(v) * self.data_handler.inlet_v_norm, v)
            elif name == 'outlet':
                p = torch.where(mask, torch.zeros_like(p), p)
            elif name == 'wall':
                u = torch.where(mask, torch.zeros_like(u), u)
                v = torch.where(mask, torch.zeros_like(v), v)
            elif name == 'symmetry':
                v = torch.where(mask, torch.zeros_like(v), v)
            elif name == 'axis_symmetry':
                u = torch.where(mask, torch.zeros_like(u), u)

        return u, v, p

    def _compute_rar_point_score(
        self,
        cont: torch.Tensor,
        mom1: torch.Tensor,
        mom2: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        pde_mask: torch.Tensor,
        pde_mult: float,
        w_pde: Optional[torch.Tensor],
        w_bc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Возвращает локальный score для RAR.

        rar_score_mode='total_loss' приближает batch total_loss на уровне
        отдельных точек: weighted PDE + pointwise BC. Глобальные integral/FVM
        части не добавляются, так как они не задают место вставки новой точки.
        rar_score_mode='pde' оставляет прежнее поведение.
        """
        if self.rar_score_mode in ('pde', 'residual', 'pde_residual'):
            score = cont + mom1 + mom2
            score = torch.where(pde_mask, score, torch.zeros_like(score))
            return torch.nan_to_num(score.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

        lw = self.loss_computer.loss_weights
        pde_terms = [
            lw['cont'] * cont,
            lw[self.loss_computer.pde_components[1]] * mom1,
            lw[self.loss_computer.pde_components[2]] * mom2,
        ]
        if w_pde is not None:
            pde_terms = [
                pde_terms[0] * w_pde[:, 0].view(-1, 1),
                pde_terms[1] * w_pde[:, 1].view(-1, 1),
                pde_terms[2] * w_pde[:, 2].view(-1, 1),
            ]
        score = pde_mult * (pde_terms[0] + pde_terms[1] + pde_terms[2])
        score = torch.where(pde_mask, score, torch.zeros_like(score))

        if self.loss_computer.bc_type == 'soft' and boundary_types is not None:
            score = score + self._compute_bc_point_score(
                u=u,
                v=v,
                p=p,
                x=x,
                y=y,
                boundary_types=boundary_types,
                w_bc=w_bc,
            ).view(-1, 1)

        return torch.nan_to_num(score.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_bc_point_score(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        boundary_types: torch.Tensor,
        w_bc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Точечный вклад BC-loss для RAR total_loss score."""
        score = torch.zeros_like(u).flatten()
        if boundary_types.shape[1] == 0:
            return score

        for i, name in enumerate(self.data_handler.boundary_type_names):
            if i >= boundary_types.shape[1]:
                continue
            mask = (boundary_types[:, i].view(-1, 1) > 0.5)
            if not bool(mask.any()):
                continue

            point_loss = torch.zeros_like(u)
            extra_point_loss = torch.zeros_like(u)
            if name in ('inlet', 'farfield', 'velocity_dirichlet'):
                point_loss = (
                    (u - self.data_handler.inlet_u_norm) ** 2
                    + (v - self.data_handler.inlet_v_norm) ** 2
                )
            elif name == 'outlet':
                point_loss = p ** 2
            elif name == 'wall':
                point_loss = u ** 2 + v ** 2
            elif name == 'slip':
                tol = getattr(self.loss_computer, 'tol', 1e-6)
                x_min = getattr(self.loss_computer, 'x_min', 0.0)
                x_max = getattr(self.loss_computer, 'x_max', 0.0)
                y_min = getattr(self.loss_computer, 'y_min', 0.0)
                y_max = getattr(self.loss_computer, 'y_max', 0.0)
                left = mask & (x <= x_min + tol)
                right = mask & (x >= x_max - tol)
                bottom = mask & (y <= y_min + tol)
                top = mask & (y >= y_max - tol)
                vertical = left | right
                horizontal = bottom | top
                if bool(horizontal.any()) or bool(vertical.any()):
                    point_loss = torch.where(
                        horizontal,
                        (v - self.data_handler.inlet_v_norm) ** 2,
                        point_loss,
                    )
                    point_loss = torch.where(
                        vertical,
                        (u - self.data_handler.inlet_u_norm) ** 2,
                        point_loss,
                    )
                    slip_velocity = self.loss_computer.loss_weights.get('slip_velocity', 0.1)
                    slip_tangent_grad = self.loss_computer.loss_weights.get('slip_tangent_grad', 1.0)
                    if bool(horizontal.any()):
                        du_dy = self._grad(u, y, create_graph=True, silent=True)
                        extra_point_loss = torch.where(
                            horizontal,
                            extra_point_loss + slip_tangent_grad * du_dy ** 2
                            + slip_velocity * (u - self.data_handler.inlet_u_norm) ** 2,
                            extra_point_loss,
                        )
                    if bool(vertical.any()):
                        dv_dx = self._grad(v, x, create_graph=True, silent=True)
                        extra_point_loss = torch.where(
                            vertical,
                            extra_point_loss + slip_tangent_grad * dv_dx ** 2
                            + slip_velocity * (v - self.data_handler.inlet_v_norm) ** 2,
                            extra_point_loss,
                        )
                else:
                    point_loss = (v - self.data_handler.inlet_v_norm) ** 2
                    du_dy = self._grad(u, y, create_graph=True, silent=True)
                    extra_point_loss = (
                        self.loss_computer.loss_weights.get('slip_tangent_grad', 1.0) * du_dy ** 2
                        + self.loss_computer.loss_weights.get('slip_velocity', 0.1)
                        * (u - self.data_handler.inlet_u_norm) ** 2
                    )
            elif name == 'symmetry':
                # create_graph=True keeps the training graph intact for the
                # subsequent total_loss.backward().
                du_dy = self._grad(u, y, create_graph=True, silent=True)
                point_loss = v ** 2
                extra_point_loss = self.loss_computer.loss_weights.get('symmetry_grad', 10.0) * du_dy ** 2
            elif name == 'axis_symmetry':
                # create_graph=True keeps the training graph intact for the
                # subsequent total_loss.backward().
                dv_dr = self._grad(v, x, create_graph=True, silent=True)
                point_loss = u ** 2
                extra_point_loss = self.loss_computer.loss_weights.get('axis_symmetry_grad', 10.0) * dv_dr ** 2
            else:
                continue

            if self.sa_weights is not None and self.loss_computer.sa_apply_to_bc and w_bc is not None:
                weighted = point_loss.flatten() * w_bc[:, i]
            else:
                weighted = point_loss.flatten() * self.loss_computer.loss_weights.get(name, 0.0)
            weighted = weighted + extra_point_loss.flatten()
            score = torch.where(mask.flatten(), score + weighted, score)

        return score

    def _update_rar_point_score(self, idx: torch.Tensor, point_score: torch.Tensor) -> None:
        """Обновляет EMA и максимум score, по которому adaptive RAR выбирает области."""
        score_cpu = point_score.detach().cpu().flatten()
        idx_cpu = idx.detach().cpu()
        alpha = 0.9
        self.point_residual[idx_cpu] = alpha * self.point_residual[idx_cpu] + (1 - alpha) * score_cpu
        self.max_residual[idx_cpu] = torch.max(self.max_residual[idx_cpu], score_cpu)

    def _update_pde_point_residual(self, idx: torch.Tensor, pde_score: torch.Tensor) -> None:
        """Обновляет EMA и максимум чистой PDE-невязки для residual_map."""
        score_cpu = pde_score.detach().cpu().flatten()
        idx_cpu = idx.detach().cpu()
        alpha = 0.9
        self.point_pde_residual[idx_cpu] = alpha * self.point_pde_residual[idx_cpu] + (1 - alpha) * score_cpu
        self.max_pde_residual[idx_cpu] = torch.max(self.max_pde_residual[idx_cpu], score_cpu)

    # ------------------------------------------------------------------
    # Вычисление градиентов PDE и BC отдельно
    # ------------------------------------------------------------------
    def compute_grad_norms_separately(self, batch: Tuple) -> Dict[str, float]:
        """
        Вычисляет нормы градиентов от PDE loss и BC loss, а также косинус угла между ними.
        Возвращает словарь с ключами 'pde_norm', 'bc_norm', 'cosine'.
        """
        self.model.zero_grad()
        fwd = self.forward_pass(batch)

        # Извлекаем компоненты потерь (уже тензоры)
        loss_pde = fwd['loss_components']['cont'] + fwd['loss_components']['mom1'] + fwd['loss_components']['mom2']
        loss_bc = fwd['loss_components']['bc']
        # Интегральная потеря не включается в раздельные градиенты

        def _flatten_grads() -> torch.Tensor:
            grads = [p.grad.reshape(-1).clone()
                     for p in self.model.parameters()
                     if p.grad is not None]
            if not grads:
                return torch.empty(0, device=self.device, dtype=self.dtype)
            return torch.cat(grads)

        # Градиенты от PDE
        loss_pde.backward(retain_graph=True)
        grad_pde = _flatten_grads()
        self.model.zero_grad()

        # Градиенты от BC
        loss_bc.backward(retain_graph=True)
        grad_bc = _flatten_grads()
        self.model.zero_grad()

        norm_pde = torch.norm(grad_pde).item() if grad_pde.numel() else 0.0
        norm_bc = torch.norm(grad_bc).item() if grad_bc.numel() else 0.0
        if norm_pde > 0 and norm_bc > 0:
            cosine = torch.dot(grad_pde, grad_bc).item() / (norm_pde * norm_bc)
        else:
            cosine = 0.0

        return {'pde_norm': norm_pde, 'bc_norm': norm_bc, 'cosine': cosine}

    def compute_ntk_condition(self, batch: Tuple, max_samples: int = 64) -> float:
        """
        Оценивает condition number эмпирического NTK по малому подбатчу.

        Диагностика дорогая, поэтому используется только при явном включении
        compute_ntk_condition и ограничивает число точек max_samples.
        """
        was_training = self.model.training
        was_training = self.model.training
        self.model.eval()

        idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
        n = min(max_samples, x.shape[0])
        if n <= 1:
            if was_training:
                self.model.train()
            return 0.0

        x = x[:n].to(self.device, dtype=self.dtype)
        y = y[:n].to(self.device, dtype=self.dtype)
        sdf = sdf[:n].to(self.device, dtype=self.dtype)
        bt = bt[:n].to(self.device, dtype=self.dtype) if bt.shape[1] > 0 else None
        params = [p for p in self.model.parameters() if p.requires_grad]
        rows = []

        for i in range(n):
            input_i = self.loss_computer.build_input_tensor(
                x[i:i + 1],
                y[i:i + 1],
                sdf[i:i + 1],
                bt[i:i + 1] if bt is not None else None
            )
            out_i = self.model(input_i).sum()
            grads = torch.autograd.grad(out_i, params, retain_graph=False, allow_unused=True)
            flat = [
                g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g, p in zip(grads, params)
            ]
            rows.append(torch.cat(flat).detach())

        if was_training:
            self.model.train()

        J = torch.stack(rows, dim=0).float()
        K = J @ J.t()
        eigvals = torch.linalg.eigvalsh(K)
        positive = eigvals[eigvals > 1e-12]
        if positive.numel() == 0:
            return float('inf')
        return (positive.max() / positive.min()).item()

    # ------------------------------------------------------------------
    # Получение предсказаний на всём датасете (для диагностики)
    # ------------------------------------------------------------------
    def get_predictions_on_dataset(self) -> Dict[str, np.ndarray]:
        """
        Возвращает предсказания модели на всём датасете.
        Учитывает hard constraints, если они используются.
        Возвращает словарь с ключами 'x', 'y', 'u', 'v', 'p' (физические единицы).
        """
        was_training = self.model.training
        self.model.eval()
        dataloader = self.data_handler.get_eval_dataloader()
        all_u = []
        all_v = []
        all_p = []
        all_x = []
        all_y = []

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = sdf.to(self.device, dtype=self.dtype)
                if bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None

                input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                out = self.model(input_tensor)
                u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

                # Применяем hard constraints, если нужно
                if self.loss_computer.bc_type == 'hard':
                    mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
                    mask_v = mask_v.to(self.device, dtype=torch.bool).view(-1, 1)
                    mask_p = mask_p.to(self.device, dtype=torch.bool).view(-1, 1)
                    u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    v_bc = v_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    p_bc = p_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    u, v, p = self.loss_computer.apply_hard_constraints(
                        u_raw, v_raw, p_raw, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc
                    )
                else:
                    u, v, p = u_raw, v_raw, p_raw
                u, v = self.loss_computer.apply_velocity_dirichlet_constraint(u, v, bt)

                all_u.append(u.cpu().numpy().flatten())
                all_v.append(v.cpu().numpy().flatten())
                all_p.append(p.cpu().numpy().flatten())
                all_x.append(x.cpu().numpy().flatten())
                all_y.append(y.cpu().numpy().flatten())

        # Конкатенация
        x_np = np.concatenate(all_x)
        y_np = np.concatenate(all_y)
        u_np = np.concatenate(all_u)
        v_np = np.concatenate(all_v)
        p_np = np.concatenate(all_p)

        # Переводим в физические единицы
        chord = self.data_handler.chord
        U_inf = self.data_handler.U_inf
        rho = self.data_handler.rho
        p_dyn = 0.5 * rho * U_inf**2

        if was_training:
            self.model.train()

        return {
            'x': x_np * chord,
            'y': y_np * chord,
            'u': u_np * U_inf,
            'v': v_np * U_inf,
            'p': p_np * p_dyn + self.data_handler.p_ref
        }

    # ------------------------------------------------------------------
    # Получение профиля скорости в заданном сечении x (УСТАРЕЛО)
    # ------------------------------------------------------------------
    def get_profile_at_x(self, x_target: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        ВНИМАНИЕ: Этот метод устарел и использует начальные поля (u_target).
        Для диагностики используйте data_handler.interpolate_profile с предсказаниями модели.
        """
        logger.warning("Метод get_profile_at_x устарел. Используйте data_handler.interpolate_profile с предсказаниями модели.")
        dataset = self.data_handler.dataset
        x_np = dataset.x.cpu().numpy().flatten()
        y_np = dataset.y.cpu().numpy().flatten()
        u_np = dataset.u_target.cpu().numpy().flatten()  # используем target как начальное приближение

        # Выбираем точки, близкие к x_target
        tol = 0.05 * (self.data_handler.x_max - self.data_handler.x_min)
        mask = np.abs(x_np - x_target) < tol
        if np.sum(mask) < 3:
            logger.warning(f"Мало точек в сечении x={x_target}, интерполяция может быть неточной.")
        y_vals = y_np[mask]
        u_vals = u_np[mask]

        # Сортируем по y
        sort_idx = np.argsort(y_vals)
        y_sorted = y_vals[sort_idx]
        u_sorted = u_vals[sort_idx]

        return y_sorted, u_sorted

    # ------------------------------------------------------------------
    # Вычисление спектра профиля (УСТАРЕЛО)
    # ------------------------------------------------------------------
    def compute_spectrum(self, y_vals: np.ndarray, u_vals: np.ndarray, spacing: Optional[float] = None) -> Dict:
        """
        Устаревший метод. Для диагностики используйте visualizer.plot_2d_spectrum.
        """
        logger.warning("Метод compute_spectrum устарел.")
        if len(y_vals) < 2:
            return {'freq': np.array([]), 'amp': np.array([])}
        # Обеспечиваем равномерную сетку: интерполируем u на равномерную сетку
        y_uniform = np.linspace(y_vals.min(), y_vals.max(), num=len(y_vals))
        u_uniform = np.interp(y_uniform, y_vals, u_vals)
        if spacing is None:
            spacing = y_uniform[1] - y_uniform[0]
        # БПФ
        n = len(u_uniform)
        fft_vals = np.fft.fft(u_uniform)
        freqs = np.fft.fftfreq(n, d=spacing)
        # Берём положительные частоты
        positive = freqs > 0
        return {
            'freq': freqs[positive],
            'amp': np.abs(fft_vals[positive])
        }

    # ------------------------------------------------------------------
    # Вычисление производной du/dy в пристеночных точках
    # ------------------------------------------------------------------
    def compute_wall_gradient(self, threshold: float = 0.1) -> float:
        """
        Вычисляет среднее значение ∂u/∂y в точках, близких к стенке (y < threshold).
        """
        dataset = self.data_handler.dataset
        # Предполагаем, что стенка – это точки с типом 'wall'
        if 'wall' not in self.data_handler.boundary_type_names:
            return 0.0
        wall_idx = self.data_handler.boundary_type_names.index('wall')
        bt = dataset.boundary_types.cpu().numpy()
        wall_mask = bt[:, wall_idx] > 0.5
        if not np.any(wall_mask):
            return 0.0
        # Берём точки вблизи стенки
        y_np = dataset.y.cpu().numpy().flatten()
        near_wall = (y_np < threshold) & wall_mask
        if np.sum(near_wall) == 0:
            return 0.0
        indices = np.where(near_wall)[0]
        self.model.eval()
        x_pts = dataset.x[indices].to(self.device, dtype=self.dtype)
        y_pts = dataset.y[indices].to(self.device, dtype=self.dtype).requires_grad_(True)
        sdf_pts = dataset.sdf[indices].to(self.device, dtype=self.dtype)
        bt_pts = dataset.boundary_types[indices].to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None
        input_tensor = self.loss_computer.build_input_tensor(x_pts, y_pts, sdf_pts, bt_pts)
        out = self.model(input_tensor)
        u_vals = out[:, 0:1]
        # Обнуляем накопленные градиенты перед вычислением du/dy
        self.model.zero_grad()
        if y_pts.grad is not None:
            y_pts.grad.zero_()
        u_vals.sum().backward(retain_graph=True)
        grad_u_y = y_pts.grad
        if grad_u_y is not None:
            mean_grad = grad_u_y.mean().item()
        else:
            mean_grad = 0.0
        self.model.zero_grad()
        return mean_grad

    def get_state(self) -> Dict[str, Any]:
        """Состояние Trainer, необходимое для корректного resume после RAR."""
        return {
            'point_residual': self.point_residual,
            'max_residual': self.max_residual,
            'point_pde_residual': self.point_pde_residual,
            'max_pde_residual': self.max_pde_residual,
            'rar_history': self.rar_history,
            'rar_used_sources': sorted(getattr(self, '_rar_used_sources', set())),
            'fvm_epoch_counter': self._fvm_epoch_counter,
            'last_fvm_loss': self._last_fvm_loss,
            'last_integral_loss': self._last_integral_loss,
            'rar_fvm_rebuild_counter': self._rar_fvm_rebuild_counter,
            'rar_new_points_since_fvm': self._rar_new_points_since_fvm,
            'rar_step_counter': self._rar_step_counter,
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает состояние Trainer из чекпоинта."""
        if not state:
            return

        n_points = self.data_handler.n_points

        def restore_tensor(name: str, current: torch.Tensor) -> torch.Tensor:
            value = state.get(name)
            if value is None:
                return current
            tensor = value.detach().cpu().to(dtype=self.dtype).flatten()
            if tensor.numel() == n_points:
                return tensor
            restored = torch.zeros(n_points, device='cpu', dtype=self.dtype)
            n = min(n_points, tensor.numel())
            restored[:n] = tensor[:n]
            logger.warning(
                "Trainer state '%s' имеет длину %d, а датасет %d; состояние усечено/дополнено.",
                name,
                tensor.numel(),
                n_points,
            )
            return restored

        self.point_residual = restore_tensor('point_residual', self.point_residual)
        self.max_residual = restore_tensor('max_residual', self.max_residual)
        self.point_pde_residual = restore_tensor('point_pde_residual', self.point_pde_residual)
        self.max_pde_residual = restore_tensor('max_pde_residual', self.max_pde_residual)
        self.rar_history = list(state.get('rar_history', self.rar_history) or [])
        self._rar_used_sources = set(int(v) for v in state.get('rar_used_sources', []))
        self._fvm_epoch_counter = int(state.get('fvm_epoch_counter', self._fvm_epoch_counter))
        self._last_fvm_loss = float(state.get('last_fvm_loss', self._last_fvm_loss))
        self._last_integral_loss = float(state.get('last_integral_loss', self._last_integral_loss))
        self._rar_fvm_rebuild_counter = int(state.get('rar_fvm_rebuild_counter', self._rar_fvm_rebuild_counter))
        self._rar_new_points_since_fvm = int(state.get('rar_new_points_since_fvm', self._rar_new_points_since_fvm))
        self._rar_step_counter = int(state.get('rar_step_counter', self._rar_step_counter))
        self._fvm_tensor_cache = None

    # ------------------------------------------------------------------
    # RAR с сохранением статистики и корректным расширением данных/весов
    # ------------------------------------------------------------------
    def apply_rar(self):
        """
        Adaptive RAR добавляет новые координаты в областях высокой PDE-невязки.

        Старый режим дублирования сохранён как rar_mode='legacy'. Он полезен
        только как перевзвешивание датасета и не меняет FVM-сетку.
        """
        if not self.rar_enable:
            return 0

        if self.rar_mode == 'legacy':
            return self._apply_rar_legacy()

        logger.info("Adaptive RAR: генерация новых точек через RARManager...")
        generated = self.rar_manager.generate(self.rar_num_points)
        if generated is None or generated.n_points == 0:
            logger.info("Adaptive RAR: новые точки не добавлены.")
            return 0

        x_new, y_new, sdf_new = generated.x, generated.y, generated.sdf
        added = self.data_handler.add_interior_points(x_new, y_new, sdf_new)
        if added == 0:
            logger.info("Adaptive RAR: DataHandler не добавил новых точек.")
            return 0

        new_residual = torch.zeros(added, device='cpu', dtype=self.dtype)
        self.point_residual = torch.cat([self.point_residual, new_residual], dim=0)
        self.max_residual = torch.cat([self.max_residual, torch.zeros_like(new_residual)], dim=0)
        self.point_pde_residual = torch.cat([self.point_pde_residual, torch.zeros_like(new_residual)], dim=0)
        self.max_pde_residual = torch.cat([self.max_pde_residual, torch.zeros_like(new_residual)], dim=0)
        self._extend_sa_weights(added)

        history_entry = {'interior': added}
        history_entry.update(generated.stats)
        self.rar_history.append(history_entry)
        self._rar_step_counter += 1
        self._rar_new_points_since_fvm += added
        self._maybe_rebuild_fvm_after_rar()
        self._log_rar_sdf_delta_statistics()

        logger.info(
            "Adaptive RAR: добавлено %d новых точек. Теперь всего точек: %d",
            added,
            self.data_handler.n_points,
        )
        return added

    def _apply_rar_legacy(self) -> int:
        """Старый RAR: дублирует точки с высокой невязкой без изменения координат."""
        logger.info("Legacy RAR: дублирование точек с наибольшей невязкой...")
        dataset = self.data_handler.dataset
        source_indices = getattr(dataset, 'source_indices', torch.arange(len(self.point_residual)))
        used_sources = getattr(self, '_rar_used_sources', set())
        source_list = source_indices.cpu().tolist()
        candidate_mask = torch.tensor(
            [src not in used_sources for src in source_list],
            dtype=torch.bool,
            device='cpu'
        )
        candidate_count = int(candidate_mask.sum().item())
        top_k = min(self.rar_num_points, candidate_count)
        if top_k == 0:
            logger.info("Legacy RAR: нет новых исходных точек для перевзвешивания.")
            return 0

        candidate_residual = self.point_residual.clone()
        candidate_residual[~candidate_mask] = -torch.inf
        _, indices = torch.topk(candidate_residual, top_k)
        indices_np = indices.numpy()
        selected_sources = source_indices[indices].cpu().tolist()
        self._rar_used_sources = used_sources | set(int(s) for s in selected_sources)

        if dataset.boundary_types is not None:
            bt = dataset.boundary_types[indices_np].cpu().numpy()
            counts = bt.sum(axis=0)
            stat = dict(zip(self.data_handler.boundary_type_names, counts))
            self.rar_history.append(stat)

        self.data_handler.add_points(indices.long())

        new_residual = torch.zeros(top_k, device='cpu', dtype=self.dtype)
        self.point_residual = torch.cat([self.point_residual, new_residual], dim=0)
        self.max_residual = torch.cat([self.max_residual, torch.zeros_like(new_residual)], dim=0)
        self.point_pde_residual = torch.cat([self.point_pde_residual, torch.zeros_like(new_residual)], dim=0)
        self.max_pde_residual = torch.cat([self.max_pde_residual, torch.zeros_like(new_residual)], dim=0)
        self._extend_sa_weights(top_k)
        logger.info("Legacy RAR: добавлено %d копий. FVM-сетка не перестраивается.", top_k)
        return top_k

    def _log_rar_sdf_delta_statistics(self) -> None:
        """Печатает SDF/delta статистику interior-точек после RAR."""
        try:
            ds = self.data_handler.dataset
            if ds.boundary_types is not None and 'interior' in ds.type_to_idx:
                mask = (ds.boundary_types[:, ds.type_to_idx['interior']] > 0.5).cpu().numpy()
            else:
                mask = ~getattr(ds, 'on_boundary', torch.zeros(ds.n_points, dtype=torch.bool)).cpu().numpy()
            if not np.any(mask):
                return
            chord = float(getattr(self.data_handler, 'chord', 1.0))
            re = float(getattr(self.data_handler, 'Re', 0.0))
            delta = getattr(self.data_handler, 'delta_bl_99_est', None)
            if delta is None:
                delta, _ = estimate_boundary_layer_delta(
                    chord,
                    re,
                    model=str(getattr(self.data_handler, 'boundary_layer_model', 'laminar')),
                )
            sdf_m = ds.sdf.cpu().numpy().reshape(-1)[mask].astype(np.float64) * chord
            summary = compute_sdf_summary(sdf_m, chord=chord, delta=delta)
            for line in format_sdf_summary(
                summary,
                title="SDF-статистика после RAR (interior -> WALL)",
            ):
                logger.info(line)
        except Exception as exc:
            logger.warning("Adaptive RAR: не удалось вывести SDF/delta статистику: %s", exc)

    def _extend_sa_weights(self, n_new: int) -> None:
        """Расширяет SA-веса после добавления новых точек."""
        if self.sa_weights is None or n_new <= 0:
            return
        old_weights = self.sa_weights.log_weights.detach().cpu()
        init_raw = float(getattr(self.sa_weights, 'initial_raw_value', 0.0))
        new_weights = torch.full((n_new, old_weights.shape[1]), init_raw, device='cpu')
        combined_weights = torch.cat([old_weights, new_weights], dim=0)
        self.sa_weights.log_weights = nn.Parameter(combined_weights.to(self.device))
        self.sa_weights_changed = True
        logger.warning("RAR: SA-веса расширены. Требуется пересоздание оптимизатора в solver.")

    def _generate_adaptive_rar_points(
        self,
        n_points: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Генерирует новые interior-точки около областей с высокой point_residual.

        Плотность задаётся вероятностью выбора source-точки:
        P(source_i) ∝ residual_i ** rar_residual_power.
        """
        if n_points <= 0 or len(self.point_residual) == 0:
            return None

        dh = self.data_handler
        ds = dh.dataset
        coords = np.column_stack([
            ds.x.cpu().numpy().reshape(-1),
            ds.y.cpu().numpy().reshape(-1),
        ]).astype(np.float64)
        residual = self.point_residual.detach().cpu().numpy().reshape(-1).astype(np.float64)

        if len(residual) != len(coords):
            logger.warning(
                "Adaptive RAR: размер residual (%d) не совпадает с датасетом (%d)",
                len(residual), len(coords),
            )
            return None

        if ds.boundary_types is not None and 'interior' in ds.type_to_idx:
            interior_mask = (ds.boundary_types[:, ds.type_to_idx['interior']] > 0.5).cpu().numpy()
        else:
            interior_mask = ~getattr(ds, 'on_boundary', torch.zeros(ds.n_points, dtype=torch.bool)).cpu().numpy()

        finite = np.isfinite(residual) & (residual > 0.0) & interior_mask
        if not np.any(finite):
            logger.info("Adaptive RAR: пока нет положительной PDE-невязки на interior-точках.")
            return None

        source_idx_all = np.where(finite)[0]
        source_res_all = residual[source_idx_all]
        q = min(max(self.rar_error_quantile, 0.0), 0.99)
        threshold = float(np.quantile(source_res_all, q))
        source_idx = source_idx_all[source_res_all >= threshold]
        source_res = residual[source_idx]
        if len(source_idx) == 0:
            return None

        weights = np.maximum(source_res, 0.0) + 1e-30
        weights = np.power(weights, max(self.rar_residual_power, 1e-6))
        weights = weights / weights.sum()

        existing_tree = cKDTree(coords)
        k = min(2, len(coords))
        nn_dist, _ = existing_tree.query(coords, k=k)
        if np.ndim(nn_dist) > 1:
            local_h = nn_dist[:, 1]
        else:
            local_h = np.full(len(coords), float(np.median(nn_dist)))
        positive_h = local_h[local_h > 1e-12]
        median_h = float(np.median(positive_h)) if len(positive_h) else 1e-3
        local_h = np.where(local_h > 1e-12, local_h, median_h)

        domain_norm = None
        if dh.mesh_domain_coords is not None:
            domain_norm = np.asarray(dh.mesh_domain_coords, dtype=np.float64) / float(dh.chord)
        foil_norm = None
        if dh.mesh_airfoil_coords is not None:
            foil_norm = np.asarray(dh.mesh_airfoil_coords, dtype=np.float64) / float(dh.chord)

        wall_tree = self._build_wall_tree(coords, ds)
        accepted: List[np.ndarray] = []
        accepted_sdf: List[float] = []
        accepted_arr = np.empty((0, 2), dtype=np.float64)
        max_attempts = max(n_points * self.rar_candidate_multiplier * 20, n_points)
        attempts = 0

        while len(accepted) < n_points and attempts < max_attempts:
            need = n_points - len(accepted)
            batch_n = min(max(need * self.rar_candidate_multiplier, 64), 4096)
            attempts += batch_n

            src = np.random.choice(source_idx, size=batch_n, replace=True, p=weights)
            radius = self.rar_jitter_scale * local_h[src] * np.sqrt(np.random.random(batch_n))
            theta = 2.0 * np.pi * np.random.random(batch_n)
            cand = coords[src] + np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

            valid = self._valid_interior_mask(cand, domain_norm, foil_norm)
            if not np.any(valid):
                continue
            cand = cand[valid]
            src = src[valid]

            min_dist = self.rar_min_distance_factor * local_h[src]
            dist_existing, _ = existing_tree.query(cand, k=1, workers=-1)
            valid = dist_existing >= min_dist
            if not np.any(valid):
                continue
            cand = cand[valid]
            min_dist = min_dist[valid]

            if len(accepted_arr) > 0:
                dist_accepted, _ = cKDTree(accepted_arr).query(cand, k=1, workers=-1)
                valid = dist_accepted >= min_dist
                if not np.any(valid):
                    continue
                cand = cand[valid]
                min_dist = min_dist[valid]

            if wall_tree is not None:
                sdf_vals, _ = wall_tree.query(cand, k=1, workers=-1)
                min_sdf = self.rar_min_boundary_distance_factor * median_h
                valid = sdf_vals >= min_sdf
                if not np.any(valid):
                    continue
                cand = cand[valid]
                min_dist = min_dist[valid]
                sdf_vals = sdf_vals[valid]
            else:
                sdf_vals = np.zeros(len(cand), dtype=np.float64)

            batch_pts: List[np.ndarray] = []
            batch_sdf: List[float] = []
            for xy, sdf, d_min in zip(cand, sdf_vals, min_dist):
                if len(accepted) + len(batch_pts) >= n_points:
                    break
                if batch_pts:
                    batch_arr = np.asarray(batch_pts, dtype=np.float64)
                    if float(np.linalg.norm(batch_arr - xy, axis=1).min()) < float(d_min):
                        continue
                batch_pts.append(np.asarray(xy, dtype=np.float64))
                batch_sdf.append(float(sdf))

            if batch_pts:
                accepted.extend(batch_pts)
                accepted_sdf.extend(batch_sdf)
                accepted_arr = np.vstack([accepted_arr, np.asarray(batch_pts, dtype=np.float64)])

        if not accepted:
            logger.warning("Adaptive RAR: не удалось сгенерировать допустимые новые точки.")
            return None

        pts = np.vstack(accepted)
        sdf = np.asarray(accepted_sdf, dtype=np.float64)
        if len(pts) < n_points:
            logger.warning(
                "Adaptive RAR: принято %d/%d точек (фильтры дубликатов/домена слишком строгие).",
                len(pts), n_points,
            )

        logger.info(
            "Adaptive RAR: source=%d точек выше %.0f%% квантиля, residual=[%.3e, %.3e], accepted=%d",
            len(source_idx),
            q * 100.0,
            float(source_res.min()),
            float(source_res.max()),
            len(pts),
        )
        return pts[:, 0], pts[:, 1], sdf

    @staticmethod
    def _build_wall_tree(coords: np.ndarray, dataset) -> Optional[cKDTree]:
        """KDTree по wall-точкам для быстрого SDF новых RAR-точек."""
        if dataset.boundary_types is None or 'wall' not in dataset.type_to_idx:
            return None
        wall_mask = (dataset.boundary_types[:, dataset.type_to_idx['wall']] > 0.5).cpu().numpy()
        if not np.any(wall_mask):
            return None
        return cKDTree(coords[wall_mask])

    @staticmethod
    def _valid_interior_mask(
        xy: np.ndarray,
        domain_norm: Optional[np.ndarray],
        foil_norm: Optional[np.ndarray],
    ) -> np.ndarray:
        """Фильтр: внутри домена и вне замкнутого профиля."""
        valid = np.isfinite(xy).all(axis=1)
        if domain_norm is not None and len(domain_norm) >= 3:
            try:
                from matplotlib.path import Path
                valid &= Path(domain_norm).contains_points(xy)
            except Exception:
                from shapely.geometry import Point, Polygon
                domain_poly = Polygon(domain_norm)
                if not domain_poly.is_valid:
                    domain_poly = domain_poly.buffer(0)
                valid &= np.array(
                    [domain_poly.contains(Point(float(p[0]), float(p[1]))) for p in xy],
                    dtype=bool,
                )

        if foil_norm is not None and len(foil_norm) >= 4 and np.allclose(foil_norm[0], foil_norm[-1]):
            try:
                from matplotlib.path import Path
                valid &= ~Path(foil_norm).contains_points(xy)
            except Exception:
                from shapely.geometry import Point, Polygon
                foil_poly = Polygon(foil_norm)
                if not foil_poly.is_valid:
                    foil_poly = foil_poly.buffer(0)
                valid &= ~np.array(
                    [foil_poly.contains(Point(float(p[0]), float(p[1]))) for p in xy],
                    dtype=bool,
                )
        return valid

    def _maybe_rebuild_fvm_after_rar(self) -> None:
        """Лениво перестраивает FVM-сетку после накопления новых RAR-точек."""
        if (
            not self.rar_rebuild_fvm
            or self.loss_computer.fvm is None
            or self._rar_new_points_since_fvm < self.rar_fvm_min_new_points
        ):
            return

        self._rar_fvm_rebuild_counter += 1
        if self._rar_fvm_rebuild_counter % self.rar_fvm_rebuild_every != 0:
            logger.info(
                "Adaptive RAR: FVM rebuild отложен (%d новых точек накоплено).",
                self._rar_new_points_since_fvm,
            )
            return

        try:
            logger.info(
                "Adaptive RAR: перестройка FVM-сетки по %d точкам...",
                self.data_handler.n_points,
            )
            mesh = self.data_handler.build_mesh_from_current_points(verbose=False)
            self.data_handler.set_mesh_from_mesh2d(mesh)
            self.loss_computer.fvm.rebuild_from_mesh(mesh)
            velocity_bc = torch.tensor(
                [
                    getattr(self.data_handler, 'inlet_u_norm', 1.0),
                    getattr(self.data_handler, 'inlet_v_norm', 0.0),
                ],
                dtype=self.dtype,
                device=self.device,
            )
            self.loss_computer.fvm.set_inlet_velocity(velocity_bc)
            try:
                from airfoil_pinn.physics.fvm import BC_CODES
                code = BC_CODES.get('velocity_dirichlet')
                if code is not None and bool((self.loss_computer.fvm.bnd_type == code).any()):
                    self.loss_computer.fvm.set_bc_velocity('velocity_dirichlet', velocity_bc)
            except Exception:
                logger.debug("Adaptive RAR: velocity_dirichlet не задан для FVM.", exc_info=True)
            self._fvm_tensor_cache = None
            self._rar_new_points_since_fvm = 0
            self._save_rar_mesh_volume_plot(mesh)
            logger.info("Adaptive RAR: FVM-сетка перестроена.")
        except Exception as exc:
            logger.warning("Adaptive RAR: не удалось перестроить FVM-сетку: %s", exc, exc_info=True)

    def _save_rar_mesh_volume_plot(self, mesh) -> None:
        """Сохраняет карту объёмов контрольных объёмов после RAR rebuild."""
        if not self.rar_save_mesh_volume_plot:
            return

        base_dir = self.plots_dir or PICTURES_DIR
        out_dir = os.path.join(base_dir, "rar_mesh")
        os.makedirs(out_dir, exist_ok=True)

        step = max(1, self._rar_step_counter)
        path = os.path.join(out_dir, f"rar_{step:04d}_mesh_cv_volume.png")
        try:
            mesh.plot_cv_volume_map(
                save_path=path,
                dpi=self.rar_mesh_plot_dpi,
                show=False,
                log_scale=True,
            )
            logger.info("Adaptive RAR: график CV-объёмов сетки сохранён: %s", path)
        except Exception as exc:
            logger.warning(
                "Adaptive RAR: не удалось сохранить график CV-объёмов сетки: %s",
                exc,
                exc_info=True,
            )

    def _get_fvm_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Возвращает FVM-тензоры на рабочем device/dtype, кэшируя перенос."""
        dh = self.data_handler
        if not hasattr(dh, 'fvm_x'):
            raise RuntimeError("FVM-тензоры не подготовлены")
        cache_key = (id(dh.fvm_x), self.device.type, self.device.index, self.dtype)
        if self._fvm_tensor_cache is None or self._fvm_tensor_cache[0] != cache_key:
            bt = dh.fvm_bt.to(self.device, dtype=self.dtype) if dh.fvm_bt is not None else None
            self._fvm_tensor_cache = (
                cache_key,
                dh.fvm_x.to(self.device, dtype=self.dtype),
                dh.fvm_y.to(self.device, dtype=self.dtype),
                dh.fvm_sdf.to(self.device, dtype=self.dtype),
                bt,
            )
        _, x_fvm, y_fvm, sdf_fvm, bt_fvm = self._fvm_tensor_cache
        return x_fvm, y_fvm, sdf_fvm, bt_fvm

    def _compute_global_integral_loss_from_dataset(self) -> torch.Tensor:
        """Fallback integral loss по полной выборке inlet/outlet, если FVM-сетки нет."""
        if not self.loss_computer.integral_condition:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        ds = self.data_handler.dataset
        if ds.boundary_types is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        names = self.data_handler.boundary_type_names
        if 'inlet' not in names or 'outlet' not in names:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        inlet_idx = ds.type_to_idx.get('inlet')
        outlet_idx = ds.type_to_idx.get('outlet')
        if inlet_idx is None or outlet_idx is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        mask = (ds.boundary_types[:, inlet_idx] > 0.5) | (ds.boundary_types[:, outlet_idx] > 0.5)
        if not bool(mask.any()):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        x = ds.x[mask].to(self.device, dtype=self.dtype)
        y = ds.y[mask].to(self.device, dtype=self.dtype)
        sdf = ds.sdf[mask].to(self.device, dtype=self.dtype)
        bt = ds.boundary_types[mask].to(self.device, dtype=self.dtype)

        inp = self.loss_computer.build_input_tensor(x, y, sdf, bt)
        out = self.model(inp)
        u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        if self.loss_computer.bc_type == 'hard':
            u, v, _ = self.loss_computer.apply_hard_constraints(
                u_raw,
                v_raw,
                p_raw,
                ds.mask_u_fixed[mask].to(self.device, dtype=torch.bool).view(-1, 1),
                ds.mask_v_fixed[mask].to(self.device, dtype=torch.bool).view(-1, 1),
                ds.mask_p_fixed[mask].to(self.device, dtype=torch.bool).view(-1, 1),
                ds.u_bc[mask].to(self.device, dtype=self.dtype).view(-1, 1),
                ds.v_bc[mask].to(self.device, dtype=self.dtype).view(-1, 1),
                ds.p_bc[mask].to(self.device, dtype=self.dtype).view(-1, 1),
            )
        else:
            u, v = u_raw, v_raw
        u, v = self.loss_computer.apply_velocity_dirichlet_constraint(u, v, bt)

        return self.loss_computer.compute_integral_loss(u, v, bt, x, y)

    # ------------------------------------------------------------------
    # FVM-шаг (полный датасет, отдельный от батч-цикла)
    # ------------------------------------------------------------------
    def _run_fvm_step(self) -> float:
        """
        УСТАРЕЛО. FVM-потеря теперь включается в total_loss последнего батча
        через параметр add_fvm=True в forward_pass (единый backward с PDE/BC).
        Этот метод сохранён для совместимости и не вызывается из train_epoch_adam.

        Returns:
            Значение FVM-потери после шага (float)
        """
        logger.warning("_run_fvm_step устарел: FVM интегрирован в forward_pass через add_fvm=True.")
        if self.loss_computer.fvm is None or self.loss_computer.fvm_weight == 0.0:
            return 0.0

        dh = self.data_handler
        if not hasattr(dh, 'fvm_x'):
            logger.warning("FVM step: fvm-тензоры не подготовлены (load_mesh не вызывался?), пропускаю.")
            return 0.0

        self.model.train()
        # Явная очистка градиентов перед FVM-проходом — предотвращает
        # случайное накопление с остатками предыдущих батчей.
        self.optimizer.zero_grad()

        # Используем координаты узлов сетки напрямую — не датасет.
        # Это корректно даже если датасет и сетка имеют разный размер.
        x_all, y_all, sdf_all, bt_all = self._get_fvm_tensors()

        input_all = self.loss_computer.build_input_tensor(x_all, y_all, sdf_all, bt_all)
        out_all   = self.model(input_all)          # (N_mesh, output_dim)
        u_fvm     = out_all[:, :2]                  # (N_mesh, 2) — u и v

        fvm_loss = self.loss_computer.compute_fvm_loss(u_fvm)

        fvm_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.sa_weights is not None:
                torch.nn.utils.clip_grad_norm_(self.sa_weights.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        val = fvm_loss.item()
        logger.debug("FVM step: loss=%.4e", val)
        return val

    def _scheduled_fvm_batches(self, n_batches: int, enabled: bool) -> set:
        """
        Возвращает индексы batch-ей, где запускается полный FVM-проход.

        batch_interval задаётся пользователем в one-based терминах:
        1 — каждый batch, 5 — batch 5, 10, 15, ...
        Если interval больше числа batch-ей, FVM запускается один раз на
        последнем batch-е эпохи.
        Каждый запуск потом масштабируется так, чтобы суммарный FVM-вклад за
        эпоху не становился больше заданного веса только из-за частоты запусков.
        """
        if not enabled or n_batches <= 0:
            return set()
        interval = max(1, int(self.fvm_batch_interval))
        batches = set(range(interval - 1, n_batches, interval))
        if not batches:
            batches.add(n_batches - 1)
        return batches

    # ------------------------------------------------------------------
    # Шаг обучения (Adam)
    # ------------------------------------------------------------------
    def train_batch_adam(
        self,
        batch: Tuple,
        accum_step: bool,
        pde_mult: float = 1.0,
        add_fvm: bool = False,
        fvm_scale: float = 1.0,
        add_integral: bool = False,
        integral_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Выполняет один прямой проход и накопление градиентов для батча (Adam).
        Если accum_step=True, выполняет шаг оптимизации.

        Args:
            batch: кортеж данных
            accum_step: флаг, указывающий, нужно ли выполнять шаг оптимизации
            pde_mult: множитель для PDE потерь
            add_fvm: если True, добавляет FVM-потерю (полный проход по сетке) к total_loss

        Returns:
            словарь с результатами (те же поля, что и в forward_pass, плюс grad_norm)
        """
        with self._autocast_context():
            fwd = self.forward_pass(
                batch,
                pde_mult,
                add_fvm=add_fvm,
                fvm_scale=fvm_scale,
                add_integral=add_integral,
                integral_scale=integral_scale,
            )
            total_loss = fwd['total'] / self.accum_steps

        # Обратное распространение
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Вычисляем норму градиентов (до клиппинга и шага)
        grad_norm = 0.0
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        if self.sa_weights is not None:
            for p in self.sa_weights.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        # Если пора делать шаг оптимизации
        if accum_step:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.sa_weights is not None:
                        torch.nn.utils.clip_grad_norm_(self.sa_weights.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.sa_weights is not None:
                        torch.nn.utils.clip_grad_norm_(self.sa_weights.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return {
            'total': fwd['total'].item(),
            'pde_raw': fwd['pde_raw'],
            'pde_max': fwd['pde_max'],
            'bc_comp': fwd['bc_comp'],
            'extra_grad': fwd['extra_grad'],
            'integral_loss': fwd['integral_loss'],
            'fvm_loss': fwd.get('fvm_loss', 0.0),
            'wall_speed': fwd['wall_speed'],
            'grad_norm': grad_norm
        }

    # ------------------------------------------------------------------
    # Эпоха обучения (Adam)
    # ------------------------------------------------------------------
    def train_epoch_adam(self, pde_mult: float = 1.0) -> Tuple[
        float,
        Dict[str, float],
        Dict[str, float],
        float,
        float,
        float,
        Dict[str, float],
        Dict[str, float],
        float
    ]:
        """
        Выполняет одну эпоху обучения с оптимизатором Adam.

        Returns:
            кортеж:
                avg_loss (float): средняя полная потеря за эпоху
                avg_pde (Dict[str, float]): средние значения PDE по компонентам
                avg_bc (Dict[str, float]): средние значения BC по компонентам
                avg_extra_grad (float): средняя дополнительная потеря от градиентов
                avg_integral (float): средняя интегральная потеря
                avg_wall_speed (float): средняя скорость на стенке
                avg_max (Dict[str, float]): средние максимумы PDE
                pde_rate (Dict[str, float]): отношение текущих потерь к предыдущим (для отчёта)
                avg_grad_norm (float): средняя норма градиентов
        """
        self.model.train()
        dataloader = self.data_handler.get_dataloader()

        # Инициализация накопителей
        epoch_losses = {
            'total': 0.0,
            'pde_raw': {c: 0.0 for c in self.loss_computer.pde_components},
            'pde_max': {c: 0.0 for c in self.loss_computer.pde_components},
            'bc_comp': {b: 0.0 for b in self.data_handler.boundary_type_names} if self.loss_computer.bc_type == 'soft' else {},
            'extra_grad': 0.0,
            'integral_loss': 0.0,
            'wall_speed_sum': 0.0,
            'n_wall_batches': 0,
            'grad_norm_sum': 0.0,
            'grad_norm_count': 0,
            'fvm_loss_sum': 0.0,
            'fvm_loss_count': 0,
        }
        n_batches = 0
        n_total_batches = len(dataloader)

        # FVM добавляется по пользовательскому batch_interval, но нормируется
        # по числу запусков, чтобы эффективный вес за эпоху оставался тем же.
        self._fvm_epoch_counter += 1
        fvm_this_epoch = (
            self.loss_computer.fvm is not None
            and self.loss_computer.fvm_weight > 0.0
            and self._fvm_epoch_counter % self.fvm_interval == 0
        )
        fvm_batches = self._scheduled_fvm_batches(n_total_batches, fvm_this_epoch)
        fvm_scale = (float(self.accum_steps) / len(fvm_batches)) if fvm_batches else 1.0

        integral_this_epoch = (
            self.loss_computer.integral_condition
            and self.loss_computer.integral_weight != 0.0
        )
        integral_batches = {n_total_batches - 1} if integral_this_epoch and n_total_batches > 0 else set()
        integral_scale = float(self.accum_steps) if integral_batches else 1.0
        self._last_integral_loss = 0.0

        for i, batch in enumerate(dataloader):
            is_last_batch = (i == n_total_batches - 1)
            accum_step = ((i + 1) % self.accum_steps == 0) or is_last_batch
            add_fvm = i in fvm_batches
            add_integral = i in integral_batches
            batch_losses = self.train_batch_adam(
                batch,
                accum_step,
                pde_mult,
                add_fvm=add_fvm,
                fvm_scale=fvm_scale,
                add_integral=add_integral,
                integral_scale=integral_scale,
            )

            epoch_losses['total'] += batch_losses['total']
            for k, v in batch_losses['pde_raw'].items():
                epoch_losses['pde_raw'][k] += v
            for k, v in batch_losses['pde_max'].items():
                epoch_losses['pde_max'][k] += v
            if self.loss_computer.bc_type == 'soft':
                for k, v in batch_losses['bc_comp'].items():
                    epoch_losses['bc_comp'][k] += v
                epoch_losses['extra_grad'] += batch_losses['extra_grad']
            epoch_losses['integral_loss'] += batch_losses['integral_loss']
            if batch_losses['wall_speed'] is not None:
                epoch_losses['wall_speed_sum'] += batch_losses['wall_speed']
                epoch_losses['n_wall_batches'] += 1
            if batch_losses['grad_norm'] > 0:
                epoch_losses['grad_norm_sum'] += batch_losses['grad_norm']
                epoch_losses['grad_norm_count'] += 1
            if add_fvm:
                epoch_losses['fvm_loss_sum'] += batch_losses.get('fvm_loss', 0.0)
                epoch_losses['fvm_loss_count'] += 1
            if add_integral:
                self._last_integral_loss = batch_losses.get('integral_loss', 0.0)
            n_batches += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  Batch {i+1}/{len(dataloader)} | текущая loss: {batch_losses['total']:.4e}")

        # Усреднение
        avg_loss = epoch_losses['total'] / n_batches
        avg_pde = {k: v / n_batches for k, v in epoch_losses['pde_raw'].items()}
        avg_max = {k: v / n_batches for k, v in epoch_losses['pde_max'].items()}
        if self.loss_computer.bc_type == 'soft':
            avg_bc = {k: v / n_batches for k, v in epoch_losses['bc_comp'].items()}
            avg_extra_grad = epoch_losses['extra_grad'] / n_batches
        else:
            avg_bc = {}
            avg_extra_grad = 0.0
        avg_integral = self._last_integral_loss if self.loss_computer.integral_condition else 0.0
        avg_wall_speed = (epoch_losses['wall_speed_sum'] / epoch_losses['n_wall_batches']
                          if epoch_losses['n_wall_batches'] > 0 else 0.0)
        avg_grad_norm = epoch_losses['grad_norm_sum'] / epoch_losses['grad_norm_count'] if epoch_losses['grad_norm_count'] > 0 else 0.0

        # Обновление скользящих средних для адаптивных весов
        for comp, val in avg_pde.items():
            self.loss_ma[comp].append(val)
        if self.loss_computer.bc_type == 'soft':
            for comp, val in avg_bc.items():
                self.loss_ma[comp].append(val)

        # Вычисление rate (отношение к предыдущему значению) – для отчёта
        pde_rate = {c: 1.0 for c in self.loss_computer.pde_components}

        if epoch_losses['fvm_loss_count'] > 0:
            self._last_fvm_loss = epoch_losses['fvm_loss_sum'] / epoch_losses['fvm_loss_count']
        else:
            self._last_fvm_loss = 0.0

        return avg_loss, avg_pde, avg_bc, avg_extra_grad, avg_integral, avg_wall_speed, avg_max, pde_rate, avg_grad_norm

    # ------------------------------------------------------------------
    # Эпоха обучения (L-BFGS)
    # ------------------------------------------------------------------
    def train_epoch_lbfgs(self) -> Tuple[
        float,
        Dict[str, float],
        Dict[str, float],
        float,
        float,
        float,
        Dict[str, float],
        Dict[str, float],
        float
    ]:
        """
        Выполняет одну эпоху обучения с оптимизатором L-BFGS (один шаг на всём датасете).

        Returns:
            аналогично train_epoch_adam, но grad_norm = 0.0 (не вычисляется)
        """
        self.model.train()
        dataloader = self.data_handler.get_dataloader()

        pde_raw_list = []
        pde_max_list = []
        bc_comp_list = []
        integral_loss_list = []
        fvm_loss_list = []
        wall_speed_list = []
        n_batches = 0

        # FVM добавляется прямо в closure — единый backward с PDE/BC
        self._fvm_epoch_counter += 1
        fvm_this_epoch = (
            self.loss_computer.fvm is not None
            and self.loss_computer.fvm_weight > 0.0
            and self._fvm_epoch_counter % self.fvm_interval == 0
        )

        def closure():
            self.optimizer.zero_grad()
            total_loss = 0.0
            nonlocal n_batches
            pde_raw_list.clear()
            pde_max_list.clear()
            bc_comp_list.clear()
            integral_loss_list.clear()
            fvm_loss_list.clear()
            wall_speed_list.clear()
            n_batches = 0
            batches_list = list(dataloader)
            fvm_batches = self._scheduled_fvm_batches(len(batches_list), fvm_this_epoch)
            fvm_scale = (1.0 / len(fvm_batches)) if fvm_batches else 1.0
            integral_batches = (
                {len(batches_list) - 1}
                if self.loss_computer.integral_condition
                and self.loss_computer.integral_weight != 0.0
                and len(batches_list) > 0
                else set()
            )
            self._last_integral_loss = 0.0
            for k, batch in enumerate(batches_list):
                add_fvm_b = k in fvm_batches
                add_integral_b = k in integral_batches
                fwd = self.forward_pass(
                    batch,
                    pde_mult=1.0,
                    add_fvm=add_fvm_b,
                    fvm_scale=fvm_scale,
                    add_integral=add_integral_b,
                    integral_scale=1.0,
                )
                total_loss += fwd['total']

                pde_raw_list.append(fwd['pde_raw'])
                pde_max_list.append(fwd['pde_max'])
                if self.loss_computer.bc_type == 'soft':
                    bc_comp_list.append(fwd['bc_comp'])
                if add_integral_b:
                    integral_loss_list.append(fwd['integral_loss'])
                    self._last_integral_loss = fwd['integral_loss']
                if fwd['wall_speed'] is not None:
                    wall_speed_list.append(fwd['wall_speed'])
                if add_fvm_b and fwd.get('fvm_loss', 0.0) > 0:
                    fvm_loss_list.append(fwd.get('fvm_loss', 0.0))
                n_batches += 1

            total_loss.backward()
            return total_loss

        loss = self.optimizer.step(closure)

        # Усреднение результатов
        avg_pde = {}
        avg_max = {}
        for key in self.loss_computer.pde_components:
            vals_pde = [d[key] for d in pde_raw_list]
            vals_max = [d[key] for d in pde_max_list]
            avg_pde[key] = np.mean(vals_pde) if vals_pde else 0.0
            avg_max[key] = np.mean(vals_max) if vals_max else 0.0

        if self.loss_computer.bc_type == 'soft':
            avg_bc = {}
            for key in self.data_handler.boundary_type_names:
                vals = [d.get(key, 0.0) for d in bc_comp_list]
                avg_bc[key] = np.mean(vals) if vals else 0.0
        else:
            avg_bc = {}

        avg_integral = np.mean(integral_loss_list) if integral_loss_list else 0.0
        avg_wall_speed = np.mean(wall_speed_list) if wall_speed_list else 0.0
        avg_loss = loss.item() if torch.is_tensor(loss) else loss
        pde_rate = {c: 1.0 for c in self.loss_computer.pde_components}
        avg_extra_grad = 0.0  # не вычисляется для L-BFGS

        if fvm_loss_list:
            self._last_fvm_loss = float(np.mean(fvm_loss_list))
        else:
            self._last_fvm_loss = 0.0

        return avg_loss, avg_pde, avg_bc, avg_extra_grad, avg_integral, avg_wall_speed, avg_max, pde_rate, 0.0

    # ------------------------------------------------------------------
    # Адаптивная балансировка весов BC (через LossComputer)
    # ------------------------------------------------------------------
    def update_adaptive_weights(self, loss_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Обновляет веса граничных условий на основе скользящих средних потерь.
        Использует метод LossComputer.update_adaptive_weights для инкапсуляции.

        Args:
            loss_weights: текущие веса (словарь)

        Returns:
            обновлённый словарь весов
        """
        if not self.adaptive_weights:
            return loss_weights

        # Вычисляем среднее значение PDE потерь
        pde_vals = []
        for comp in self.loss_computer.pde_components:
            if self.loss_ma[comp]:
                pde_vals.append(np.mean(list(self.loss_ma[comp])))
        if not pde_vals:
            return loss_weights

        avg_pde = np.mean(pde_vals)
        target = avg_pde * np.mean([loss_weights[comp] for comp in self.loss_computer.pde_components])
        eps = 1e-8
        new_weights = {}
        for key in self.data_handler.boundary_type_names:
            if key in self.loss_ma and self.loss_ma[key]:
                avg_bc = np.mean(list(self.loss_ma[key]))
                new_weights[key] = target / (avg_bc + eps)
                new_weights[key] = np.clip(new_weights[key], 0.1, 10000.0)
            else:
                new_weights[key] = loss_weights.get(key, 1.0)

        # Передаём обновление в LossComputer
        self.loss_computer.update_adaptive_weights(new_weights)
        logger.info(f"Адаптивные веса BC обновлены: { {k: f'{v:.2f}' for k, v in new_weights.items()} }")
        return self.loss_computer.loss_weights  # возвращаем обновлённый словарь

    # ------------------------------------------------------------------
    # Сброс оптимизатора с шумом
    # ------------------------------------------------------------------
    def reset_optimizer(self, reset_noise: Optional[float] = None):
        """
        Добавляет шум к параметрам модели и (опционально) SA-весов,
        после чего пересоздаёт оптимизатор.

        Args:
            reset_noise: амплитуда шума (если None, используется self.reset_noise)
        """
        if self.using_lbfgs:
            logger.info("Сброс оптимизатора недоступен в режиме L-BFGS.")
            return

        noise = reset_noise if reset_noise is not None else self.reset_noise
        logger.info(f"Перезапуск оптимизатора с шумом {noise}")

        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * noise)
            if self.sa_weights is not None:
                for param in self.sa_weights.parameters():
                    param.add_(torch.randn_like(param) * noise)

        # Примечание: оптимизатор должен быть пересоздан вызывающим кодом,
        # так как здесь нет доступа к конфигурации для создания нового.
        # Возвращаем управление, чтобы solver мог пересоздать оптимизатор.
        logger.info("Шум добавлен. Требуется пересоздание оптимизатора.")
