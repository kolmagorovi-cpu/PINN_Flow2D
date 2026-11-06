#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль metrics для проекта PINN Learning.
Содержит класс MetricsCollector, отвечающий за сбор, вывод и сохранение метрик обучения,
включая расширенный набор диагностических метрик.

Исправленная версия:
- Уточнены комментарии о размерности возвращаемых значений.
- Добавлена защита от отсутствия ключей в истории.
- Метод get_boundary_avg_velocities теперь явно возвращает нормированные значения.
- Добавлен метод get_boundary_avg_velocities_physical для получения физических скоростей.
"""

import os
import logging
import datetime
import sys
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """
    Сбор, вывод в консоль, логирование и финальный расчёт метрик обучения.
    Хранит историю обучения и предоставляет методы для её обновления и визуализации в текстовом виде.
    """

    def __init__(
        self,
        model: nn.Module,
        data_handler,
        loss_computer,
        config: Dict,
        device: torch.device,
        dtype: torch.dtype,
        logs_dir: str,
        config_name: str = "unnamed",
        reference_data: Optional[Dict] = None
    ):
        """
        Args:
            model: нейросетевая модель
            data_handler: объект DataHandler (с метаданными, датасетом)
            loss_computer: объект LossComputer (знает компоненты PDE, имена границ)
            config: полная конфигурация (для доступа к настройкам вывода и диагностики)
            device: устройство для вычислений
            dtype: тип данных (torch.float32, torch.float64)
            logs_dir: директория для сохранения логов
            config_name: имя конфигурации (для заголовков)
            reference_data: словарь с эталонным решением (опционально)
        """
        self.model = model
        self.data_handler = data_handler
        self.loss_computer = loss_computer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.logs_dir = logs_dir
        self.config_name = config_name
        self.reference_data = reference_data
        diagnostics_config = config.get("diagnostics", {}) or {}
        section_flow_config = (
            diagnostics_config.get("section_flow", {}) or {}
        )
        self.section_flow_enabled = bool(
            section_flow_config.get("enabled", True)
        )
        self.section_flow_sections = tuple(
            float(value)
            for value in section_flow_config.get(
                "sections",
                diagnostics_config.get("profile_sections", ()),
            )
        )

        # История обучения (расширенная)
        self.history = {
            'loss': [],
            # Compatibility keys ``pde``/``bc`` contain MSE values.
            'pde': {c: [] for c in loss_computer.pde_components},
            'pde_mse': {c: [] for c in loss_computer.pde_components},
            'pde_rms': {c: [] for c in loss_computer.pde_components},
            'bc': {b: [] for b in data_handler.boundary_type_names} if loss_computer.bc_type == 'soft' else {},
            'bc_mse': {b: [] for b in data_handler.boundary_type_names} if loss_computer.bc_type == 'soft' else {},
            'bc_rms': {b: [] for b in data_handler.boundary_type_names} if loss_computer.bc_type == 'soft' else {},
            'wall_speed': [],
            'sa_weights_mean': [] if loss_computer.use_sa_weights else None,
            'integral_loss': [] if loss_computer.integral_condition else None,
            'integral_mse': [] if loss_computer.integral_condition else None,
            'integral_rms': [] if loss_computer.integral_condition else None,
            'max_res': {c: [] for c in loss_computer.pde_components},
            'max_abs_res': {c: [] for c in loss_computer.pde_components},
            'rate': {c: [] for c in loss_computer.pde_components},
            'grad_norm': [],
            'total_bc': [] if loss_computer.bc_type == 'soft' else None,
            'total_pde': [],
            'fvm_loss': [],
            'fvm_raw_loss': [],
            'fvm_mse': [],
            'fvm_weighted_loss': [],
            'fvm_rms': [],
            'fvm_evaluations': [],
            # Новые поля для диагностики
            'grad_norm_pde': [],
            'grad_norm_bc': [],
            'grad_cosine': [],
            'mass_balance_error': [],
            'mass_flux_in': [],
            'mass_flux_out': [],
            'max_v_top': [],
            'avg_u_out': [],
            'avg_u_in': [],
            'wall_gradient': [],
            'ntk_condition': [],
            'ntk_diagnostics': [],
        }

        # Для кэширования средних скоростей на границах (нормированные)
        self._last_avg_epoch = -1
        self._cached_u_in_norm = 0.0
        self._cached_u_out_norm = 0.0
        self._last_mass_epoch = -1
        self._cached_mass_metrics = {
            'mass_flux_in': 0.0,
            'mass_flux_out': 0.0,
            'mass_balance_error': 0.0,
        }

        # Файл лога
        self.log_file = None

        os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Обновление истории
    # ------------------------------------------------------------------
    def update_history(
        self,
        loss: float,
        pde_raw: Dict[str, float],
        bc_comp: Dict[str, float],
        integral_loss: float,
        wall_speed: float,
        pde_max: Dict[str, float],
        pde_rate: Dict[str, float],
        grad_norm: float,
        extra_diag: Optional[Dict] = None
    ) -> None:
        """
        Добавляет метрики текущей эпохи в историю.

        Args:
            loss: полная потеря
            pde_raw: словарь MSE PDE-невязок
            bc_comp: словарь MSE граничных невязок
            integral_loss: интегральная потеря
            wall_speed: средняя скорость на стенке
            pde_max: словарь максимальных квадратов PDE-невязок
            pde_rate: отношение текущих потерь к предыдущим
            grad_norm: норма градиентов
            extra_diag: словарь с дополнительными диагностическими метриками (опционально)
        """
        self._ensure_history_schema()
        self.history['loss'].append(loss)
        self.history['convergence_loss'].append(self.convergence_metric(pde_raw, bc_comp))
        if extra_diag and 'sa_diagnostics' in extra_diag:
            self.history['sa_diagnostics'].append(extra_diag['sa_diagnostics'])
        for k, v in pde_raw.items():
            self.history['pde'][k].append(v)
            self.history['pde_mse'][k].append(v)
            self.history['pde_rms'][k].append(
                float(np.sqrt(max(float(v), 0.0)))
            )
        if self.loss_computer.bc_type == 'soft':
            for k, v in bc_comp.items():
                self.history['bc'][k].append(v)
                self.history['bc_mse'][k].append(v)
                self.history['bc_rms'][k].append(
                    float(np.sqrt(max(float(v), 0.0)))
                )
        self.history['wall_speed'].append(wall_speed)
        if self.history['sa_weights_mean'] is not None:
            if extra_diag and 'sa_weights_mean' in extra_diag:
                self.history['sa_weights_mean'].append(extra_diag['sa_weights_mean'])
        if self.history['integral_loss'] is not None:
            self.history['integral_loss'].append(integral_loss)
        if self.history['integral_mse'] is not None:
            self.history['integral_mse'].append(integral_loss)
        if self.history['integral_rms'] is not None:
            self.history['integral_rms'].append(
                float(np.sqrt(max(float(integral_loss), 0.0)))
            )
        for k, v in pde_max.items():
            self.history['max_res'][k].append(v)
            self.history['max_abs_res'][k].append(
                float(np.sqrt(max(float(v), 0.0)))
            )
        for k, v in pde_rate.items():
            self.history['rate'][k].append(v)
        self.history['grad_norm'].append(grad_norm)

        total_pde = sum(pde_raw.values())
        self.history['total_pde'].append(total_pde)
        if self.loss_computer.bc_type == 'soft':
            total_bc = sum(bc_comp.values())
            self.history['total_bc'].append(total_bc)

        # Добавляем дополнительные метрики, если они есть
        if extra_diag:
            if 'fvm_loss' in extra_diag:
                self.history['fvm_loss'].append(extra_diag['fvm_loss'])
            if 'fvm_raw_loss' in extra_diag:
                self.history['fvm_raw_loss'].append(
                    extra_diag['fvm_raw_loss']
                )
            if 'fvm_mse' in extra_diag:
                self.history['fvm_mse'].append(extra_diag['fvm_mse'])
            if 'fvm_weighted_loss' in extra_diag:
                self.history['fvm_weighted_loss'].append(
                    extra_diag['fvm_weighted_loss']
                )
            if 'fvm_rms' in extra_diag:
                self.history['fvm_rms'].append(extra_diag['fvm_rms'])
            if 'fvm_evaluations' in extra_diag:
                self.history['fvm_evaluations'].append(
                    extra_diag['fvm_evaluations']
                )
            if 'grad_norm_pde' in extra_diag:
                self.history['grad_norm_pde'].append(extra_diag['grad_norm_pde'])
            if 'grad_norm_bc' in extra_diag:
                self.history['grad_norm_bc'].append(extra_diag['grad_norm_bc'])
            if 'grad_cosine' in extra_diag:
                self.history['grad_cosine'].append(extra_diag['grad_cosine'])
            if 'mass_balance_error' in extra_diag:
                self.history['mass_balance_error'].append(extra_diag['mass_balance_error'])
            if 'mass_flux_in' in extra_diag:
                self.history['mass_flux_in'].append(extra_diag['mass_flux_in'])
            if 'mass_flux_out' in extra_diag:
                self.history['mass_flux_out'].append(extra_diag['mass_flux_out'])
            if 'max_v_top' in extra_diag:
                self.history['max_v_top'].append(extra_diag['max_v_top'])
            if 'avg_u_in' in extra_diag:
                self.history['avg_u_in'].append(extra_diag['avg_u_in'])
            if 'avg_u_out' in extra_diag:
                self.history['avg_u_out'].append(extra_diag['avg_u_out'])
            if 'wall_gradient' in extra_diag:
                self.history['wall_gradient'].append(extra_diag['wall_gradient'])
            if 'ntk_condition' in extra_diag:
                self.history['ntk_condition'].append(extra_diag['ntk_condition'])
            if 'ntk_diagnostics' in extra_diag:
                self.history['ntk_diagnostics'].append(extra_diag['ntk_diagnostics'])

    @staticmethod
    def convergence_metric(pde_mse: Dict[str, float], bc_mse: Dict[str, float]) -> float:
        """Sum point-count-normalized physical MSEs without loss weights.

        Auxiliary integral/FVM objectives are excluded: their intermittent
        evaluation must not change the convergence criterion between epochs.
        """
        return float(sum(pde_mse.values()) + sum(bc_mse.values()))

    def _ensure_history_schema(self) -> None:
        """Добавляет новые ключи при resume из старых чекпоинтов."""
        if 'convergence_loss' not in self.history:
            # Old histories already contain unweighted component MSEs.
            components = [
                values for group in ('pde', 'bc')
                for values in (self.history.get(group) or {}).values()
            ]
            self.history['convergence_loss'] = [
                sum(values[index] for values in components if index < len(values))
                if components else float('nan')
                for index in range(len(self.history.get('loss', [])))
            ]
        defaults = {
            'sa_diagnostics': [],
            'fvm_loss': [],
            'fvm_raw_loss': [],
            'fvm_mse': [],
            'fvm_weighted_loss': [],
            'fvm_rms': [],
            'fvm_evaluations': [],
            'grad_norm_pde': [],
            'grad_norm_bc': [],
            'grad_cosine': [],
            'mass_balance_error': [],
            'mass_flux_in': [],
            'mass_flux_out': [],
            'max_v_top': [],
            'avg_u_out': [],
            'avg_u_in': [],
            'wall_gradient': [],
            'ntk_condition': [],
            'ntk_diagnostics': [],
            'grad_norm': [],
            'total_pde': [],
            'max_res': {c: [] for c in self.loss_computer.pde_components},
            'max_abs_res': {
                c: [] for c in self.loss_computer.pde_components
            },
            'rate': {c: [] for c in self.loss_computer.pde_components},
            'pde_mse': {
                c: [] for c in self.loss_computer.pde_components
            },
            'pde_rms': {
                c: [] for c in self.loss_computer.pde_components
            },
        }
        for key, value in defaults.items():
            if key not in self.history:
                self.history[key] = value
        if self.loss_computer.bc_type == 'soft' and 'bc' not in self.history:
            self.history['bc'] = {b: [] for b in self.data_handler.boundary_type_names}
        if self.loss_computer.bc_type == 'soft' and 'bc_mse' not in self.history:
            self.history['bc_mse'] = {
                b: [] for b in self.data_handler.boundary_type_names
            }
        if self.loss_computer.bc_type == 'soft' and 'bc_rms' not in self.history:
            self.history['bc_rms'] = {
                b: [] for b in self.data_handler.boundary_type_names
            }
        if self.loss_computer.bc_type == 'soft' and 'total_bc' not in self.history:
            self.history['total_bc'] = []
        if 'integral_loss' not in self.history:
            self.history['integral_loss'] = [] if self.loss_computer.integral_condition else None
        elif self.loss_computer.integral_condition and self.history.get('integral_loss') is None:
            self.history['integral_loss'] = []
        if 'integral_mse' not in self.history:
            self.history['integral_mse'] = (
                [] if self.loss_computer.integral_condition else None
            )
        elif (
            self.loss_computer.integral_condition
            and self.history.get('integral_mse') is None
        ):
            self.history['integral_mse'] = []
        if 'integral_rms' not in self.history:
            self.history['integral_rms'] = (
                [] if self.loss_computer.integral_condition else None
            )
        elif (
            self.loss_computer.integral_condition
            and self.history.get('integral_rms') is None
        ):
            self.history['integral_rms'] = []

    # ------------------------------------------------------------------
    # Вычисление средней скорости на входе/выходе (с кэшированием)
    # ------------------------------------------------------------------
    def get_boundary_avg_velocities(self, epoch: int, interval: int = 1) -> Tuple[float, float]:
        """
        Возвращает средние значения u на inlet и outlet (НОРМИРОВАННЫЕ).
        Вычисление происходит только если epoch кратно interval и эпоха изменилась.

        Args:
            epoch: текущий номер эпохи
            interval: интервал обновления (по умолчанию 1 – обновляется каждую эпоху)

        Returns:
            (avg_u_in_norm, avg_u_out_norm) – средние скорости на входе и выходе (нормированные).
        """
        if epoch % interval == 0 and epoch != self._last_avg_epoch:
            self._update_boundary_avg_velocities()
            self._last_avg_epoch = epoch
        return self._cached_u_in_norm, self._cached_u_out_norm

    def get_boundary_avg_velocities_physical(self, epoch: int, interval: int = 1) -> Tuple[float, float]:
        """
        Возвращает средние значения u на inlet и outlet в физических единицах (м/с).
        """
        u_in_norm, u_out_norm = self.get_boundary_avg_velocities(epoch, interval)
        return u_in_norm * self.data_handler.U_inf, u_out_norm * self.data_handler.U_inf

    def invalidate_boundary_cache(self) -> None:
        """Сбрасывает кэши boundary diagnostics после изменения датасета/RAR."""
        self._last_avg_epoch = -1
        self._last_mass_epoch = -1

    def get_mass_flow_metrics(
        self,
        epoch: int,
        interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Возвращает интегральный баланс массы через inlet/outlet.

        В приоритете используются FVM boundary faces с векторами площадей.
        Если сетки нет, используется fallback-интеграл по полному набору
        boundary-точек, а не среднее u.
        """
        if epoch % interval == 0 and epoch != self._last_mass_epoch:
            self._cached_mass_metrics = self._compute_mass_flow_metrics()
            self._last_mass_epoch = epoch
        return dict(self._cached_mass_metrics)

    def _compute_mass_flow_metrics(self) -> Dict[str, Any]:
        result = self._compute_mass_flow_from_fvm_faces()
        if result is None:
            result = self._compute_mass_flow_from_boundary_points()
        return result

    def _predict_velocity_on_fvm_nodes(
        self,
        case_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Predict dimensionless FVM-node velocity for one operating point."""
        if self.loss_computer.fvm is None or not hasattr(self.data_handler, 'fvm_x'):
            return None

        x = self.data_handler.fvm_x.to(self.device, dtype=self.dtype)
        y = self.data_handler.fvm_y.to(self.device, dtype=self.dtype)
        sdf = self.loss_computer.evaluate_sdf_tensor(x, y)
        bt = (
            self.data_handler.fvm_bt.to(self.device, dtype=self.dtype)
            if self.data_handler.fvm_bt is not None
            else None
        )
        return self._predict_velocity_at_fvm_inputs(x, y, sdf, bt, case_id)

    def _predict_velocity_on_fvm_boundary_quadrature(
        self,
        case_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Predict velocity at physical boundary half-edge midpoints."""
        if (
            self.loss_computer.fvm is None
            or not hasattr(self.data_handler, "fvm_quad_x")
        ):
            return None
        offset = int(self.data_handler.fvm_int_quad_count)
        x = self.data_handler.fvm_quad_x[offset:].to(
            self.device,
            dtype=self.dtype,
        )
        y = self.data_handler.fvm_quad_y[offset:].to(
            self.device,
            dtype=self.dtype,
        )
        sdf = self.data_handler.fvm_quad_sdf[offset:].to(
            self.device,
            dtype=self.dtype,
        )
        bt = (
            self.data_handler.fvm_quad_bt[offset:].to(
                self.device,
                dtype=self.dtype,
            )
            if self.data_handler.fvm_quad_bt is not None
            else None
        )
        return self._predict_velocity_at_fvm_inputs(x, y, sdf, bt, case_id)

    def _predict_velocity_at_fvm_inputs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
        bt: Optional[torch.Tensor],
        case_id: int,
    ) -> torch.Tensor:
        """Run the common transformed PINN velocity inference path."""

        case_ids = torch.full(
            (x.shape[0],),
            int(case_id),
            device=self.device,
            dtype=torch.long,
        )
        inp = self.loss_computer.build_input_tensor(
            x,
            y,
            sdf,
            case_ids=case_ids,
        )
        out = self.model(inp)
        u, v, _ = self.loss_computer.transform_outputs(
            out[:, 0:1],
            out[:, 1:2],
            out[:, 2:3],
            x,
            y,
            sdf=sdf,
            case_ids=case_ids,
        )

        if self.loss_computer.bc_type == 'hard' and bt is not None:
            u, v = self._apply_hard_velocity_constraints(
                u,
                v,
                bt,
                case_ids=case_ids,
            )
        return torch.cat([u, v], dim=1)

    def _apply_hard_velocity_constraints(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        bt: torch.Tensor,
        case_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u = u_raw
        v = v_raw
        target_u, target_v = self.loss_computer.get_freestream_targets(
            u_raw,
            case_ids,
        )
        names = self.data_handler.boundary_type_names
        for idx, name in enumerate(names):
            if idx >= bt.shape[1]:
                continue
            mask = bt[:, idx].view(-1, 1) > 0.5
            if not bool(mask.any()):
                continue
            if name in ('inlet', 'velocity_dirichlet'):
                u = torch.where(mask, target_u, u)
                v = torch.where(mask, target_v, v)
            elif (
                name == 'wall'
                and self.loss_computer.pde_system
                == 'incompressible_navier_stokes'
            ):
                u = torch.where(mask, torch.zeros_like(u), u)
                v = torch.where(mask, torch.zeros_like(v), v)
            elif name == 'axis_symmetry':
                u = torch.where(mask, torch.zeros_like(u), u)
        return u, v

    def _compute_mass_flow_from_fvm_faces(
        self,
    ) -> Optional[Dict[str, Any]]:
        fvm = self.loss_computer.fvm
        if fvm is None:
            return None

        was_training = self.model.training
        self.model.eval()
        try:
            from airfoil_pinn.physics.fvm import BC_CODES
            inlet_code = BC_CODES['inlet']
            outlet_code = BC_CODES['outlet']
            farfield_code = BC_CODES['farfield']
            velocity_code = BC_CODES['velocity_dirichlet']

            case_metrics = {}
            with torch.no_grad():
                if fvm.bnd_node.numel() == 0:
                    return None
                bnd_area = fvm.bnd_S.to(self.device, dtype=self.dtype)
                bnd_type = fvm.bnd_type.to(self.device)
                inlet_mask = (
                    (bnd_type == inlet_code)
                    | (bnd_type == velocity_code)
                )
                outlet_mask = (
                    (bnd_type == outlet_code)
                    | (bnd_type == farfield_code)
                )
                flow_cases = getattr(
                    self.data_handler,
                    "flow_cases",
                    {0: None},
                )
                for case_id in sorted(flow_cases):
                    boundary_velocity = (
                        self._predict_velocity_on_fvm_boundary_quadrature(
                            case_id
                        )
                    )
                    if boundary_velocity is None:
                        return None
                    case_tensor = torch.full(
                        (1,),
                        int(case_id),
                        device=self.device,
                        dtype=torch.long,
                    )
                    target_u, target_v = (
                        self.loss_computer.get_freestream_targets(
                            boundary_velocity[:1, :1],
                            case_tensor,
                        )
                    )
                    freestream = torch.cat((target_u, target_v), dim=1).reshape(2)
                    boundary_velocity = fvm.apply_boundary_velocity(
                        boundary_velocity,
                        freestream_velocity=freestream,
                    )
                    flux = (boundary_velocity * bnd_area).sum(dim=1)

                    if self.loss_computer.flow_type == 'axisymmetric':
                        offset = int(self.data_handler.fvm_int_quad_count)
                        x = self.data_handler.fvm_quad_x[offset:].to(
                            self.device,
                            dtype=self.dtype,
                        ).flatten()
                        two_pi = torch.as_tensor(
                            2.0 * np.pi,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        flux = (
                            flux
                            * two_pi
                            * torch.clamp(x, min=0.0)
                        )

                    q_in = (
                        -flux[inlet_mask].sum().item()
                        if bool(inlet_mask.any())
                        else 0.0
                    )
                    q_out = (
                        flux[outlet_mask].sum().item()
                        if bool(outlet_mask.any())
                        else 0.0
                    )
                    denominator = abs(q_in) + 1e-12
                    values = {
                        "mass_flux_in": q_in,
                        "mass_flux_out": q_out,
                        "mass_balance_error": (
                            abs(q_in - q_out) / denominator
                            if denominator > 0.0
                            else 0.0
                        ),
                    }
                    node_velocity = self._predict_velocity_on_fvm_nodes(case_id)
                    if node_velocity is not None:
                        values.update(
                            self._compute_section_flow_metrics(
                                node_velocity,
                                reference_flux=q_in,
                            )
                        )
                    case_metrics[int(case_id)] = values
        finally:
            if was_training:
                self.model.train()

        result = {
            key: float(np.mean([values[key] for values in case_metrics.values()]))
            for key in (
                "mass_flux_in",
                "mass_flux_out",
                "mass_balance_error",
            )
        }
        section_maps = [
            values["section_volume_flux"]
            for values in case_metrics.values()
            if values.get("section_volume_flux")
        ]
        unavailable_sections = sorted(
            {
                section
                for values in case_metrics.values()
                for section in values.get(
                    "section_volume_flux_unavailable",
                    (),
                )
            },
            key=float,
        )
        if unavailable_sections:
            result["section_volume_flux_unavailable"] = (
                unavailable_sections
            )
        if section_maps:
            result["section_volume_flux_normalization"] = (
                "Q/(U_inf*c)=integral((u/U_inf), d(y/c))"
            )
            common_sections = set(section_maps[0])
            for section_map in section_maps[1:]:
                common_sections.intersection_update(section_map)
            result["section_volume_flux"] = {
                section: float(
                    np.mean(
                        [
                            section_map[section]
                            for section_map in section_maps
                        ]
                    )
                )
                for section in sorted(
                    common_sections,
                    key=float,
                )
            }
            relative_maps = [
                values["section_volume_flux_relative_error"]
                for values in case_metrics.values()
                if values.get("section_volume_flux_relative_error")
            ]
            result["section_volume_flux_relative_error"] = {
                section: float(
                    np.mean(
                        [
                            section_map[section]
                            for section_map in relative_maps
                            if section in section_map
                        ]
                    )
                )
                for section in sorted(
                    common_sections,
                    key=float,
                )
            }
            deviation_maps = [
                values["section_volume_flux_relative_deviation"]
                for values in case_metrics.values()
                if values.get(
                    "section_volume_flux_relative_deviation"
                )
            ]
            result["section_volume_flux_relative_deviation"] = {
                section: float(
                    np.mean(
                        [
                            section_map[section]
                            for section_map in deviation_maps
                            if section in section_map
                        ]
                    )
                )
                for section in sorted(
                    common_sections,
                    key=float,
                )
            }
            result["section_volume_flux_relative_spread"] = float(
                np.mean(
                    [
                        values[
                            "section_volume_flux_relative_spread"
                        ]
                        for values in case_metrics.values()
                        if (
                            "section_volume_flux_relative_spread"
                            in values
                        )
                    ]
                )
            )
            result["section_volume_flux_max_relative_error"] = float(
                np.mean(
                    [
                        values[
                            "section_volume_flux_max_relative_error"
                        ]
                        for values in case_metrics.values()
                        if (
                            "section_volume_flux_max_relative_error"
                            in values
                        )
                    ]
                )
            )
        for case_id, values in case_metrics.items():
            for key, value in values.items():
                result[f"{key}_case_{case_id}"] = value
        return result

    def _compute_section_flow_metrics(
        self,
        velocity: torch.Tensor,
        *,
        reference_flux: float,
    ) -> Dict[str, Any]:
        """Integrate ``u`` on exact vertical cuts through the FVM mesh."""
        if (
            not self.section_flow_enabled
            or not self.section_flow_sections
            or self.data_handler.mesh_data is None
            or getattr(
                getattr(self, "loss_computer", None),
                "flow_type",
                "2d",
            )
            == "axisymmetric"
        ):
            return {}

        u_nodes = (
            velocity[:, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        section_fluxes: Dict[str, float] = {}
        unavailable_sections: List[str] = []
        for x_section in self.section_flow_sections:
            flux = self.data_handler.integrate_fvm_vertical_flux(
                x_section,
                u_nodes,
            )
            if np.isfinite(flux):
                section_fluxes[
                    self._format_section_coordinate(x_section)
                ] = float(flux)
            else:
                unavailable_sections.append(
                    self._format_section_coordinate(x_section)
                )

        if not section_fluxes:
            return {
                "section_volume_flux": {},
                "section_volume_flux_unavailable": (
                    unavailable_sections
                ),
            }
        denominator = abs(float(reference_flux)) + 1e-12
        relative_errors = {
            section: abs(flux - float(reference_flux)) / denominator
            for section, flux in section_fluxes.items()
        }
        relative_deviations = {
            section: (flux - float(reference_flux)) / denominator
            for section, flux in section_fluxes.items()
        }
        flux_values = np.asarray(
            list(section_fluxes.values()),
            dtype=np.float64,
        )
        relative_spread = (
            float(np.ptp(flux_values) / denominator)
            if flux_values.size > 1
            else 0.0
        )
        return {
            "section_volume_flux": section_fluxes,
            "section_volume_flux_unavailable": unavailable_sections,
            "section_volume_flux_relative_error": relative_errors,
            "section_volume_flux_relative_deviation": (
                relative_deviations
            ),
            "section_volume_flux_relative_spread": relative_spread,
            "section_volume_flux_max_relative_error": float(
                max(relative_errors.values(), default=0.0)
            ),
        }

    @staticmethod
    def _format_section_coordinate(value: float) -> str:
        """Return a stable compact JSON key for a section coordinate."""
        return f"{float(value):.12g}"

    @staticmethod
    def format_section_flow_log(
        metrics: Dict[str, Any],
    ) -> Optional[str]:
        """Format exact section-flow diagnostics for the text run log."""
        section_fluxes = metrics.get("section_volume_flux", {})
        unavailable = metrics.get(
            "section_volume_flux_unavailable",
            (),
        )
        if not section_fluxes and not unavailable:
            return None
        parts = [
            f"Q*(x/c={section})={value:.6g}"
            for section, value in section_fluxes.items()
        ]
        line = (
            "   section Q*=integral(u* d(y/c)): "
            + (", ".join(parts) if parts else "no valid sections")
        )
        if section_fluxes:
            line += (
                "; relative spread="
                f"{metrics.get('section_volume_flux_relative_spread', 0.0):.3e}"
                "; max relative error vs inlet="
                f"{metrics.get('section_volume_flux_max_relative_error', 0.0):.3e}"
            )
        if unavailable:
            line += "; unavailable x/c=" + ",".join(
                str(section) for section in unavailable
            )
        return line

    @staticmethod
    def _trapz_integral(values: np.ndarray, tangent: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        order = np.argsort(tangent)
        values = values[order]
        tangent = tangent[order]
        if values.size == 1:
            return float(values[0])
        trapezoid = getattr(np, 'trapezoid', None)
        if trapezoid is None:
            trapezoid = np.trapz
        return float(trapezoid(values, tangent))

    def _compute_mass_flow_from_boundary_points(self) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        case_data: Dict[int, Dict[str, List[np.ndarray]]] = {}

        with torch.no_grad():
            dataloader = self.data_handler.get_boundary_eval_dataloader(('inlet', 'outlet'))
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                case_ids = getattr(batch, 'case_id', None)
                if case_ids is None:
                    case_ids = torch.zeros(
                        x.shape[0],
                        dtype=torch.long,
                    )
                case_ids = case_ids.to(
                    self.device,
                    dtype=torch.long,
                ).reshape(-1)
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = self.loss_computer.evaluate_sdf_tensor(x, y)
                bt = bt.to(self.device, dtype=self.dtype) if bt.shape[1] > 0 else None
                if bt is None:
                    continue

                inp = self.loss_computer.build_input_tensor(
                    x,
                    y,
                    sdf,
                    case_ids=case_ids,
                )
                out = self.model(inp)
                u, v, _ = self.loss_computer.transform_outputs(
                    out[:, 0:1],
                    out[:, 1:2],
                    out[:, 2:3],
                    x,
                    y,
                    sdf=sdf,
                    case_ids=case_ids,
                )
                if self.loss_computer.bc_type == 'hard':
                    u, v = self._apply_hard_velocity_constraints(
                        u,
                        v,
                        bt,
                        case_ids=case_ids,
                    )

                names = self.data_handler.boundary_type_names
                inlet_idx = (
                    names.index("inlet") if "inlet" in names else None
                )
                outlet_idx = (
                    names.index("outlet") if "outlet" in names else None
                )
                for case_id in torch.unique(case_ids, sorted=True):
                    value = int(case_id.item())
                    storage = case_data.setdefault(
                        value,
                        {
                            "inlet_u": [],
                            "inlet_y": [],
                            "outlet_u": [],
                            "outlet_y": [],
                        },
                    )
                    case_mask = case_ids == case_id
                    if inlet_idx is not None:
                        inlet_mask = (
                            (bt[:, inlet_idx] > 0.5) & case_mask
                        )
                        if bool(inlet_mask.any()):
                            storage["inlet_u"].append(
                                u[inlet_mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                            )
                            storage["inlet_y"].append(
                                y[inlet_mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                            )
                    if outlet_idx is not None:
                        outlet_mask = (
                            (bt[:, outlet_idx] > 0.5) & case_mask
                        )
                        if bool(outlet_mask.any()):
                            storage["outlet_u"].append(
                                u[outlet_mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                            )
                            storage["outlet_y"].append(
                                y[outlet_mask]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                            )

        if was_training:
            self.model.train()

        case_metrics = {}
        empty = np.array([], dtype=np.float64)
        for case_id, storage in case_data.items():
            q_in = self._trapz_integral(
                (
                    np.concatenate(storage["inlet_u"])
                    if storage["inlet_u"]
                    else empty
                ),
                (
                    np.concatenate(storage["inlet_y"])
                    if storage["inlet_y"]
                    else empty
                ),
            )
            q_out = self._trapz_integral(
                (
                    np.concatenate(storage["outlet_u"])
                    if storage["outlet_u"]
                    else empty
                ),
                (
                    np.concatenate(storage["outlet_y"])
                    if storage["outlet_y"]
                    else empty
                ),
            )
            denominator = abs(q_in) + 1e-12
            case_metrics[case_id] = {
                "mass_flux_in": q_in,
                "mass_flux_out": q_out,
                "mass_balance_error": (
                    abs(q_in - q_out) / denominator
                    if denominator > 0.0
                    else 0.0
                ),
            }
        if not case_metrics:
            return {
                "mass_flux_in": 0.0,
                "mass_flux_out": 0.0,
                "mass_balance_error": 0.0,
            }
        result = {
            key: float(np.mean([values[key] for values in case_metrics.values()]))
            for key in (
                "mass_flux_in",
                "mass_flux_out",
                "mass_balance_error",
            )
        }
        for case_id, values in case_metrics.items():
            for key, value in values.items():
                result[f"{key}_case_{case_id}"] = value
        return result

    def _update_boundary_avg_velocities(self) -> None:
        """Вычисляет средние скорости на inlet/outlet с помощью модели (нормированные)."""
        was_training = self.model.training
        self.model.eval()
        dataloader = self.data_handler.get_boundary_eval_dataloader(('inlet', 'outlet'))
        u_inlet_sum = 0.0
        u_outlet_sum = 0.0
        count_inlet = 0
        count_outlet = 0

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                case_ids = getattr(batch, 'case_id', None)
                # Приводим к нужному устройству и типу данных
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = self.loss_computer.evaluate_sdf_tensor(x, y)
                if bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None
                if bt is None:
                    continue
                mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
                u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)

                # Построение входного тензора
                input_tensor = self.loss_computer.build_input_tensor(
                    x,
                    y,
                    sdf,
                    case_ids=case_ids,
                )
                out = self.model(input_tensor)
                u_raw = out[:, 0:1]
                u, _, _ = self.loss_computer.transform_outputs(
                    u_raw,
                    out[:, 1:2],
                    out[:, 2:3],
                    x,
                    y,
                    sdf=sdf,
                    case_ids=case_ids,
                )

                # Legacy pointwise override.
                if self.loss_computer.bc_type == 'hard':
                    u = torch.where(mask_u, u_bc, u)

                if 'inlet' in self.data_handler.boundary_type_names:
                    inlet_idx = self.data_handler.boundary_type_names.index('inlet')
                    mask = bt[:, inlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_inlet_sum += u[mask].sum().item()
                        count_inlet += mask.sum().item()
                if 'outlet' in self.data_handler.boundary_type_names:
                    outlet_idx = self.data_handler.boundary_type_names.index('outlet')
                    mask = bt[:, outlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_outlet_sum += u[mask].sum().item()
                        count_outlet += mask.sum().item()

        self._cached_u_in_norm = u_inlet_sum / count_inlet if count_inlet > 0 else 0.0
        self._cached_u_out_norm = u_outlet_sum / count_outlet if count_outlet > 0 else 0.0
        if was_training:
            self.model.train()

    # ------------------------------------------------------------------
    # Вывод таблицы потерь в консоль
    # ------------------------------------------------------------------
    def get_recent_loss_change(
        self,
        max_epochs: int = 25,
    ) -> Tuple[Optional[float], int, Optional[float], Optional[float]]:
        """Return total-loss change over the latest available epoch window.

        The window contains at most ``max_epochs`` recorded epoch values. The
        percentage is relative to the first loss in that window, so a negative
        value means that the loss decreased. With fewer than two valid values
        the percentage is undefined and ``None`` is returned.
        """

        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

        values = self.history.get("loss", [])
        if not isinstance(values, (list, tuple)) or not values:
            return None, 0, None, None

        window_epochs = min(max_epochs, len(values))
        start_loss = float(values[-window_epochs])
        end_loss = float(values[-1])
        if (
            window_epochs < 2
            or not np.isfinite(start_loss)
            or not np.isfinite(end_loss)
            or abs(start_loss) <= 1e-12
        ):
            return None, window_epochs, start_loss, end_loss

        change_percent = 100.0 * (end_loss - start_loss) / abs(start_loss)
        return change_percent, window_epochs, start_loss, end_loss

    def print_epoch_table(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        pde_raw: Dict[str, float],
        bc_comp: Dict[str, float],
        integral_loss: float,
        pde_max: Dict[str, float],
        epoch_time: float,
        total_time: float,
        fvm_loss: float = 0.0,
        fvm_raw_loss: Optional[float] = None,
        fvm_rms: Optional[float] = None,
        optimizer_name: str = "Adam",
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
        grad_clip: Optional[float] = None,
        pde_multiplier: float = 1.0,
        eta_seconds: Optional[float] = None,
        avg_u_in: float = 0.0,
        avg_u_out: float = 0.0,
        mass_balance_error: float = 0.0,
    ) -> None:
        """Print the complete per-epoch training summary."""

        unicode_chars = "┌┬┐├┼┤└┴┘─│·Δ↓↑→—∞"
        output_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            unicode_chars.encode(output_encoding)
            use_unicode = True
        except (LookupError, UnicodeEncodeError):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                use_unicode = True
            except (AttributeError, OSError, ValueError):
                use_unicode = False

        vertical = "│" if use_unicode else "|"
        horizontal = "─" if use_unicode else "-"
        bullet = "·" if use_unicode else "|"
        missing = "—" if use_unicode else "--"
        down_arrow = "↓" if use_unicode else "v"
        up_arrow = "↑" if use_unicode else "^"
        flat_arrow = "→" if use_unicode else "="

        def format_duration(seconds: Optional[float]) -> str:
            if seconds is None or not np.isfinite(seconds):
                return "--:--"
            total_seconds = max(int(round(float(seconds))), 0)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            if hours:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:02d}:{secs:02d}"

        def format_number(value: Optional[float]) -> str:
            if value is None:
                return missing
            return f"{float(value):.2e}"

        def format_weight(value: Optional[float]) -> str:
            if value is None:
                return "—"
            number = float(value)
            text = f"{number:.2f}"
            return text if len(text) <= 6 else f"{number:.0e}"

        def history_values(
            key: str,
            component: Optional[str] = None,
        ) -> List[float]:
            values = self.history.get(key, [])
            if component is not None:
                if not isinstance(values, dict):
                    return []
                values = values.get(component, [])
            return values if isinstance(values, list) else []

        def format_delta(
            current: float,
            values: List[float],
        ) -> str:
            if len(values) < 2:
                return missing
            previous = float(values[-2])
            current_value = float(current)
            scale = abs(previous)
            if scale <= 1e-12:
                if abs(current_value) <= 1e-12:
                    return f"{flat_arrow} 0.0"
                return f"{up_arrow}  inf"
            delta = 100.0 * (current_value - previous) / scale
            if abs(delta) < 0.05:
                return f"{flat_arrow} 0.0"
            arrow = down_arrow if delta < 0.0 else up_arrow
            magnitude = abs(delta)
            if magnitude >= 9999.5:
                return f"{arrow} >9999"
            return f"{arrow} {magnitude:.1f}"

        def print_row(
            name: str,
            mse: float,
            rms: Optional[float],
            max_abs: Optional[float],
            weight: Optional[float],
            weighted_mse: Optional[float],
            delta: str,
        ) -> None:
            print(
                f"{vertical} {name:<16} "
                f"{vertical} {format_number(mse):>8} "
                f"{vertical} {format_number(rms):>8} "
                f"{vertical} {format_number(max_abs):>8} "
                f"{vertical} {format_weight(weight):>6} "
                f"{vertical} {format_number(weighted_mse):>8} "
                f"{vertical} {delta:>7} {vertical}"
            )

        cell_widths = (18, 10, 10, 10, 8, 10, 9)
        if use_unicode:
            top_chars = ("┌", "┬", "┐")
            divider_chars = ("├", "┼", "┤")
            bottom_chars = ("└", "┴", "┘")
        else:
            top_chars = divider_chars = bottom_chars = ("+", "+", "+")

        def border(chars: Tuple[str, str, str]) -> str:
            left, junction, right = chars
            return (
                left
                + junction.join(
                    horizontal * width for width in cell_widths
                )
                + right
            )

        def print_group(name: str) -> None:
            inner_width = sum(cell_widths) + len(cell_widths) - 1
            print(
                f"{vertical} {name:<{inner_width - 2}} {vertical}"
            )

        top = border(top_chars)
        divider = border(divider_chars)
        bottom = border(bottom_chars)

        completed_percent = (
            100.0 * epoch / total_epochs if total_epochs > 0 else 100.0
        )
        finish_text = "--:--"
        if eta_seconds is not None and np.isfinite(eta_seconds):
            finish_at = (
                datetime.datetime.now()
                + datetime.timedelta(seconds=max(float(eta_seconds), 0.0))
            )
            finish_text = finish_at.strftime("%H:%M")

        print()
        print(
            f"Epoch {epoch}/{total_epochs} [{completed_percent:.1f}%] "
            f"{bullet} {optimizer_name} {bullet} "
            f"{epoch_time:.2f} s/epoch {bullet} "
            f"elapsed {format_duration(total_time)} {bullet} "
            f"ETA {format_duration(eta_seconds)} {bullet} "
            f"finish ~{finish_text}"
        )
        print(
            f"LR {learning_rate:.2e} {bullet} "
            f"Grad {grad_norm:.2e} {bullet} "
            f"PDE multiplier {pde_multiplier:.2f}"
        )
        (
            loss_change_percent,
            loss_change_epochs,
            loss_window_start,
            loss_window_end,
        ) = self.get_recent_loss_change()
        if loss_change_percent is None:
            print(
                f"Loss change (last {loss_change_epochs} epoch): n/a"
            )
        else:
            print(
                f"Loss change (last {loss_change_epochs} epochs): "
                f"{loss_change_percent:+.2f}% "
                f"({loss_window_start:.3e} -> {loss_window_end:.3e})"
            )
        print()
        print(top)
        print(
            f"{vertical} Component        {vertical}      MSE "
            f"{vertical}      RMS {vertical}  Max Abs "
            f"{vertical} Weight {vertical} Weighted "
            f"{vertical}   {'Δ' if use_unicode else 'd'}, %  {vertical}"
        )
        print(divider)

        w = self.loss_computer.loss_weights
        print_group("PDE")

        pde_names = {}
        if self.loss_computer.pde_components:
            pde_names[self.loss_computer.pde_components[0]] = "Continuity"
        if len(self.loss_computer.pde_components) > 1:
            pde_names[self.loss_computer.pde_components[1]] = (
                "Momentum R"
                if self.loss_computer.flow_type == "axisymmetric"
                else "Momentum X"
            )
        if len(self.loss_computer.pde_components) > 2:
            pde_names[self.loss_computer.pde_components[2]] = (
                "Momentum Z"
                if self.loss_computer.flow_type == "axisymmetric"
                else "Momentum Y"
            )

        for component in self.loss_computer.pde_components:
            mse = float(pde_raw.get(component, 0.0))
            weight = float(w.get(component, 0.0))
            print_row(
                pde_names.get(component, component.replace("_", " ").title()),
                mse,
                float(np.sqrt(max(mse, 0.0))),
                float(np.sqrt(max(pde_max.get(component, 0.0), 0.0))),
                weight,
                mse * weight,
                format_delta(mse, history_values("pde", component)),
            )

        print(divider)
        print_group("BOUNDARY")
        for name in self.data_handler.boundary_type_names:
            mse = (
                float(bc_comp.get(name, 0.0))
                if self.loss_computer.bc_type == "soft"
                else 0.0
            )
            weight = (
                float(w.get(name, 0.0))
                if self.loss_computer.bc_type == "soft"
                else 0.0
            )
            print_row(
                name.replace("_", " ").title(),
                mse,
                float(np.sqrt(max(mse, 0.0))),
                None,
                weight,
                mse * weight,
                format_delta(mse, history_values("bc", name)),
            )

        auxiliary_rows = []
        if self.loss_computer.integral_condition:
            integral_weight = float(self.loss_computer.integral_weight)
            auxiliary_rows.append(
                (
                    "Integral Flow",
                    integral_loss,
                    float(np.sqrt(max(integral_loss, 0.0))),
                    integral_weight,
                    integral_loss * integral_weight,
                    history_values("integral_loss"),
                )
            )

        fvm_weight = float(
            getattr(self.loss_computer, "fvm_weight", 0.0)
        )
        if fvm_weight > 0.0:
            fvm_raw = (
                fvm_loss / fvm_weight
                if fvm_raw_loss is None
                else float(fvm_raw_loss)
            )
            fvm_rms_value = (
                float(fvm_rms)
                if fvm_rms is not None
                else float(np.sqrt(max(fvm_raw, 0.0)))
            )
            fvm_history = history_values("fvm_mse")
            if not fvm_history:
                fvm_history = history_values("fvm_raw_loss")
            auxiliary_rows.append(
                (
                    "FVM Divergence",
                    fvm_raw,
                    fvm_rms_value,
                    fvm_weight,
                    fvm_loss,
                    fvm_history,
                )
            )

        if auxiliary_rows:
            print(divider)
            print_group("AUXILIARY")
            for (
                name,
                mse,
                rms,
                weight,
                weighted_mse,
                values,
            ) in auxiliary_rows:
                print_row(
                    name,
                    float(mse),
                    float(rms),
                    None,
                    float(weight),
                    float(weighted_mse),
                    format_delta(float(mse), values),
                )

        total_pde = float(sum(pde_raw.values()))
        total_bc = (
            float(sum(bc_comp.values()))
            if self.loss_computer.bc_type == "soft"
            else 0.0
        )
        weighted_total_pde = sum(
            pde_raw.get(name, 0.0) * w.get(name, 0.0)
            for name in self.loss_computer.pde_components
        )
        weighted_total_bc = sum(
            bc_comp.get(name, 0.0) * w.get(name, 0.0)
            for name in bc_comp
        )
        max_abs_pde = max(
            (
                np.sqrt(max(pde_max.get(name, 0.0), 0.0))
                for name in self.loss_computer.pde_components
            ),
            default=0.0,
        )

        print(divider)
        print_row(
            "Total PDE",
            total_pde,
            float(np.sqrt(max(total_pde, 0.0))),
            float(max_abs_pde),
            None,
            float(weighted_total_pde),
            format_delta(total_pde, history_values("total_pde")),
        )
        print_row(
            "Total BC",
            total_bc,
            float(np.sqrt(max(total_bc, 0.0))),
            None,
            None,
            float(weighted_total_bc),
            format_delta(total_bc, history_values("total_bc")),
        )
        print_row(
            "TOTAL LOSS",
            float(loss),
            None,
            None,
            None,
            float(loss),
            format_delta(float(loss), history_values("loss")),
        )
        print(bottom)
        clip_enabled = (
            grad_clip is not None
            and np.isfinite(float(grad_clip))
            and float(grad_clip) > 0.0
        )
        if clip_enabled:
            clip_limit = float(grad_clip)
            print(
                f"Mean step gradient norm (pre-clip, Adam): {grad_norm:.3e} {bullet} "
                f"clip limit: {clip_limit:.3e} {bullet} "
                "clipping: see per-group gradient report"
            )
        else:
            print(
                f"Mean step gradient norm (pre-clip, Adam): {grad_norm:.3e} {bullet} "
                "clipping: disabled"
            )
        print(
            f"Flow: u_in={avg_u_in:.4f} {bullet} "
            f"u_out={avg_u_out:.4f} {bullet} "
            f"mass error={mass_balance_error:.3e}"
        )

    def _compute_rate(self, category: str, key: Optional[str], current: float) -> float:
        """
        Вычисляет отношение текущего значения к предыдущему для заданной категории.

        Args:
            category: 'pde', 'bc', 'integral', 'total_bc', 'total_pde', 'loss'
            key: ключ внутри категории (для PDE и BC)
            current: текущее значение

        Returns:
            отношение current / previous (с защитой от деления на ноль)
        """
        eps = 1e-12
        if len(self.history['loss']) < 1:
            return 1.0

        if category == 'pde' and key is not None:
            hist = self.history['pde'].get(key, [])
            if len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'bc' and key is not None:
            hist = self.history['bc'].get(key, [])
            if len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'integral':
            hist = self.history.get('integral_loss', [])
            if hist and len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'total_bc':
            if self.history.get('total_bc') and len(self.history['total_bc']) >= 2:
                prev = self.history['total_bc'][-2]
            else:
                return 1.0
        elif category == 'total_pde':
            if self.history.get('total_pde') and len(self.history['total_pde']) >= 2:
                prev = self.history['total_pde'][-2]
            else:
                return 1.0
        elif category == 'fvm':
            hist = self.history.get('fvm_loss', [])
            if hist and len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'loss':
            if len(self.history['loss']) >= 2:
                prev = self.history['loss'][-2]
            else:
                return 1.0
        else:
            return 1.0

        return current / (prev + eps)

    # ------------------------------------------------------------------
    # Логирование в файл
    # ------------------------------------------------------------------
    def save_log(self, text: str) -> None:
        """
        Записывает строку в лог-файл (создаётся при первом вызове).

        Args:
            text: текст для записи
        """
        if self.log_file is None:
            log_name = f"{self.config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_path = os.path.join(self.logs_dir, log_name)
            self.log_file = open(log_path, 'w', encoding='utf-8')
        self.log_file.write(text + "\n")
        self.log_file.flush()

    # ------------------------------------------------------------------
    # Финальные метрики
    # ------------------------------------------------------------------
    def compute_final_metrics(self) -> Dict[str, Any]:
        """
        Вычисляет итоговые метрики после завершения обучения:
        - средние скорости на входе/выходе (нормированные и физические)
        - ошибка баланса масс
        - финальные потери (PDE, BC, integral)
        - лучшая эпоха
        - финальная норма градиентов
        - относительная L2-ошибка, если есть эталонное решение

        Returns:
            словарь с метриками
        """
        # Последние значения из истории (с защитой от пустых списков)
        pde_hist = self.history.get('pde', {})
        if pde_hist and all(len(pde_hist.get(c, [])) > 0 for c in self.loss_computer.pde_components):
            final_pde_loss = sum(pde_hist[c][-1] for c in self.loss_computer.pde_components)
        else:
            final_pde_loss = 0.0
        final_pde_mse = {
            component: (
                float(pde_hist[component][-1])
                if pde_hist.get(component)
                else 0.0
            )
            for component in self.loss_computer.pde_components
        }
        final_pde_rms = {
            component: float(np.sqrt(max(value, 0.0)))
            for component, value in final_pde_mse.items()
        }

        bc_hist = self.history.get('bc', {})
        final_bc_loss = sum(v[-1] for v in bc_hist.values() if v and len(v) > 0) if bc_hist else 0.0
        final_bc_mse = {
            boundary: float(values[-1]) if values else 0.0
            for boundary, values in bc_hist.items()
        }
        final_bc_rms = {
            boundary: float(np.sqrt(max(value, 0.0)))
            for boundary, value in final_bc_mse.items()
        }

        int_hist = self.history.get('integral_loss')
        final_integral_loss = int_hist[-1] if int_hist and len(int_hist) > 0 else 0.0

        # Средние скорости на границах (нормированные)
        avg_u_in_norm, avg_u_out_norm = self._cached_u_in_norm, self._cached_u_out_norm
        avg_u_in_phys = avg_u_in_norm * self.data_handler.U_inf
        avg_u_out_phys = avg_u_out_norm * self.data_handler.U_inf

        mass_metrics = self._compute_mass_flow_metrics()
        Qin = mass_metrics['mass_flux_in']
        Qout = mass_metrics['mass_flux_out']
        mass_balance_error = mass_metrics['mass_balance_error']

        # Лучшая эпоха
        self._ensure_history_schema()
        convergence = self.history['convergence_loss']
        candidates = [(value, index + 1) for index, value in enumerate(convergence) if np.isfinite(value)]
        best_loss, best_epoch = min(candidates) if candidates else (float('inf'), 0)

        # Финальная норма градиентов
        final_grad_norm = self.history['grad_norm'][-1] if self.history['grad_norm'] else 0.0

        fvm_hist = self.history.get('fvm_loss', [])
        final_fvm_loss = fvm_hist[-1] if fvm_hist else 0.0
        fvm_raw_hist = self.history.get('fvm_raw_loss', [])
        final_fvm_raw_loss = fvm_raw_hist[-1] if fvm_raw_hist else 0.0
        fvm_mse_hist = self.history.get('fvm_mse', [])
        final_fvm_mse = (
            fvm_mse_hist[-1] if fvm_mse_hist else final_fvm_raw_loss
        )
        fvm_rms_hist = self.history.get('fvm_rms', [])
        final_fvm_rms = fvm_rms_hist[-1] if fvm_rms_hist else 0.0

        metrics = {
            'experiment': self.config_name,
            'final_u_out_phys': avg_u_out_phys,
            'final_u_in_phys': avg_u_in_phys,
            'final_u_out_norm': avg_u_out_norm,
            'final_u_in_norm': avg_u_in_norm,
            'mass_flux_in': Qin,
            'mass_flux_out': Qout,
            'mass_balance_error': mass_balance_error,
            'final_pde_loss': final_pde_loss,
            'final_pde_mse': final_pde_mse,
            'final_pde_rms': final_pde_rms,
            'final_bc_loss': final_bc_loss,
            'final_boundary_mse': final_bc_mse,
            'final_boundary_rms': final_bc_rms,
            'final_integral_loss': final_integral_loss,
            'final_fvm_loss': final_fvm_loss,
            'final_fvm_raw_loss': final_fvm_raw_loss,
            'final_fvm_mse': final_fvm_mse,
            'final_fvm_weighted_loss': final_fvm_loss,
            'final_fvm_rms': final_fvm_rms,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_loss_metric': 'unweighted_pde_bc_mse',
            'final_gradient_norm': final_grad_norm,
            'relative_l2_error': 'N/A',
        }

        # Если есть эталонное решение, вычисляем L2-ошибку
        if self.reference_data is not None:
            try:
                from airfoil_pinn.training.reference import (
                    compute_l2_errors,
                    interpolate_reference,
                )

                # Получаем предсказания на всех точках датасета
                dataset = self.data_handler.dataset
                x_norm = dataset.x.to(self.device, dtype=self.dtype)
                y_norm = dataset.y.to(self.device, dtype=self.dtype)
                sdf_norm = self.loss_computer.evaluate_sdf_tensor(
                    x_norm,
                    y_norm,
                )
                bt = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None

                self.model.eval()
                with torch.no_grad():
                    input_tensor = self.loss_computer.build_input_tensor(
                        x_norm,
                        y_norm,
                        sdf_norm,
                        case_ids=dataset.case_id,
                    )
                    out = self.model(input_tensor)
                    u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                    u_pred_t, v_pred_t, p_pred_t = (
                        self.loss_computer.transform_outputs(
                            u_raw,
                            v_raw,
                            p_raw,
                            x_norm,
                            y_norm,
                            sdf=sdf_norm,
                            case_ids=dataset.case_id,
                        )
                    )
                    if self.loss_computer.bc_type == 'hard':
                        u_pred_t, v_pred_t, p_pred_t = self.loss_computer.apply_hard_constraints(
                            u_pred_t, v_pred_t, p_pred_t,
                            dataset.mask_u_fixed.to(self.device).view(-1, 1),
                            dataset.mask_v_fixed.to(self.device).view(-1, 1),
                            dataset.mask_p_fixed.to(self.device).view(-1, 1),
                            dataset.u_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                            dataset.v_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                            dataset.p_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                        )
                        u_pred_t, v_pred_t = self.loss_computer.apply_velocity_dirichlet_constraint(
                            u_pred_t,
                            v_pred_t,
                            bt,
                            case_ids=dataset.case_id,
                        )

                    p_dyn = 0.5 * self.data_handler.rho * self.data_handler.U_inf ** 2
                    u_pred = (u_pred_t * self.data_handler.U_inf).cpu().numpy().flatten()
                    v_pred = (v_pred_t * self.data_handler.U_inf).cpu().numpy().flatten()
                    p_pred = (p_pred_t * p_dyn + self.data_handler.p_ref).cpu().numpy().flatten()

                # Физические координаты
                x_phys = x_norm.cpu().numpy().flatten() * self.data_handler.chord
                y_phys = y_norm.cpu().numpy().flatten() * self.data_handler.chord

                # Интерполяция эталонных данных
                ref_interp = interpolate_reference(
                    self.reference_data,
                    x_phys,
                    y_phys,
                    method='linear'
                )

                # Вычисление ошибок (относительные)
                errors = compute_l2_errors(
                    u_pred, v_pred, p_pred,
                    ref_interp['u'], ref_interp['v'], ref_interp['p'],
                    norm='relative'
                )
                metrics['relative_l2_error'] = errors['mean']
                metrics['l2_u'] = errors['u']
                metrics['l2_v'] = errors['v']
                metrics['l2_p'] = errors['p']

                logger.info(f"Относительная L2-ошибка: {errors['mean']:.6e}")
            except Exception as e:
                logger.warning(f"Не удалось вычислить L2-ошибку: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Закрытие лог-файла
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Закрывает файл лога, если он открыт."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
