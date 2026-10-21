from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Вычисление функции потерь для Pipe-Bend PINN.

Поддерживает только 2D плоское течение (planar).
BC: inlet, outlet, wall (без symmetry).
Исправленная версия:
- compute_loss() возвращает словарь с полями loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max
- _compute_bc_loss() использует boundary_types one-hot тензор вместо эвристик по значению скорости
- Добавлен compute_integral_loss()

Автор: Адаптировано из Airfoil PINN
Версия: 1.1
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from pipe_bend_pinn.physics.pde import PDEResiduals
from pipe_bend_pinn.data.handler import DataHandler
from pipe_bend_pinn.utils.helpers import safe_grad

logger = logging.getLogger(__name__)


class LossComputer:
    """
    Вычисление функции потерь: PDE невязки + граничные условия + интегральные.

    Для pipe bend используются только типы: inlet, outlet, wall.
    compute_loss() возвращает словарь для совместимости с Trainer.train_epoch_adam().
    """

    def __init__(self, config: Dict, data_handler: DataHandler,
                 rho_norm: torch.Tensor, mu_norm: torch.Tensor,
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 device: torch.device, dtype: torch.dtype,
                 re_ref: Optional[float] = None):
        self.config = config
        self.data_handler = data_handler
        self.rho_norm = rho_norm
        self.mu_norm = mu_norm
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.device = device
        self.dtype = dtype

        # Типы границ для pipe bend (без symmetry)
        self.boundary_type_names = ['inlet', 'outlet', 'wall']
        self.bc_type = config.get('boundary_conditions', {}).get('type', 'soft')

        # Веса потерь
        lw = config['loss_weights']
        self.pde_weights = lw['pde_weights']
        self.bc_weights = lw['bc_weights']
        self.loss_weights = lw
        self.adaptive_bc = lw.get('adaptive_bc', False)

        # Интегральное условие (баланс масс)
        self.integral_condition = lw.get('integral_condition', False)
        self.integral_weight = lw.get('integral_weight', 100000.0)

        # Слабая supervised-привязка к начальному полю (функция тока):
        # якорь против дрейфа модели в нефизичный минимум на внутренних точках.
        ref_cfg = lw.get('reference_field', {})
        self.reference_weight = float(ref_cfg.get('weight', 0.0)) if isinstance(ref_cfg, dict) else 0.0

        # Pressure-Poisson: ∇²p + 2·∇·(u·∇u) = 0 (развивает градиент давления).
        pp_cfg = lw.get('pressure_poisson', {})
        self.pp_weight = float(pp_cfg.get('weight', 0.0)) if isinstance(pp_cfg, dict) else 0.0

        # SA-веса
        self.use_sa_weights = lw.get('use_sa_weights', False)
        self.sa_apply_to_bc = lw.get('sa_apply_to_bc', False)

        # --- Параметризация по Re и data-loss по CFD-полям --------------------
        arch_cfg = config.get('architecture', {})
        self.parametric = bool(arch_cfg.get('parametric', False))
        self.re_ref = float(re_ref) if re_ref is not None else 1.0
        cfd_cfg = config.get('data', {}).get('cfd', {}) or {}
        re_range = cfd_cfg.get('re_range', config.get('data', {}).get('re_range'))
        if re_range:
            re_min, re_max = float(re_range[0]), float(re_range[1])
            if re_min <= 0.0 or re_max < re_min:
                raise ValueError(f"data.cfd.re_range: некорректный диапазон {re_range}")
            self.re_min, self.re_max = re_min, re_max
        else:
            self.re_min = self.re_max = 1.0
        self.re_default = float(cfd_cfg.get('re_default', 1.0))

        dl_cfg = lw.get('data_loss', {}) or {}
        self.data_enabled = bool(dl_cfg.get('enabled', False))
        self.data_weight = float(dl_cfg.get('weight', 1.0))
        dl_weights = dl_cfg.get('weights', {}) or {}
        self.data_w_u = float(dl_weights.get('u', 1.0))
        self.data_w_v = float(dl_weights.get('v', 1.0))
        self.data_w_p = float(dl_weights.get('p', 1.0))
        if self.data_enabled:
            logger.info(
                "data-loss включён: weight=%.3g (u=%.3g, v=%.3g, p=%.3g)",
                self.data_weight, self.data_w_u, self.data_w_v, self.data_w_p,
            )

        # Пристеночное усиление PDE (вариант 1, скорректированный): у стенок
        # центробежный источник 2·uθ²/r → 0, и физический ∂p/∂n там задаётся
        # вязким балансом пограничного слоя. Усиливать надо ВЕСЬ NS (momentum
        # И continuity вместе), иначе модель «удовлетворяет» momentum ценой
        # искажения скорости и разбаланса массы (эффект v7: balance 17.6%).
        # Ядро w_nw = 1 + mult·exp(-sdf/width) (sdf нормирован на pipe_width)
        # домножает выбранные компоненты pde_raw (apply_to), не трогая BC.
        nw_cfg = lw.get('near_wall_pde', lw.get('near_wall_momentum', {}))
        self.nw_enabled = bool(nw_cfg.get('enabled', False)) if isinstance(nw_cfg, dict) else False
        self.nw_mult = float(nw_cfg.get('weight_mult', 8.0)) if isinstance(nw_cfg, dict) else 0.0
        self.nw_width = float(nw_cfg.get('width_norm', 0.2)) if isinstance(nw_cfg, dict) else 0.2
        nw_apply = nw_cfg.get('apply_to') if isinstance(nw_cfg, dict) else None
        if nw_apply is None:
            # совместимость с v7-ключом near_wall_momentum (только momentum)
            self.nw_apply_to = frozenset(['mom_x', 'mom_y'])
        else:
            self.nw_apply_to = frozenset(str(c).strip() for c in nw_apply)

        # Давление-Нейман на стенке (Этап 1): нормальная невязка momentum на
        # wall-точках замыкает pressure-Poisson (0.5·∂p/∂n = ν·(∇²u)·n) и убирает
        # гармоническую свободу p у стенок («двухслойность»). Нормали берутся из
        # data_handler.wall_normals (геометрия-агностично, по контуру области).
        wp_cfg = lw.get('wall_pressure', {})
        self.wp_enabled = bool(wp_cfg.get('enabled', False)) if isinstance(wp_cfg, dict) else False
        self.wp_weight = float(wp_cfg.get('weight', 10.0)) if isinstance(wp_cfg, dict) else 0.0
        self.wp_components = str(wp_cfg.get('components', 'normal')) if isinstance(wp_cfg, dict) else 'normal'
        self._wall_nx = None
        self._wall_ny = None
        self.wall_idx_in_bt = None
        wnorm = getattr(self.data_handler, 'wall_normals', None)
        if self.wp_enabled and wnorm is not None:
            self._wall_nx = torch.tensor(np.ascontiguousarray(wnorm[:, 0]),
                                         dtype=self.dtype, device=self.device).view(-1, 1)
            self._wall_ny = torch.tensor(np.ascontiguousarray(wnorm[:, 1]),
                                         dtype=self.dtype, device=self.device).view(-1, 1)
        names = self.data_handler.boundary_type_names
        if 'wall' in names:
            self.wall_idx_in_bt = names.index('wall')

    def _near_wall_kernel(self, sdf: torch.Tensor) -> torch.Tensor:
        """Возвращает (B,1) множитель весов PDE: больше у стенки (sdf→0)."""
        if not self.nw_enabled or self.nw_mult <= 0.0:
            return torch.ones_like(sdf)
        return (1.0 + self.nw_mult * torch.exp(-sdf / max(self.nw_width, 1e-6))).detach()

    def refresh_normals(self) -> None:
        """Пересоздаёт GPU-тензоры нормалей после расширения датасета (RAR)."""
        wnorm = getattr(self.data_handler, 'wall_normals', None)
        if wnorm is not None:
            self._wall_nx = torch.tensor(np.ascontiguousarray(wnorm[:, 0]),
                                         dtype=self.dtype, device=self.device).view(-1, 1)
            self._wall_ny = torch.tensor(np.ascontiguousarray(wnorm[:, 1]),
                                         dtype=self.dtype, device=self.device).view(-1, 1)

    def build_input_tensor(self, x: torch.Tensor, y: torch.Tensor,
                           sdf: torch.Tensor, bt: Optional[torch.Tensor] = None,
                           re_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Формирует входной тензор для модели.

        Базовый режим — ``(x, y, sdf)``. В параметрическом режиме добавляется
        4-й признак ``re_norm`` (безразмерная скорость/Re). ``bt`` на вход
        по-прежнему НЕ подаётся — он нужен только для масок BC.
        """
        if not getattr(self, 'parametric', False):
            return torch.cat([x, y, sdf], dim=1)
        if re_norm is None:
            re_norm = torch.ones_like(x)
        else:
            re_norm = re_norm.to(device=x.device, dtype=x.dtype).view(-1, 1)
        return torch.cat([x, y, sdf, re_norm], dim=1)

    def make_re_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Один скаляр ``re_norm`` на батч, размноженный по точкам.

        Один Re на батч (а не на точку) важен для согласованности BC: профиль
        на входе соответствует конкретному Re. Покрытие диапазона достигается
        разными батчами/эпохами.
        """
        if not getattr(self, 'parametric', False):
            return torch.ones_like(x)
        if self.re_max > self.re_min:
            value = torch.empty((1, 1), device=x.device, dtype=x.dtype).uniform_(
                self.re_min, self.re_max
            )
        else:
            value = torch.full((1, 1), self.re_default, device=x.device, dtype=x.dtype)
        return value.expand_as(x)

    def mu_from_re_norm(self, re_norm: torch.Tensor) -> torch.Tensor:
        """Per-point динамическая вязкость ``mu = rho / Re`` (``rho_norm`` = 1)."""
        reynolds = (re_norm * self.re_ref).clamp_min(1e-9)
        return self.rho_norm / reynolds

    def compute_data_loss(self, data_batch: Dict, model: nn.Module) -> Dict:
        """MSE по CFD-полям ``(u, v, p)`` на data-точках (data-loss)."""
        x = data_batch['x'].to(device=self.device, dtype=self.dtype)
        y = data_batch['y'].to(device=self.device, dtype=self.dtype)
        sdf = data_batch['sdf'].to(device=self.device, dtype=self.dtype)
        re_norm = data_batch['re_norm'].to(device=self.device, dtype=self.dtype)
        u_t = data_batch['u'].to(device=self.device, dtype=self.dtype)
        v_t = data_batch['v'].to(device=self.device, dtype=self.dtype)
        p_t = data_batch['p'].to(device=self.device, dtype=self.dtype)

        out = model(self.build_input_tensor(x, y, sdf, None, re_norm))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        loss_u = ((u - u_t) ** 2).mean()
        loss_v = ((v - v_t) ** 2).mean()
        loss_p = ((p - p_t) ** 2).mean()
        total = self.data_w_u * loss_u + self.data_w_v * loss_v + self.data_w_p * loss_p
        return {
            'loss': total,
            'data_u': float(loss_u.detach()),
            'data_v': float(loss_v.detach()),
            'data_p': float(loss_p.detach()),
            'data_total': float(total.detach()),
        }

    def compute_loss(self, batch: tuple, model: nn.Module,
                     pde_mult: float = 1.0, sa_weights: Optional[nn.Module] = None,
                     data_batch: Optional[Dict] = None) -> dict:
        """
        Вычисляет полную функцию потерь.

        Args:
            batch: кортеж (idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc)
            model: нейронная сеть
            pde_mult: множитель для PDE (для двухэтапного обучения)
            sa_weights: Self-Adaptive веса (опционально)

        Returns:
            словарь с ключами: loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max
        """
        idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch

        # Тензоры уже на правильном устройстве (GPU-батчер или DataLoader)
        # Дублируем requires_grad для autograd
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        # Приводим типы если нужно
        if x.dtype != self.dtype:
            x = x.to(dtype=self.dtype).requires_grad_(True)
            y = y.to(dtype=self.dtype).requires_grad_(True)
            sdf = sdf.to(dtype=self.dtype)
            if bt is not None and bt.shape[1] > 0:
                bt = bt.to(dtype=self.dtype)
            mask_u = mask_u.to(dtype=torch.bool)
            mask_v = mask_v.to(dtype=torch.bool)
            mask_p = mask_p.to(dtype=torch.bool)
            u_bc = u_bc.to(dtype=self.dtype)
            v_bc = v_bc.to(dtype=self.dtype)
            p_bc = p_bc.to(dtype=self.dtype)
        else:
            # Просто гарантируем bool маски
            mask_u = mask_u.to(dtype=torch.bool)
            mask_v = mask_v.to(dtype=torch.bool)
            mask_p = mask_p.to(dtype=torch.bool)

        # Гарантируем [B,1] формат
        mask_u = mask_u.view(-1, 1)
        mask_v = mask_v.view(-1, 1)
        mask_p = mask_p.view(-1, 1)
        u_bc = u_bc.view(-1, 1)
        v_bc = v_bc.view(-1, 1)
        p_bc = p_bc.view(-1, 1)

        # Параметр Re: один скаляр на батч (согласованные BC) и per-point ν.
        re_norm = self.make_re_norm(x)
        mu_batch = self.mu_from_re_norm(re_norm)
        # Профиль на inlet масштабируется как U_in/U_ref = re_norm. Для wall
        # (u_bc=0) и outlet (p_bc=0) масштабирование ничего не меняет.
        u_bc_eff = u_bc * re_norm
        v_bc_eff = v_bc * re_norm

        # Прямой проход
        input_tensor = self.build_input_tensor(x, y, sdf, bt, re_norm)
        out = model(input_tensor)
        u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # Применение hard constraints (если нужно)
        if self.bc_type == 'hard':
            u, v, p = self.apply_hard_constraints(u_raw, v_raw, p_raw,
                                                    mask_u, mask_v, mask_p,
                                                    u_bc_eff, v_bc_eff, p_bc)
        else:
            u, v, p = u_raw, v_raw, p_raw

        # PDE невязки
        pde_raw = PDEResiduals.compute_residuals_2d(
            u, v, p, x, y, self.rho_norm, mu_batch, safe_grad
        )

        # Граничные условия через boundary_types one-hot тензор
        bc_comp = self._compute_bc_loss(u, v, p, bt, mask_u, mask_v, mask_p,
                                         u_bc_eff, v_bc_eff, p_bc)

        # Взвешенная сумма PDE (SA-PINN: per-point веса масштабируют вклад точек).
        # Пристеночное ядро w_nw домножает выбранные компоненты pde_raw
        # (по умолчанию весь NS: cont+mom_x+mom_y, чтобы не «развязывать» массу).
        w_cont = self.pde_weights.get('cont', 1.0)
        w_mom_x = self.pde_weights.get('mom_x', 1.0)
        w_mom_y = self.pde_weights.get('mom_y', 1.0)
        w_nw = self._near_wall_kernel(sdf)   # (B,1), без градиента
        wcomp = {'cont': w_cont, 'mom_x': w_mom_x, 'mom_y': w_mom_y}
        pde_keys = ['cont', 'mom_x', 'mom_y']

        if sa_weights is not None:
            sa_out = sa_weights()                     # (N, 3) — per-point веса
            idx_dev = idx.to(self.device)             # DataLoader отдаёт idx на CPU
            wb = sa_out[idx_dev]                      # (B, 3) — веса для точек батча
            loss_pde = sum(
                (wb[:, ci:ci + 1] * wcomp[key] * (w_nw if key in self.nw_apply_to else 1.0) * pde_raw[key]).mean()
                for ci, key in enumerate(pde_keys)
            )
        else:
            loss_pde = sum(
                wcomp[key] * (pde_raw[key] * (w_nw if key in self.nw_apply_to else 1.0)).mean()
                for key in pde_keys
            )

        loss_bc = sum(bc_comp.values())

        # Pressure-Poisson: развивает градиент давления (часть PDE-слагаемых).
        if self.pp_weight > 0:
            pp_res = PDEResiduals.compute_pressure_poisson_residual(
                u, v, p, x, y, safe_grad
            )
            loss_pde = loss_pde + self.pp_weight * pp_res.mean()

        # Давление-Нейман на стенке (замыкание pressure-Poisson). Режимы:
        #  'normal' — нормальная проекция momentum (0.5·∇p − ν·∇²u)·n;
        #  'full'    — обе компоненты momentum на wall-точках (задаёт и ∂p/∂n,
        #             и ∂p/∂s, «протягивающий» уровень давления вдоль стенки).
        # Считаем на всём батче (у не-wall строк нормаль=0 — они не войдут в маску).
        if self.wp_weight > 0 and self.wall_idx_in_bt is not None \
                and bt is not None and bt.shape[1] > self.wall_idx_in_bt:
            wm = bt[:, self.wall_idx_in_bt] > 0.5
            if bool(wm.any()):
                if self.wp_components == 'full':
                    wp_res = pde_raw['mom_x'] + pde_raw['mom_y']
                elif self._wall_nx is not None:
                    idx_dev = idx.to(self.device)
                    nx_b = self._wall_nx[idx_dev]
                    ny_b = self._wall_ny[idx_dev]
                    wp_res = PDEResiduals.compute_wall_normal_momentum_residual(
                        u, v, p, x, y, nx_b, ny_b,
                        self.rho_norm, mu_batch, safe_grad,
                    )
                else:
                    wp_res = None
                if wp_res is not None:
                    loss_pde = loss_pde + self.wp_weight * wp_res[wm].mean()

        # Слабая привязка к начальному полю (функция тока) на внутренних точках:
        # якорь против дрейфа модели в нефизичный минимум.
        if self.reference_weight > 0 and bt is not None and bt.shape[1] > 0:
            interior = (bt.sum(dim=1, keepdim=True) < 0.5)
            if bool(interior.any()):
                n_int = interior.sum().clamp_min(1)
                loss_ref = (((u - u_bc_eff) ** 2) * interior).sum() / n_int
                loss_ref = loss_ref + (((v - v_bc_eff) ** 2) * interior).sum() / n_int
                loss_bc = loss_bc + self.reference_weight * loss_ref

        # Интегральное условие (баланс масс)
        integral_loss = 0.0
        if self.integral_condition and self.integral_weight > 0:
            integral_loss = self.compute_integral_loss(u, v, bt)
            loss_bc = loss_bc + self.integral_weight * integral_loss

        loss = pde_mult * loss_pde + loss_bc

        # Data-loss по CFD-полям (включается/отключается через
        # loss_weights.data_loss.enabled). Не масштабируется pde_mult.
        data_loss_value = 0.0
        data_comp: Dict[str, float] = {}
        if self.data_enabled and data_batch is not None:
            data_result = self.compute_data_loss(data_batch, model)
            data_loss_value = data_result['loss']
            loss = loss + self.data_weight * data_loss_value
            data_comp = {k: v for k, v in data_result.items() if k != 'loss'}

        # Скорость на стенке (для мониторинга)
        wall_speed = self._compute_wall_speed(u, v, bt)

        max_res = max(
            pde_raw['cont'].max().item(),
            pde_raw['mom_x'].max().item(),
            pde_raw['mom_y'].max().item()
        )

        return {
            'loss': loss,
            'pde_raw': {k: v.mean().item() for k, v in pde_raw.items()},
            'bc_comp': {k: float(v.item() if hasattr(v, 'item') else v) for k, v in bc_comp.items()},
            'integral_loss': integral_loss.item() if hasattr(integral_loss, 'item') else float(integral_loss),
            'wall_speed': wall_speed,
            'pde_max': max_res,
            'data_loss': float(data_loss_value.detach()) if torch.is_tensor(data_loss_value) else float(data_loss_value),
            'data_comp': data_comp,
        }

    def _compute_bc_loss(self, u, v, p, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc):
        """
        Вычисляет потери на граничных условиях через boundary_types one-hot тензор.

        Использует bt (boundary types one-hot) для определения типа каждой точки,
        а не эвристики по значениям скорости.
        """
        bc_loss = {}
        weights = self.bc_weights

        # Если нет boundary_types, используем маски (fallback)
        if bt is None or bt.shape[1] == 0:
            return self._compute_bc_loss_fallback(u, v, p, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc)

        type_names = self.data_handler.boundary_type_names

        # Приводим all к плоскому виду
        flat = lambda x: x.view(-1)
        u_f = flat(u)
        v_f = flat(v)
        p_f = flat(p)
        u_bc_f = flat(u_bc)
        v_bc_f = flat(v_bc)
        p_bc_f = flat(p_bc)

        for i, bt_name in enumerate(type_names):
            if i >= bt.shape[1]:
                continue
            bt_mask = bt[:, i] > 0.5  # points of this boundary type

            if not bt_mask.any():
                continue

            bc_weight = weights.get(bt_name, 5000.0)

            if bt_name == 'inlet':
                # inlet: U = U_bc, V = V_bc
                loss_u = ((u_f[bt_mask] - u_bc_f[bt_mask]) ** 2).mean()
                loss_v = ((v_f[bt_mask] - v_bc_f[bt_mask]) ** 2).mean()
                bc_loss[bt_name] = bc_weight * (loss_u + loss_v)

            elif bt_name == 'outlet':
                # outlet: pressure-outlet, ТОЛЬКО P = P_bc (избыточное = 0).
                # Скорость на выходе свободна — профиль развивается из PDE.
                # Баланс массы обеспечивается integral_condition.
                loss_p_val = ((p_f[bt_mask] - p_bc_f[bt_mask]) ** 2).mean()
                if not torch.isnan(loss_p_val) and not torch.isinf(loss_p_val):
                    bc_loss[bt_name] = bc_weight * loss_p_val

            elif bt_name == 'wall':
                # wall: U = 0, V = 0 (условие прилипания)
                loss_u = (u_f[bt_mask] ** 2).mean()
                loss_v = (v_f[bt_mask] ** 2).mean()
                bc_loss[bt_name] = bc_weight * (loss_u + loss_v)

        return bc_loss

    def _compute_bc_loss_fallback(self, u, v, p, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc):
        """Fallback: вычисляет BC loss через маски (если нет boundary_types)."""
        bc_loss = {}
        weights = self.bc_weights
        type_names = self.data_handler.boundary_type_names

        flat = lambda x: x.view(-1)
        u_f, v_f, p_f = flat(u), flat(v), flat(p)
        u_bc_f, v_bc_f, p_bc_f = flat(u_bc), flat(v_bc), flat(p_bc)
        mu_f = flat(mask_u)
        mv_f = flat(mask_v)
        mp_f = flat(mask_p)

        # INLET: используем маску u_fixed (все inlet точки)
        if 'inlet' in type_names:
            inlet_mask = mu_f & mv_f
            if inlet_mask.any():
                bc_loss['inlet'] = weights.get('inlet', 5000.0) * (
                    ((u_f[inlet_mask] - u_bc_f[inlet_mask])**2).mean() +
                    ((v_f[inlet_mask] - v_bc_f[inlet_mask])**2).mean()
                )

        # OUTLET: pressure-outlet, только давление p_bc
        if 'outlet' in type_names:
            outlet_mask = mp_f
            if outlet_mask.any():
                outlet_loss = ((p_f[outlet_mask] - p_bc_f[outlet_mask])**2).mean()
                if not torch.isnan(outlet_loss) and not torch.isinf(outlet_loss):
                    bc_loss['outlet'] = weights.get('outlet', 5000.0) * outlet_loss

        # WALL: используем маску u_fixed & v_fixed (u_bc == 0)
        if 'wall' in type_names:
            wall_mask = mu_f & mv_f
            if wall_mask.any():
                bc_loss['wall'] = weights.get('wall', 5000.0) * (
                    (u_f[wall_mask]**2).mean() + (v_f[wall_mask]**2).mean()
                )

        return bc_loss

    def _compute_wall_speed(self, u, v, bt):
        """Вычисляет среднюю скорость на стенке для мониторинга."""
        if bt is None or bt.shape[1] == 0:
            return 0.0

        type_names = self.data_handler.boundary_type_names
        if 'wall' not in type_names:
            return 0.0

        wall_idx = type_names.index('wall')
        if wall_idx >= bt.shape[1]:
            return 0.0

        wall_mask = bt[:, wall_idx] > 0.5
        if not wall_mask.any():
            return 0.0

        speed = torch.sqrt(u[wall_mask, 0]**2 + v[wall_mask, 0]**2)
        return speed.mean().item()

    def compute_integral_loss(self, u, v, bt):
        """
        Вычисляет интегральное условие баланса масс.

        ∫(u·n) dS = 0 на всех границах (несжимаемая жидкость).
        Нормали (наружу области):
          - inlet (левая грань x=x_min): n = (-1, 0)  → u·n = -u
          - outlet (верхняя грань y=y_max): n = (0, +1) → u·n = +v
        Для канала постоянной ширины условие сводится к
          mean(u_inlet) == mean(v_outlet).
        """
        if bt is None or bt.shape[1] == 0:
            return torch.tensor(0.0, device=self.device)

        type_names = self.data_handler.boundary_type_names
        integral = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Поток через inlet: n=(-1,0), вклад в ∫u·n = -u
        if 'inlet' in type_names:
            inlet_idx = type_names.index('inlet')
            inlet_mask = bt[:, inlet_idx] > 0.5
            if inlet_mask.any():
                u_in = u[inlet_mask, 0]
                if torch.isfinite(u_in).all():
                    integral = integral - u_in.mean()

        # Поток через outlet: n=(0,+1), вклад в ∫u·n = +v
        if 'outlet' in type_names:
            outlet_idx = type_names.index('outlet')
            outlet_mask = bt[:, outlet_idx] > 0.5
            if outlet_mask.any():
                v_out = v[outlet_mask, 0]
                if torch.isfinite(v_out).all():
                    integral = integral + v_out.mean()

        return integral ** 2

    def apply_hard_constraints(self, u_raw, v_raw, p_raw,
                                mask_u, mask_v, mask_p, u_bc, v_bc, p_bc):
        """Применяет hard граничные условия."""
        u = torch.where(mask_u, u_bc, u_raw)
        v = torch.where(mask_v, v_bc, v_raw)
        p = torch.where(mask_p, p_bc, p_raw)
        return u, v, p

    def apply_velocity_dirichlet_constraint(self, u, v, bt):
        """Применяет velocity Dirichlet constraint (заглушка для совместимости)."""
        return u, v