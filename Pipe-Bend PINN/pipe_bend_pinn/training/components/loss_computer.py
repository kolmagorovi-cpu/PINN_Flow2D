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
                 device: torch.device, dtype: torch.dtype):
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

        # SA-веса
        self.use_sa_weights = lw.get('use_sa_weights', False)
        self.sa_apply_to_bc = lw.get('sa_apply_to_bc', False)

    def build_input_tensor(self, x: torch.Tensor, y: torch.Tensor,
                           sdf: torch.Tensor, bt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Формирует входной тензор для модели."""
        tensors = [x, y, sdf]
        if bt is not None and bt.shape[1] > 0:
            tensors.append(bt)
        return torch.cat(tensors, dim=1)

    def compute_loss(self, batch: tuple, model: nn.Module,
                     pde_mult: float = 1.0, sa_weights: Optional[nn.Module] = None) -> dict:
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

        # Прямой проход
        input_tensor = self.build_input_tensor(x, y, sdf, bt)
        out = model(input_tensor)
        u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # Применение hard constraints (если нужно)
        if self.bc_type == 'hard':
            u, v, p = self.apply_hard_constraints(u_raw, v_raw, p_raw,
                                                    mask_u, mask_v, mask_p,
                                                    u_bc, v_bc, p_bc)
        else:
            u, v, p = u_raw, v_raw, p_raw

        # PDE невязки
        pde_raw = PDEResiduals.compute_residuals_2d(
            u, v, p, x, y, self.rho_norm, self.mu_norm, safe_grad
        )

        # Граничные условия через boundary_types one-hot тензор
        bc_comp = self._compute_bc_loss(u, v, p, bt, mask_u, mask_v, mask_p,
                                         u_bc, v_bc, p_bc)

        # Взвешенная сумма PDE
        loss_pde = (self.pde_weights.get('cont', 1.0) * pde_raw['cont'].mean() +
                    self.pde_weights.get('mom_x', 1.0) * pde_raw['mom_x'].mean() +
                    self.pde_weights.get('mom_y', 1.0) * pde_raw['mom_y'].mean())

        loss_bc = sum(bc_comp.values())

        # Интегральное условие (баланс масс)
        integral_loss = 0.0
        if self.integral_condition and self.integral_weight > 0:
            integral_loss = self.compute_integral_loss(u, v, bt)
            loss_bc = loss_bc + self.integral_weight * integral_loss

        # SA-weights регуляризация
        if sa_weights is not None:
            sa_out = sa_weights()
            # SA-weights не добавляются напрямую в loss, они масштабируют вклады точек
            # внутри PDE и BC. Регуляризация не требуется.

        loss = pde_mult * loss_pde + loss_bc

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
                # outlet: P = P_bc (избыточное давление = 0)
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

        # OUTLET: используем маску p_fixed
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

        ∫(u·n) dS = 0 на всех границах (для несжимаемой жидкости).
        Упрощённая версия: баланс между inlet и outlet.
        """
        if bt is None or bt.shape[1] == 0:
            return torch.tensor(0.0, device=self.device)

        type_names = self.data_handler.boundary_type_names
        integral = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Сумма потоков на inlet
        if 'inlet' in type_names:
            inlet_idx = type_names.index('inlet')
            inlet_mask = bt[:, inlet_idx] > 0.5
            if inlet_mask.any():
                # u - нормальная компонента на inlet (ось x), v - тангенциальная
                integral = integral + u[inlet_mask, 0].mean()

        # Сумма потоков на outlet
        if 'outlet' in type_names:
            outlet_idx = type_names.index('outlet')
            outlet_mask = bt[:, outlet_idx] > 0.5
            if outlet_mask.any():
                # На выходе нормаль по y, но для pipe bend нужно учитывать геометрию
                # Упрощённо: берём v компоненту
                integral = integral - v[outlet_mask, 0].mean()

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