#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль loss_computer для проекта PINN Learning.
Содержит класс LossComputer, инкапсулирующий вычисления функции потерь:
- PDE невязки (через PDEResiduals)
- Граничные условия (soft/hard)
- Интегральное условие (баланс масс)
- Построение входного тензора для модели
- Применение hard constraints

Исправленная версия: метод compute_integral_loss переработан для корректного
учёта направления потока (нормали) и осесимметричной геометрии.
"""

import torch
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from airfoil_pinn.utils import safe_grad
from airfoil_pinn.physics.pde import PDEResiduals

if TYPE_CHECKING:
    from airfoil_pinn.physics.fvm import FVMResiduals

logger = logging.getLogger(__name__)

__all__ = ["LossComputer"]


class LossComputer:
    """
    Вычисляет все компоненты функции потерь для PINN.
    Инкапсулирует логику, связанную с PDE, граничными условиями и интегральными условиями.
    """

    def __init__(
        self,
        config: Dict,
        data_handler,
        rho_norm: torch.Tensor,
        mu_norm: torch.Tensor,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> None:
        """
        Args:
            config: полная конфигурация (словарь)
            data_handler: объект DataHandler с метаданными и информацией о границах
            rho_norm: нормализованная плотность (тензор-скаляр)
            mu_norm: нормализованная вязкость (тензор-скаляр)
            x_min, x_max, y_min, y_max: границы области (нормированные)
            device: устройство для вычислений
            dtype: тип данных
        """
        self.config = config
        self.data_handler = data_handler
        self.rho_norm = rho_norm
        self.mu_norm = mu_norm
        self.device = device
        self.dtype = dtype

        # Параметры из конфигурации
        self.flow_type = config['pde']['flow_type']
        self.pde_system = config['pde'].get('system', 'incompressible_navier_stokes')
        self.pde_on_boundary = bool(config['pde'].get('pde_on_boundary', False))
        self.bc_type = config['boundary_conditions']['type']  # 'soft' или 'hard'
        self.pde_components = self._setup_pde_components()
        self.strict_autograd = bool(
            config.get('diagnostics', {}).get('strict_autograd', True)
        )

        # Веса потерь
        lw = config['loss_weights']
        self.loss_weights = lw['pde_weights'].copy()
        if self.bc_type == 'soft':
            self.loss_weights.update(lw['bc_weights'])

        # SA-веса (флаг)
        self.use_sa_weights = lw['use_sa_weights']
        self.sa_apply_to_bc = lw['sa_apply_to_bc']

        # Адаптивная балансировка весов BC (опциональная функция, задаётся пользователем)
        self.adaptive_bc: bool = bool(lw.get('adaptive_bc', False))

        # Интегральное условие
        self.integral_condition = lw['integral_condition']
        self.integral_weight = lw['integral_weight']
        # Целевое значение интеграла (по умолчанию 0 - баланс масс)
        self.integral_target = lw.get('integral_target', 0.0)

        # FVM-потеря (закон сохранения массы на неструктурированной сетке)
        # Поддерживаемые форматы:
        #   dict с enabled/weight (новый): {'enabled': True, 'weight': 1.0}
        #   dict без enabled (старый):     {'weight': 0.5}  → вес берётся напрямую
        #   число/строка (legacy):         0.5
        fvm_cfg = lw.get('fvm', {})
        if isinstance(fvm_cfg, dict):
            if 'enabled' in fvm_cfg:
                fvm_on = bool(fvm_cfg['enabled'])
                self.fvm_weight: float = float(fvm_cfg.get('weight', 1.0)) if fvm_on else 0.0
            else:
                self.fvm_weight: float = float(fvm_cfg.get('weight', 0.0))
            self.fvm_batch_interval: int = max(1, int(fvm_cfg.get('batch_interval', 1)))
            self.fvm_loss_reduction: str = str(
                fvm_cfg.get(
                    'reduction',
                    fvm_cfg.get('loss_reduction', 'volume_weighted'),
                )
            )
        else:
            self.fvm_weight: float = float(fvm_cfg)
            self.fvm_batch_interval: int = 1
            self.fvm_loss_reduction: str = 'legacy'
        # Заполняется из PINNSolver при наличии mesh_file в конфиге
        self.fvm: Optional["FVMResiduals"] = None

        # Границы области для интегрального условия
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.tol = 0.01 * max(x_max - x_min, y_max - y_min)

        # Входные параметры (для построения тензора)
        self.input_params = config['input_params']

        # Для axisymmetric требуется eps и множитель 2π
        self.eps_axi = 1e-12
        self.two_pi = 2.0 * torch.pi

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

    # ------------------------------------------------------------------
    # FVM-потеря (закон сохранения массы)
    # ------------------------------------------------------------------
    def compute_fvm_loss(self, u_fvm: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет взвешенную FVM-потерю закона сохранения массы.

        Args:
            u_fvm: тензор (N, 2) — нормированная скорость [u, v] во всех
                   N узлах сетки (порядок совпадает с точками коллокации)

        Returns:
            Скалярный тензор fvm_weight * mean(R_i²),
            или ноль если fvm не подключён / вес равен нулю.
        """
        if self.fvm is None or self.fvm_weight == 0.0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return self.fvm_weight * self.fvm.mass_loss(
            u_fvm,
            reduction=self.fvm_loss_reduction,
        )

    def _setup_pde_components(self) -> List[str]:
        """Возвращает список названий компонент PDE в зависимости от типа течения."""
        if self.flow_type == 'axisymmetric':
            return ['cont', 'mom_r', 'mom_z']
        else:
            return ['cont', 'mom_x', 'mom_y']

    # ------------------------------------------------------------------
    # Вычисление PDE невязок
    # ------------------------------------------------------------------
    def compute_pde_residuals(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисляет невязки PDE (квадраты) для заданных полей.

        Args:
            u, v, p: поля скорости и давления (тензоры с requires_grad)
            x, y: координаты (тензоры с requires_grad)

        Returns:
            словарь с ключами, соответствующими self.pde_components,
            значениями – тензоры невязок той же формы, что и входные поля.
        """
        if self.pde_system == 'euler':
            if self.flow_type == 'axisymmetric':
                return PDEResiduals.compute_residuals_euler_axisymmetric(
                    u, v, p, x, y, self.rho_norm, self._grad, self.eps_axi
                )
            return PDEResiduals.compute_residuals_euler_2d(
                u, v, p, x, y, self.rho_norm, self._grad
            )

        if self.flow_type == 'axisymmetric':
            return PDEResiduals.compute_residuals_axisymmetric(
                u, v, p, x, y, self.rho_norm, self.mu_norm, self._grad, self.eps_axi
            )
        else:
            return PDEResiduals.compute_residuals_2d(
                u, v, p, x, y, self.rho_norm, self.mu_norm, self._grad
            )

    # ------------------------------------------------------------------
    # Граничные условия (только для soft)
    # ------------------------------------------------------------------
    def _freestream_velocity_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.mean((u[mask] - self.data_handler.inlet_u_norm) ** 2)
            + torch.mean((v[mask] - self.data_handler.inlet_v_norm) ** 2)
        )

    def _boundary_side_masks(
        self,
        mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left = mask & (x <= self.x_min + self.tol)
        right = mask & (x >= self.x_max - self.tol)
        bottom = mask & (y <= self.y_min + self.tol)
        top = mask & (y >= self.y_max - self.tol)
        return left, right, bottom, top

    def _normal_velocity_gradient_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        left, right, bottom, top = self._boundary_side_masks(mask, x, y)
        vertical = left | right
        horizontal = bottom | top
        terms = []

        if bool(vertical.any()):
            du_dx = self._grad(u, x, silent=False)
            dv_dx = self._grad(v, x, silent=False)
            terms.extend([du_dx[vertical] ** 2, dv_dx[vertical] ** 2])

        if bool(horizontal.any()):
            du_dy = self._grad(u, y, silent=False)
            dv_dy = self._grad(v, y, silent=False)
            terms.extend([du_dy[horizontal] ** 2, dv_dy[horizontal] ** 2])

        if not terms:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return torch.mean(torch.cat([term.flatten() for term in terms]))

    def _slip_bc_losses(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right, bottom, top = self._boundary_side_masks(mask, x, y)
        vertical = left | right
        horizontal = bottom | top

        normal_terms = []
        tangent_grad_terms = []
        velocity_anchor_terms = []

        if bool(horizontal.any()):
            normal_terms.append((v[horizontal] - self.data_handler.inlet_v_norm) ** 2)
            du_dy = self._grad(u, y, silent=False)
            tangent_grad_terms.append(du_dy[horizontal] ** 2)
            velocity_anchor_terms.append((u[horizontal] - self.data_handler.inlet_u_norm) ** 2)

        if bool(vertical.any()):
            normal_terms.append((u[vertical] - self.data_handler.inlet_u_norm) ** 2)
            dv_dx = self._grad(v, x, silent=False)
            tangent_grad_terms.append(dv_dx[vertical] ** 2)
            velocity_anchor_terms.append((v[vertical] - self.data_handler.inlet_v_norm) ** 2)

        if not normal_terms:
            normal_terms.append((v[mask] - self.data_handler.inlet_v_norm) ** 2)
            du_dy = self._grad(u, y, silent=False)
            tangent_grad_terms.append(du_dy[mask] ** 2)
            velocity_anchor_terms.append((u[mask] - self.data_handler.inlet_u_norm) ** 2)

        normal_loss = torch.mean(torch.cat([term.flatten() for term in normal_terms]))
        tangent_grad_loss = torch.mean(torch.cat([term.flatten() for term in tangent_grad_terms]))
        velocity_anchor_loss = torch.mean(torch.cat([term.flatten() for term in velocity_anchor_terms]))
        return normal_loss, tangent_grad_loss, velocity_anchor_loss

    def compute_bc_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Вычисляет потери на границах для soft boundary conditions.
        Возвращает словарь потерь по каждому типу границы и дополнительную
        потерю от градиентов на симметрии (symmetry_grad, axis_symmetry_grad).

        Args:
            u, v, p: поля скорости и давления
            boundary_types: one-hot тензор типов границ (или None)
            x, y: координаты

        Returns:
            bc_comp: словарь {имя_границы: тензор потерь (скаляр)}
            extra_grad: дополнительная потеря от градиентов (скалярный тензор)
        """
        if self.bc_type == 'hard' or boundary_types is None:
            # Для hard границ потери не вычисляются
            return {}, torch.tensor(0.0, device=self.device, dtype=self.dtype)

        bc_comp = {name: torch.tensor(0.0, device=self.device, dtype=self.dtype)
                   for name in self.data_handler.boundary_type_names}
        extra_grad = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for idx, name in enumerate(self.data_handler.boundary_type_names):
            mask = boundary_types[:, idx].view(-1, 1) > 0.5
            if mask.sum() == 0:
                continue

            if name == 'inlet':
                bc_comp[name] += self._freestream_velocity_loss(u, v, mask)

            elif name in ('farfield', 'velocity_dirichlet'):
                bc_comp[name] += self._freestream_velocity_loss(u, v, mask)

            elif name == 'outlet':
                p_out = p[mask]
                bc_comp[name] += torch.mean(p_out ** 2)
                outlet_grad_weight = self.loss_weights.get('outlet_grad', 0.0)
                if outlet_grad_weight > 0.0:
                    extra_grad += outlet_grad_weight * self._normal_velocity_gradient_loss(
                        u, v, x, y, mask
                    )

            elif name == 'wall':
                u_w = u[mask]
                v_w = v[mask]
                bc_comp[name] += torch.mean(u_w ** 2)
                bc_comp[name] += torch.mean(v_w ** 2)

            elif name == 'slip':
                normal_loss, tangent_grad_loss, velocity_anchor_loss = self._slip_bc_losses(
                    u, v, x, y, mask
                )
                bc_comp[name] += normal_loss
                extra_grad += (
                    self.loss_weights.get('slip_tangent_grad', 1.0) * tangent_grad_loss
                    + self.loss_weights.get('slip_velocity', 0.1) * velocity_anchor_loss
                )

            elif name == 'symmetry':
                bc_comp[name] += torch.mean(v[mask] ** 2)
                # du/dy = 0 на линии симметрии: градиент по полному тензору, затем маска
                du_dy = self._grad(u, y, silent=False)[mask]
                extra_grad += self.loss_weights.get('symmetry_grad', 10.0) * torch.mean(du_dy ** 2)

            elif name == 'axis_symmetry':
                bc_comp[name] += torch.mean(u[mask] ** 2)
                # dv/dr = 0 на оси: градиент по полному тензору, затем маска
                dv_dr = self._grad(v, x, silent=False)[mask]
                extra_grad += self.loss_weights.get('axis_symmetry_grad', 10.0) * torch.mean(dv_dr ** 2)

        return bc_comp, extra_grad

    # ------------------------------------------------------------------
    # Интегральное условие (баланс масс) – ИСПРАВЛЕННАЯ ВЕРСИЯ
    # ------------------------------------------------------------------
    def compute_mesh_integral_loss(
        self,
        velocity_nodes: torch.Tensor,
        bnd_node: torch.Tensor,
        bnd_area: torch.Tensor,
        bnd_type: torch.Tensor,
        node_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Интегральный баланс массы по FVM-граням.

        bnd_area хранит внешние векторы площадей/длин граней, поэтому сумма
        flux = u · S по inlet+outlet является signed outflow. Для корректного
        баланса целевое значение по умолчанию равно 0.
        """
        if not self.integral_condition:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if velocity_nodes.numel() == 0 or bnd_node.numel() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        try:
            from airfoil_pinn.physics.fvm import BC_CODES
            inlet_code = BC_CODES['inlet']
            outlet_code = BC_CODES['outlet']
        except Exception:
            inlet_code = 1
            outlet_code = 2

        device = velocity_nodes.device
        dtype = velocity_nodes.dtype
        bnd_node = bnd_node.to(device=device, dtype=torch.long)
        bnd_area = bnd_area.to(device=device, dtype=dtype)
        bnd_type = bnd_type.to(device=device, dtype=torch.long)

        mask = (bnd_type == inlet_code) | (bnd_type == outlet_code)
        if not bool(mask.any()):
            return torch.tensor(0.0, device=device, dtype=dtype)

        flux = (velocity_nodes[bnd_node] * bnd_area).sum(dim=1)
        if self.flow_type == 'axisymmetric' and node_x is not None:
            r = node_x.to(device=device, dtype=dtype).flatten()[bnd_node]
            two_pi = torch.as_tensor(self.two_pi, device=device, dtype=dtype)
            flux = flux * two_pi * torch.clamp(r, min=0.0)

        total_flux = flux[mask].sum()
        target = torch.as_tensor(self.integral_target, device=device, dtype=dtype)
        return (total_flux - target) ** 2

    def compute_integral_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет point-based интегральную потерю (баланс масс) по полному
        набору точек inlet/outlet. Метод не предназначен для случайного
        mini-batch: его вызывает Trainer только на полной boundary-выборке,
        если FVM-граней нет.

        Args:
            u, v: поля скорости
            boundary_types: one-hot тензор типов границ
            x, y: координаты

        Returns:
            скалярный тензор – квадрат отклонения суммарного потока от целевого значения
        """
        if not self.integral_condition or boundary_types is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Определяем индексы inlet и outlet
        if 'inlet' not in self.data_handler.boundary_type_names or 'outlet' not in self.data_handler.boundary_type_names:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        inlet_idx = self.data_handler.boundary_type_names.index('inlet')
        outlet_idx = self.data_handler.boundary_type_names.index('outlet')

        mask_inlet = boundary_types[:, inlet_idx].view(-1, 1) > 0.5
        mask_outlet = boundary_types[:, outlet_idx].view(-1, 1) > 0.5

        # Все точки, участвующие в интеграле
        mask_all = mask_inlet | mask_outlet
        if mask_all.sum() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        tol = self.tol
        x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max

        # Маски для четырёх сторон (с учётом допуска)
        # Разрешаем точкам принадлежать нескольким сторонам (углы будут учтены дважды,
        # что соответствует вкладу в поток через обе грани)
        left_mask = mask_all & (x < x_min + tol)
        right_mask = mask_all & (x > x_max - tol)
        bottom_mask = mask_all & (y < y_min + tol)
        top_mask = mask_all & (y > y_max - tol)

        x_flat = x.flatten()
        y_flat = y.flatten()
        u_flat = u.flatten()
        v_flat = v.flatten()
        total_flux = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        def side_flux(mask: torch.Tensor, normal_velocity: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
            flat_mask = mask.flatten()
            n = int(flat_mask.sum().item())
            if n == 0:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)

            vel = normal_velocity[flat_mask]
            tau = tangent[flat_mask]
            order = torch.argsort(tau)
            vel = vel[order]
            tau = tau[order]

            if n == 1:
                length = torch.as_tensor(
                    max(x_max - x_min, y_max - y_min),
                    device=self.device,
                    dtype=self.dtype,
                )
                weights = torch.ones_like(vel) * length
            else:
                dt = torch.clamp(tau[1:] - tau[:-1], min=0.0)
                weights = torch.zeros_like(vel)
                weights[0] = 0.5 * dt[0]
                weights[-1] = 0.5 * dt[-1]
                if n > 2:
                    weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

            if self.flow_type == 'axisymmetric':
                r = x_flat[flat_mask][order]
                two_pi = torch.as_tensor(self.two_pi, device=self.device, dtype=self.dtype)
                weights = weights * two_pi * torch.clamp(r, min=0.0)
            return torch.sum(vel * weights)

        total_flux = total_flux + side_flux(left_mask, -u_flat, y_flat)
        total_flux = total_flux + side_flux(right_mask, u_flat, y_flat)
        total_flux = total_flux + side_flux(bottom_mask, -v_flat, x_flat)
        total_flux = total_flux + side_flux(top_mask, v_flat, x_flat)

        # Потеря = (суммарный поток - целевой поток)^2
        target = torch.as_tensor(self.integral_target, device=self.device, dtype=self.dtype)
        loss = (total_flux - target) ** 2
        return loss

    # ------------------------------------------------------------------
    # Построение входного тензора для модели
    # ------------------------------------------------------------------
    def build_input_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
        boundary_types: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Собирает входной тензор для модели на основе координат, SDF,
        опционального one-hot типа границы и дополнительных параметров
        (скорость на входе, вязкость, плотность), если они включены
        в конфигурацию.

        Args:
            x, y: координаты (тензоры)
            sdf: нормализованное расстояние до стенки
            boundary_types: one-hot тензор типов границ (или None).
                Для loss-масок он передается всегда, но во вход модели
                попадает только при input_params.boundary_types=True.

        Returns:
            тензор формы (batch_size, input_dim)
        """
        inputs = []
        if self.input_params.get('coordinates', True):
            inputs.extend([x, y])
        if self.input_params.get('sdf', True):
            inputs.append(sdf)
        if (
            self.input_params.get('boundary_types', True)
            and boundary_types is not None
            and boundary_types.shape[1] > 0
        ):
            inputs.append(boundary_types)

        if self.input_params.get('inlet_velocity', False):
            u_in = torch.ones_like(x) * self.data_handler.inlet_u_norm
            v_in = torch.ones_like(x) * self.data_handler.inlet_v_norm
            inputs.append(u_in)
            inputs.append(v_in)

        if self.input_params.get('viscosity', False):
            mu = torch.ones_like(x) * self.mu_norm
            inputs.append(mu)

        if self.input_params.get('density', False):
            rho = torch.ones_like(x) * self.rho_norm
            inputs.append(rho)

        if not inputs:
            raise ValueError("build_input_tensor: не выбран ни один входной признак")

        return torch.cat(inputs, dim=1)

    # ------------------------------------------------------------------
    # Применение hard граничных условий
    # ------------------------------------------------------------------
    def apply_hard_constraints(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        p_raw: torch.Tensor,
        mask_u: torch.Tensor,
        mask_v: torch.Tensor,
        mask_p: torch.Tensor,
        u_bc: torch.Tensor,
        v_bc: torch.Tensor,
        p_bc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Заменяет предсказанные значения на граничные там, где маски активны.

        Args:
            u_raw, v_raw, p_raw: выход модели (без ограничений)
            mask_u, mask_v, mask_p: булевы маски, указывающие, где нужно заменить
            u_bc, v_bc, p_bc: значения, которые нужно подставить

        Returns:
            (u, v, p) с принудительно заданными граничными условиями
        """
        u_final = torch.where(mask_u, u_bc, u_raw)
        v_final = torch.where(mask_v, v_bc, v_raw)
        p_final = torch.where(mask_p, p_bc, p_raw)
        return u_final, v_final, p_final

    def apply_velocity_dirichlet_constraint(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard speed Dirichlet for boundary_type='velocity_dirichlet' in any BC mode."""
        names = getattr(self.data_handler, 'boundary_type_names', [])
        if boundary_types is None or 'velocity_dirichlet' not in names:
            return u_raw, v_raw

        idx = names.index('velocity_dirichlet')
        if idx >= boundary_types.shape[1]:
            return u_raw, v_raw

        mask = boundary_types[:, idx].view(-1, 1) > 0.5
        if not bool(mask.any()):
            return u_raw, v_raw

        u_final = torch.where(mask, torch.ones_like(u_raw) * self.data_handler.inlet_u_norm, u_raw)
        v_final = torch.where(mask, torch.ones_like(v_raw) * self.data_handler.inlet_v_norm, v_raw)
        return u_final, v_final

    # ------------------------------------------------------------------
    # Обновление адаптивных весов
    # ------------------------------------------------------------------
    def update_adaptive_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Обновляет веса граничных условий в соответствии с адаптивной балансировкой.
        Вызывается из Trainer.update_adaptive_weights.

        Args:
            new_weights: словарь {имя_границы: новое значение веса}
        """
        for key, value in new_weights.items():
            if key in self.loss_weights:
                self.loss_weights[key] = value
            else:
                logger.warning(f"Попытка обновить неизвестный вес {key}")
        logger.debug(f"Адаптивные веса BC обновлены внутри LossComputer")
