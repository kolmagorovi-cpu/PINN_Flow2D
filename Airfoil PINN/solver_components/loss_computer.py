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
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from utils import safe_grad
from pde_residuals import PDEResiduals

logger = logging.getLogger(__name__)


class LossComputer:
    """
    Вычисляет все компоненты функции потерь для PINN.
    Инкапсулирует логику, связанную с PDE, граничными условиями и интегральными условиями.
    Поддерживает два режима:
      - Навье-Стокс (u, v, p)
      - Потенциал скорости (phi), u = ∂φ/∂x, v = ∂φ/∂y
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
    ):
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
        self.pde_system = config['pde'].get('system', 'incompressible_navier_stokes')
        self.flow_type = config['pde'].get('flow_type', '2d')
        self.bc_type = config['boundary_conditions']['type']  # 'soft' или 'hard'
        
        # Параметры для потенциала скорости
        self.use_potential = self.pde_system == 'potential'
        if self.use_potential:
            potential_cfg = config['pde'].get('potential', {})
            self.U_inf = potential_cfg.get('U_inf', 1.0)
            self.angle_of_attack = potential_cfg.get('angle_of_attack', 0.0)
            self.boundary_layer_thickness = potential_cfg.get('boundary_layer_thickness', 0.05)
            self.uniform_reg_weight = potential_cfg.get('uniform_reg_weight', 0.01)
            # Вычисляем компоненты скорости на бесконечности
            angle_rad = np.radians(self.angle_of_attack)
            self.U_inf_x = self.U_inf * np.cos(angle_rad)
            self.U_inf_y = self.U_inf * np.sin(angle_rad)
        
        self.pde_components = self._setup_pde_components()

        # Веса потерь
        lw = config['loss_weights']
        self.loss_weights = lw['pde_weights'].copy()
        if self.bc_type == 'soft':
            self.loss_weights.update(lw['bc_weights'])

        # SA-веса (флаг)
        self.use_sa_weights = lw['use_sa_weights']
        self.sa_apply_to_bc = lw['sa_apply_to_bc']

        # Интегральное условие
        self.integral_condition = lw['integral_condition']
        self.integral_weight = lw['integral_weight']
        # Целевое значение интеграла (по умолчанию 0 - баланс масс)
        self.integral_target = lw.get('integral_target', 0.0)

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

    def _setup_pde_components(self) -> List[str]:
        """Возвращает список названий компонент PDE в зависимости от типа течения."""
        if self.use_potential:
            return ['laplace']
        elif self.flow_type == 'axisymmetric':
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
        Для потенциала: принимает phi как u, но использует уже вычисленные u, v.

        Args:
            u: для НС - скорость по x, для потенциала - потенциал φ
            v: для НС - скорость по y, для потенциала - скорость ∂φ/∂y
            p: для НС - давление, для потенциала - игнорируется
            x, y: координаты (тензоры с requires_grad)

        Returns:
            словарь с ключами, соответствующими self.pde_components,
            значениями – тензоры невязок той же формы, что и входные поля.
        """
        if self.use_potential:
            # u здесь - это потенциал φ
            return PDEResiduals.compute_residuals_potential(
                u, x, y, safe_grad
            )
        elif self.flow_type == 'axisymmetric':
            return PDEResiduals.compute_residuals_axisymmetric(
                u, v, p, x, y, self.rho_norm, self.mu_norm, safe_grad, self.eps_axi
            )
        else:
            return PDEResiduals.compute_residuals_2d(
                u, v, p, x, y, self.rho_norm, self.mu_norm, safe_grad
            )

    # ------------------------------------------------------------------
    # Граничные условия (только для soft)
    # ------------------------------------------------------------------
    def compute_bc_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Вычисляет потери на границах для soft boundary conditions.
        Возвращает словарь потерь по каждому типу границы и дополнительную
        потерю от градиентов на симметрии (symmetry_grad, axis_symmetry_grad).
        
        Для потенциала скорости используются специальные граничные условия.

        Args:
            u: для НС - скорость по x, для потенциала - потенциал φ
            v: для НС - скорость по y, для потенциала - игнорируется
            p: для НС - давление, для потенциала - игнорируется
            boundary_types: one-hot тензор типов границ (или None)
            x, y: координаты
            sdf: нормализованное расстояние до стенки (для регуляризации потенциала)

        Returns:
            bc_comp: словарь {имя_границы: тензор потерь (скаляр)}
            extra_grad: дополнительная потеря от градиентов (скалярный тензор)
        """
        if self.bc_type == 'hard' or boundary_types is None:
            # Для hard границ потери не вычисляются
            return {}, torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.use_potential:
            return self._compute_potential_bc_loss(u, v, p, boundary_types, x, y, sdf)

        bc_comp = {name: torch.tensor(0.0, device=self.device, dtype=self.dtype)
                   for name in self.data_handler.boundary_type_names}
        extra_grad = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for name in self.data_handler.boundary_type_names:
            idx = self.data_handler.boundary_type_names.index(name)
            mask = boundary_types[:, idx].view(-1, 1) > 0.5
            if mask.sum() == 0:
                continue

            if name == 'inlet':
                u_in = u[mask]
                v_in = v[mask]
                bc_comp[name] += torch.mean((u_in - self.data_handler.inlet_u_norm) ** 2)
                bc_comp[name] += torch.mean((v_in - self.data_handler.inlet_v_norm) ** 2)

            elif name == 'outlet':
                p_out = p[mask]
                bc_comp[name] += torch.mean(p_out ** 2)

            elif name == 'wall':
                u_w = u[mask]
                v_w = v[mask]
                bc_comp[name] += torch.mean(u_w ** 2)
                bc_comp[name] += torch.mean(v_w ** 2)

            elif name == 'symmetry':
                v_sym = v[mask]
                bc_comp[name] += torch.mean(v_sym ** 2)
                # Дополнительное условие: du/dy = 0 на линии симметрии
                # Исправление: вычисляем градиент по исходному тензору y, затем выбираем точки симметрии
                u_sym = u[mask]
                # Градиент вычисляется по исходному тензору y (который имеет requires_grad=True)
                du_dy_full = safe_grad(u, y, silent=False)
                du_dy = du_dy_full[mask]
                extra_grad += self.loss_weights.get('symmetry_grad', 10.0) * torch.mean(du_dy ** 2)

            elif name == 'axis_symmetry':
                u_ax = u[mask]
                bc_comp[name] += torch.mean(u_ax ** 2)
                # Дополнительное условие: dv/dr = 0 на оси
                # Исправление: вычисляем градиент по исходному тензору x, затем выбираем точки оси
                v_ax = v[mask]
                dv_dr_full = safe_grad(v, x, silent=False)
                dv_dr = dv_dr_full[mask]
                extra_grad += self.loss_weights.get('axis_symmetry_grad', 10.0) * torch.mean(dv_dr ** 2)

        return bc_comp, extra_grad

    # ------------------------------------------------------------------
    # Граничные условия для потенциала скорости
    # ------------------------------------------------------------------
    def _compute_potential_bc_loss(
        self,
        phi: torch.Tensor,
        v_unused: torch.Tensor,
        p_unused: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Вычисляет потери на границах для потенциала скорости.
        
        Граничные условия:
          - inlet: ∂φ/∂x = U_inf_x, ∂φ/∂y = U_inf_y (заданная скорость)
          - wall: ∂φ/∂n = 0 (непротекание, нормальная производная = 0)
          - outlet: ∂φ/∂n = 0 (естественное условие)
          - symmetry: ∂φ/∂n = 0 (симметрия)
        
        Дополнительно: регуляризация вдали от стенки (стремление к однородному потоку)

        Args:
            phi: потенциал скорости φ
            v_unused, p_unused: игнорируются
            boundary_types: one-hot тензор типов границ
            x, y: координаты
            sdf: нормализованное расстояние до стенки

        Returns:
            bc_comp: словарь {имя_границы: тензор потерь}
            extra_grad: дополнительная потеря (регуляризация)
        """
        bc_comp = {}
        extra_grad = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        if boundary_types is None:
            return bc_comp, extra_grad

        # Вычисляем градиенты потенциала (скорости)
        phi_x = safe_grad(phi, x)  # u = ∂φ/∂x
        phi_y = safe_grad(phi, y)  # v = ∂φ/∂y

        for name in self.data_handler.boundary_type_names:
            idx = self.data_handler.boundary_type_names.index(name)
            mask = boundary_types[:, idx].view(-1, 1) > 0.5
            if mask.sum() == 0:
                continue

            bc_key = f'{name}_phi'
            
            if name == 'inlet':
                # На входе: скорость должна равняться U_inf
                # (∂φ/∂x - U_inf_x)² + (∂φ/∂y - U_inf_y)²
                u_in = phi_x[mask]
                v_in = phi_y[mask]
                bc_comp[bc_key] = (
                    torch.mean((u_in - self.U_inf_x) ** 2) +
                    torch.mean((v_in - self.U_inf_y) ** 2)
                )

            elif name == 'wall':
                # На стенке: непротекание, ∂φ/∂n = 0
                # Нормаль вычисляется через градиент SDF: n = ∇SDF / |∇SDF|
                # ∂φ/∂n = ∇φ · n = (∂φ/∂x * n_x + ∂φ/∂y * n_y)
                
                u_w = phi_x[mask]
                v_w = phi_y[mask]
                
                # Если есть SDF, вычисляем точную нормаль
                if sdf is not None:
                    # Градиент SDF даёт нормаль
                    sdf_x = safe_grad(sdf, x)
                    sdf_y = safe_grad(sdf, y)
                    
                    # Нормализуем: n = (sdf_x, sdf_y) / |(sdf_x, sdf_y)|
                    grad_norm = torch.sqrt(sdf_x[mask]**2 + sdf_y[mask]**2) + 1e-8
                    n_x = sdf_x[mask] / grad_norm
                    n_y = sdf_y[mask] / grad_norm
                    
                    # ∂φ/∂n = ∇φ · n
                    dphi_dn = u_w * n_x + v_w * n_y
                    bc_comp[bc_key] = torch.mean(dphi_dn ** 2)
                else:
                    # Если SDF нет, используем приближение: |∇φ|² = 0
                    bc_comp[bc_key] = torch.mean(u_w ** 2) + torch.mean(v_w ** 2)

            elif name == 'outlet':
                # На выходе: естественное условие, ∂φ/∂n = 0
                # Если есть SDF, используем его для вычисления нормали
                # Иначе приближённо считаем, что нормаль направлена по x
                u_out = phi_x[mask]
                
                if sdf is not None and self.flow_type != 'axisymmetric':
                    # Вычисляем нормаль через SDF
                    sdf_x = safe_grad(sdf, x)
                    sdf_y = safe_grad(sdf, y)
                    grad_norm = torch.sqrt(sdf_x[mask]**2 + sdf_y[mask]**2) + 1e-8
                    n_x = sdf_x[mask] / grad_norm
                    n_y = sdf_y[mask] / grad_norm
                    
                    v_out = phi_y[mask]
                    dphi_dn = u_out * n_x + v_out * n_y
                    bc_comp[bc_key] = torch.mean(dphi_dn ** 2)
                else:
                    # Приближённо: ∂φ/∂x = 0
                    bc_comp[bc_key] = torch.mean(u_out ** 2)

            elif name == 'symmetry':
                # На симметрии: ∂φ/∂n = 0 (нормальная компонента = 0)
                # Для горизонтальной симметрии: v = ∂φ/∂y = 0
                v_sym = phi_y[mask]
                bc_comp[bc_key] = torch.mean(v_sym ** 2)
                # Дополнительно: du/dy = 0 на линии симметрии
                du_dy_full = safe_grad(phi_x, y, silent=False)
                du_dy = du_dy_full[mask]
                extra_grad += self.loss_weights.get('symmetry_grad', 10.0) * torch.mean(du_dy ** 2)

            elif name == 'axis_symmetry':
                # На оси симметрии: u = ∂φ/∂x = 0 (радиальная скорость = 0)
                u_ax = phi_x[mask]
                bc_comp[bc_key] = torch.mean(u_ax ** 2)

        # Регуляризация: стремление к однородному потоку вдали от стенки
        # Используется маска по SDF: точки с sdf > boundary_layer_thickness
        if sdf is not None and self.uniform_reg_weight > 0:
            # Создаём маску для точек вдали от пограничного слоя
            bl_mask = sdf.view(-1) > self.boundary_layer_thickness
            if bl_mask.sum() > 0:
                u_far = phi_x[bl_mask]
                v_far = phi_y[bl_mask]
                uniform_loss = (
                    torch.mean((u_far - self.U_inf_x) ** 2) +
                    torch.mean((v_far - self.U_inf_y) ** 2)
                )
                extra_grad += self.uniform_reg_weight * uniform_loss

        return bc_comp, extra_grad

    # ------------------------------------------------------------------
    # Интегральное условие (баланс масс) – ИСПРАВЛЕННАЯ ВЕРСИЯ
    # ------------------------------------------------------------------
    def compute_integral_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет интегральную потерю (баланс масс) с учётом всех границ,
        помеченных как 'inlet' или 'outlet'. Для каждой грани определяется
        внешняя нормаль, длина стороны и вес точки. В осесимметричном случае
        добавляется множитель 2πr.

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

        # Длины сторон
        L_x = x_max - x_min
        L_y = y_max - y_min

        # Подсчёт количества точек на каждой стороне
        N_left = left_mask.sum().item()
        N_right = right_mask.sum().item()
        N_bottom = bottom_mask.sum().item()
        N_top = top_mask.sum().item()

        # Если точек на какой-то стороне нет, обнуляем длину, чтобы избежать деления на ноль
        ds_left = L_y / N_left if N_left > 0 else 0.0
        ds_right = L_y / N_right if N_right > 0 else 0.0
        ds_bottom = L_x / N_bottom if N_bottom > 0 else 0.0
        ds_top = L_x / N_top if N_top > 0 else 0.0

        # Нормали (внешние)
        # left:   nx = -1, ny = 0
        # right:  nx =  1, ny = 0
        # bottom: nx =  0, ny = -1
        # top:    nx =  0, ny =  1

        total_flux = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Левая грань
        if N_left > 0:
            u_left = u[left_mask].squeeze()
            # Вклад: (u * (-1) + v * 0) * ds_left
            flux_left = -u_left * ds_left
            if self.flow_type == 'axisymmetric':
                r_left = x[left_mask].squeeze()
                flux_left = flux_left * self.two_pi * r_left
            total_flux = total_flux + flux_left.sum()

        # Правая грань
        if N_right > 0:
            u_right = u[right_mask].squeeze()
            flux_right = u_right * ds_right
            if self.flow_type == 'axisymmetric':
                r_right = x[right_mask].squeeze()
                flux_right = flux_right * self.two_pi * r_right
            total_flux = total_flux + flux_right.sum()

        # Нижняя грань
        if N_bottom > 0:
            v_bottom = v[bottom_mask].squeeze()
            flux_bottom = -v_bottom * ds_bottom
            if self.flow_type == 'axisymmetric':
                r_bottom = x[bottom_mask].squeeze()
                flux_bottom = flux_bottom * self.two_pi * r_bottom
            total_flux = total_flux + flux_bottom.sum()

        # Верхняя грань
        if N_top > 0:
            v_top = v[top_mask].squeeze()
            flux_top = v_top * ds_top
            if self.flow_type == 'axisymmetric':
                r_top = x[top_mask].squeeze()
                flux_top = flux_top * self.two_pi * r_top
            total_flux = total_flux + flux_top.sum()

        # Потеря = (суммарный поток - целевой поток)^2
        loss = (total_flux - self.integral_target) ** 2
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
        one-hot границ и дополнительных параметров (скорость на входе,
        вязкость, плотность), если они включены в конфигурацию.

        Args:
            x, y: координаты (тензоры)
            sdf: нормализованное расстояние до стенки
            boundary_types: one-hot тензор типов границ (или None)

        Returns:
            тензор формы (batch_size, input_dim)
        """
        inputs = [x, y, sdf]
        if boundary_types is not None and boundary_types.shape[1] > 0:
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