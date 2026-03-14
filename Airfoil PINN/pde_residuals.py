# Исправленный файл: pde_residuals.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для вычисления невязок уравнений Навье-Стокса (PDE residuals).
Поддерживает 2D плоские и осесимметричные течения с полными вязкими членами.

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная, с полными вторыми производными для axisymmetric)
"""

import torch
from typing import Callable


class PDEResiduals:
    """
    Статические методы для вычисления квадратов невязок PDE.
    Все методы принимают функцию градиента grad_fn, которая вычисляет
    производную dy/dx (например, utils.safe_grad).
    """

    @staticmethod
    def compute_residuals_2d(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        grad_fn: Callable
    ) -> dict:
        """
        Вычисляет невязки для 2D плоского течения.

        Уравнения:
            неразрывность: ∂u/∂x + ∂v/∂y = 0
            импульс по x: u ∂u/∂x + v ∂u/∂y + 1/ρ ∂p/∂x - ν (∂²u/∂x² + ∂²u/∂y²) = 0
            импульс по y: u ∂v/∂x + v ∂v/∂y + 1/ρ ∂p/∂y - ν (∂²v/∂x² + ∂²v/∂y²) = 0

        Args:
            u, v, p: поля скорости и давления (тензоры с requires_grad=True)
            x, y: координаты (тензоры с requires_grad=True)
            rho, mu: плотность и вязкость (тензоры-скаляры)
            grad_fn: функция градиента (например, safe_grad)

        Returns:
            словарь с ключами 'cont', 'mom_x', 'mom_y' – квадраты невязок (тензоры той же формы)
        """
        # Первые производные
        u_x = grad_fn(u, x)
        u_y = grad_fn(u, y)
        v_x = grad_fn(v, x)
        v_y = grad_fn(v, y)
        p_x = grad_fn(p, x)
        p_y = grad_fn(p, y)

        # Вторые производные (для диффузии)
        u_xx = grad_fn(u_x, x)
        u_yy = grad_fn(u_y, y)
        v_xx = grad_fn(v_x, x)
        v_yy = grad_fn(v_y, y)

        # Неразрывность
        cont = (u_x + v_y) ** 2

        # Уравнения импульса
        nu = mu / rho
        mom_x = (u * u_x + v * u_y + (1.0 / rho) * p_x - nu * (u_xx + u_yy)) ** 2
        mom_y = (u * v_x + v * v_y + (1.0 / rho) * p_y - nu * (v_xx + v_yy)) ** 2

        return {'cont': cont, 'mom_x': mom_x, 'mom_y': mom_y}

    @staticmethod
    def compute_residuals_axisymmetric(
        u: torch.Tensor,      # радиальная скорость (u_r)
        v: torch.Tensor,      # осевая скорость (u_z)
        p: torch.Tensor,
        r: torch.Tensor,      # радиальная координата
        z: torch.Tensor,      # осевая координата
        rho: torch.Tensor,
        mu: torch.Tensor,
        grad_fn: Callable,
        eps: float = 1e-12
    ) -> dict:
        """
        Вычисляет невязки для осесимметричного течения (координаты r, z).

        Уравнения (полные, с учётом всех вязких членов в цилиндрических координатах):
            неразрывность:      (1/r) ∂(r u)/∂r + ∂v/∂z = 0   =>   u/r + ∂u/∂r + ∂v/∂z = 0
            импульс по r:       u ∂u/∂r + v ∂u/∂z + (1/ρ) ∂p/∂r -
                                ν [ ∂²u/∂r² + (1/r)∂u/∂r - u/r² + ∂²u/∂z² ] = 0
            импульс по z:       u ∂v/∂r + v ∂v/∂z + (1/ρ) ∂p/∂z -
                                ν [ ∂²v/∂r² + (1/r)∂v/∂r + ∂²v/∂z² ] = 0

        Args:
            u, v, p: поля скорости (u – радиальная, v – осевая) и давление
            r, z: координаты (r – радиальная, z – осевая)
            rho, mu: плотность и вязкость
            grad_fn: функция градиента
            eps: малое число для защиты от деления на ноль (при r=0)

        Returns:
            словарь с ключами 'cont', 'mom_r', 'mom_z' – квадраты невязок
        """
        # Производные первого порядка
        u_r = grad_fn(u, r)
        u_z = grad_fn(u, z)
        v_r = grad_fn(v, r)
        v_z = grad_fn(v, z)
        p_r = grad_fn(p, r)
        p_z = grad_fn(p, z)

        # Производные второго порядка
        u_rr = grad_fn(u_r, r)
        u_zz = grad_fn(u_z, z)
        v_rr = grad_fn(v_r, r)
        v_zz = grad_fn(v_z, z)

        # Защита от деления на ноль на оси
        r_safe = r + eps

        # Уравнение неразрывности: u/r + u_r + v_z = 0
        cont = (u / r_safe + u_r + v_z) ** 2

        # Кинематическая вязкость
        nu = mu / rho

        # Лапласиан для радиальной компоненты в цилиндрических координатах
        laplacian_u = u_rr + (1.0 / r_safe) * u_r - u / (r_safe ** 2) + u_zz
        mom_r = (u * u_r + v * u_z + (1.0 / rho) * p_r - nu * laplacian_u) ** 2

        # Лапласиан для осевой компоненты
        laplacian_v = v_rr + (1.0 / r_safe) * v_r + v_zz
        mom_z = (u * v_r + v * v_z + (1.0 / rho) * p_z - nu * laplacian_v) ** 2

        return {'cont': cont, 'mom_r': mom_r, 'mom_z': mom_z}