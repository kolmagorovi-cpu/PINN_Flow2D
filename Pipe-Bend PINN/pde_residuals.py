"""
Модуль для вычисления невязок уравнений Навье-Стокса (PDE residuals).
Поддерживает 2D плоские течения.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import torch
from typing import Callable


class PDEResiduals:
    """
    Статические методы для вычисления квадратов невязок PDE.
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
            u, v, p: поля скорости и давления
            x, y: координаты
            rho, mu: плотность и вязкость
            grad_fn: функция градиента

        Returns:
            словарь с ключами 'cont', 'mom_x', 'mom_y' – квадраты невязок
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