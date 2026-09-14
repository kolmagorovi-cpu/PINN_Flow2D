from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

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

        # Уравнения импульса.
        # ВАЖНО: p здесь нормирован как Cp = (p-p_ref)/(0.5·ρ·U²), поэтому
        # коэффициент градиента давления равен 0.5 (см. контракт Airfoil PINN).
        # Раньше стояло (1/rho)*p_x = 1.0 — модель учила давление с половинным
        # градиентом (плоское p).
        nu = mu / rho
        mom_x = (u * u_x + v * u_y + 0.5 * p_x - nu * (u_xx + u_yy)) ** 2
        mom_y = (u * v_x + v * v_y + 0.5 * p_y - nu * (v_xx + v_yy)) ** 2

        return {'cont': cont, 'mom_x': mom_x, 'mom_y': mom_y}

    @staticmethod
    def compute_pressure_poisson_residual(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        grad_fn: Callable
    ) -> torch.Tensor:
        """Квадрат невязки pressure-Poisson: ∇²p + 2·∇·(u·∇u) = 0.

        Получено дивергенцией нормированных NS (p = Cp, коэффициент 0.5):
        u·∇u + 0.5·∇p − (1/Re)∇²u = 0  =>  0.5·∇²p = −∇·(u·∇u).
        Вязкий член (1/Re)·∇²(∇·u) опущен (мал, обнуляется для div u = 0).

        Терм связывает давление со скоростью и заставляет развиваться
        градиенту давления (устраняет «плоское p»).
        """
        u_x = grad_fn(u, x)
        u_y = grad_fn(u, y)
        v_x = grad_fn(v, x)
        v_y = grad_fn(v, y)
        p_x = grad_fn(p, x)
        p_y = grad_fn(p, y)

        p_xx = grad_fn(p_x, x)
        p_yy = grad_fn(p_y, y)

        u_xx = grad_fn(u_x, x)
        u_xy = grad_fn(u_x, y)
        u_yy = grad_fn(u_y, y)
        v_xx = grad_fn(v_x, x)
        v_xy = grad_fn(v_x, y)
        v_yy = grad_fn(v_y, y)

        # div(u·∇u) = ∂x(u u_x + v u_y) + ∂y(u v_x + v v_y)
        div_x = u_x**2 + u * u_xx + v_x * u_y + v * u_xy
        div_y = u_y * v_x + u * v_xy + v_y**2 + v * v_yy
        pp = p_xx + p_yy + 2.0 * (div_x + div_y)
        return pp.square()

    @staticmethod
    def compute_wall_normal_momentum_residual(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        nx: torch.Tensor,
        ny: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        grad_fn: Callable
    ) -> torch.Tensor:
        """Квадрат нормальной (к стенке) невязки momentum на стенке (Neumann для p).

        На стенке u=v=0, поэтому momentum сводится к балансу нормального
        градиента давления и вязкого напряжения:
            (0.5·∇p − ν·∇²u)·n = 0   ->   0.5·∂p/∂n = ν·(∇²u)·n.
        Это недостающее граничное условие для pressure-Poisson: без него
        p у стенки недоопределено (гармоническая свобода), что даёт ложные
        экстремумы/«двухслойность» у стенок. Вычисляется ПОЛНЫЙ momentum,
        спроецированный на нормаль n (для мягких BC u на стенке может быть ~0,
        конвективный член учтён на всякий случай).

        Args:
            u, v, p: поля скорости и давления (B,1)
            x, y: координаты (B,1)
            nx, ny: единичные нормали внутрь жидкости (B,1)
            rho, mu: плотность и вязкость (нормированные)
            grad_fn: функция градиента

        Returns:
            (B,1) квадрат нормальной невязки momentum.
        """
        u_x = grad_fn(u, x)
        u_y = grad_fn(u, y)
        v_x = grad_fn(v, x)
        v_y = grad_fn(v, y)
        p_x = grad_fn(p, x)
        p_y = grad_fn(p, y)
        u_xx = grad_fn(u_x, x)
        u_yy = grad_fn(u_y, y)
        v_xx = grad_fn(v_x, x)
        v_yy = grad_fn(v_y, y)

        nu = mu / rho
        rx = u * u_x + v * u_y + 0.5 * p_x - nu * (u_xx + u_yy)
        ry = u * v_x + v * v_y + 0.5 * p_y - nu * (v_xx + v_yy)
        rn = rx * nx + ry * ny
        return rn.square()