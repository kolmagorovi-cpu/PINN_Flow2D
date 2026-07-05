#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт вычислителя PDE-невязок.

Реальная реализация находится в ``airfoil_pinn.physics.pde``. Обертка
сохраняет старые импорты ``from pde_residuals import PDEResiduals``.
"""

from airfoil_pinn.physics.pde import *  # noqa: F401,F403
from airfoil_pinn.physics.pde import __all__  # noqa: F401
