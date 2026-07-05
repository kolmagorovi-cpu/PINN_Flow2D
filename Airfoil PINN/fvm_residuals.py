#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт вычислителя FVM-невязок.

Реальная реализация находится в ``airfoil_pinn.physics.fvm``. Обертка
сохраняет старые импорты ``from fvm_residuals import FVMResiduals``.
"""

from airfoil_pinn.physics.fvm import *  # noqa: F401,F403
from airfoil_pinn.physics.fvm import __all__  # noqa: F401
