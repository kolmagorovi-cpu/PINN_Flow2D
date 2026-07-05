#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт SDF-диагностики.

Основная реализация перенесена в ``airfoil_pinn.data.sdf_diagnostics``. Старое
имя сохранено для модулей генерации точек и RAR-диагностики.
"""

from airfoil_pinn.data.sdf_diagnostics import *  # noqa: F401,F403
from airfoil_pinn.data.sdf_diagnostics import __all__  # noqa: F401
