#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт функций эталонного решения.

Основная реализация перенесена в ``airfoil_pinn.training.reference``. Старое
имя ``reference`` сохранено для текущего solver-кода.
"""

from airfoil_pinn.training.reference import *  # noqa: F401,F403
from airfoil_pinn.training.reference import __all__  # noqa: F401
