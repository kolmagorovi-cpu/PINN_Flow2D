#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт менеджера чекпоинтов.

Основная реализация перенесена в ``airfoil_pinn.training.checkpoint``. Старое
имя ``checkpoint`` сохранено для текущего solver-кода.
"""

from airfoil_pinn.training.checkpoint import *  # noqa: F401,F403
from airfoil_pinn.training.checkpoint import __all__  # noqa: F401
