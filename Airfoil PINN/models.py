#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт моделей PINN.

Основная реализация перенесена в ``airfoil_pinn.training.models``. Старое имя
``models`` сохранено для текущего solver-кода и тестов.
"""

from airfoil_pinn.training.models import *  # noqa: F401,F403
from airfoil_pinn.training.models import __all__  # noqa: F401
