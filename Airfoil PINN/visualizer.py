#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт визуализатора обучения.

Основная реализация перенесена в ``airfoil_pinn.training.visualizer``.
Корневой модуль сохранён для старых сценариев и внешних импортов.
"""

from airfoil_pinn.training.visualizer import *  # noqa: F401,F403
from airfoil_pinn.training.visualizer import __all__  # noqa: F401
