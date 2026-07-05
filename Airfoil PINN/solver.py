#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт основного PINN-решателя.

Основная реализация перенесена в ``airfoil_pinn.training.solver``. Старое имя
``solver`` сохранено для ``PINN_Learning.py`` и внешних сценариев.
"""

from airfoil_pinn.training.solver import *  # noqa: F401,F403
from airfoil_pinn.training.solver import __all__  # noqa: F401
