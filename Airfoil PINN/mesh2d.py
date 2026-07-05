#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт построителя FVM-сетки.

Основная реализация перенесена в ``airfoil_pinn.sampling.mesh``. Корневой
модуль сохранён, чтобы старые сценарии с ``from mesh2d import Mesh2D``
продолжали работать во время поэтапного рефакторинга.
"""

from airfoil_pinn.sampling.mesh import *  # noqa: F401,F403
from airfoil_pinn.sampling.mesh import __all__  # noqa: F401
