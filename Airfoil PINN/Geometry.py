#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт геометрического модуля.

Основная реализация перенесена в ``airfoil_pinn.geometry``. Этот файл сохранен,
чтобы старые сценарии ``import Geometry`` и динамический импорт стороннего кода
не ломались во время поэтапного рефакторинга.
"""

from airfoil_pinn.geometry import *  # noqa: F401,F403
from airfoil_pinn.geometry import __all__  # noqa: F401


if __name__ == "__main__":
    g2d()
