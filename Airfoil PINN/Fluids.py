#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт справочника жидкостей.

Основная реализация перенесена в ``airfoil_pinn.physics.fluids``. Старое имя
``Fluids`` сохранено для существующих модулей и прямого запуска.
"""

from airfoil_pinn.physics.fluids import *  # noqa: F401,F403
from airfoil_pinn.physics.fluids import __all__  # noqa: F401


if __name__ == "__main__":
    fluids_menu()
