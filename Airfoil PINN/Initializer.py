#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт модуля инициализации.

Основная реализация перенесена в ``airfoil_pinn.data.initializer``. Старое имя
``Initializer`` сохранено для текущих сценариев и smoke-тестов.
"""

from airfoil_pinn.data.initializer import *  # noqa: F401,F403
from airfoil_pinn.data.initializer import __all__  # noqa: F401


if __name__ == "__main__":
    initialize_menu()
