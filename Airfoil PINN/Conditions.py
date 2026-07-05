#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт менеджера граничных условий.

Основная реализация перенесена в ``airfoil_pinn.physics.conditions``. Старое
имя ``Conditions`` сохранено для текущих сценариев и модулей.
"""

from airfoil_pinn.physics.conditions import *  # noqa: F401,F403
from airfoil_pinn.physics.conditions import __all__  # noqa: F401


if __name__ == "__main__":
    conditions_menu()
