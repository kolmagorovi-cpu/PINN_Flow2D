#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимая точка входа пакетного запуска экспериментов.

Основная реализация перенесена в ``airfoil_pinn.training.batch``. Файл остаётся
для старых команд ``python batch_experiments.py`` и прямых импортов из этого модуля.
"""

from airfoil_pinn.training.batch import *  # noqa: F401,F403
from airfoil_pinn.training.batch import main


if __name__ == "__main__":
    main()
