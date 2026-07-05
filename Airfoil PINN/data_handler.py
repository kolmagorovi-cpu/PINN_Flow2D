#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт слоя загрузки данных.

Основная реализация перенесена в ``airfoil_pinn.data.handler``. Старое имя
``data_handler`` сохранено для текущего solver-кода и тестов.
"""

from airfoil_pinn.data.handler import *  # noqa: F401,F403
from airfoil_pinn.data.handler import __all__  # noqa: F401
