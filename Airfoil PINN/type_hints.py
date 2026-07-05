#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт типизированных структур проекта.

TypedDict-описания перенесены в ``airfoil_pinn.core.type_hints``. Старая точка
импорта сохранена, чтобы геометрия, дискретизация, инициализация и менеджеры
условий могли переезжать в пакет поэтапно.
"""

from airfoil_pinn.core.type_hints import *  # noqa: F401,F403
from airfoil_pinn.core.type_hints import __all__  # noqa: F401
