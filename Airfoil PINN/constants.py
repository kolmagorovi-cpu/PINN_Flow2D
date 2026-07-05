#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт констант проекта.

Основной файл констант перенесен в ``airfoil_pinn.core.constants``. Эта
обертка нужна на переходном этапе рефакторинга, чтобы существующие модули со
старыми импортами ``from constants import ...`` продолжали работать без
изменений.
"""

from airfoil_pinn.core.constants import *  # noqa: F401,F403
from airfoil_pinn.core.constants import __all__  # noqa: F401
