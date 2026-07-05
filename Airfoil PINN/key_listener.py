#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт клавиатурного слушателя.

Основная реализация перенесена в ``airfoil_pinn.training.key_listener``.
Корневой модуль сохранён для старых сценариев и внешних импортов.
"""

from airfoil_pinn.training.key_listener import *  # noqa: F401,F403
from airfoil_pinn.training.key_listener import __all__  # noqa: F401
