#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт менеджера конфигурации.

Основная реализация перенесена в ``airfoil_pinn.config.manager``. Старое имя
``config_manager`` сохранено для внешних сценариев и поэтапного перехода.
"""

from airfoil_pinn.config.manager import *  # noqa: F401,F403
from airfoil_pinn.config.manager import __all__  # noqa: F401
