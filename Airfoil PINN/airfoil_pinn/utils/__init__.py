#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакет утилит для проекта Airfoil PINN.

Экспортирует:
  - UI-утилиты (ui_utils): меню, ввод, очистка экрана
  - I/O-утилиты (io_utils): работа с файлами JSON
  - Вспомогательные функции (auxiliary_utils): seed, память, градиент

Модули внутри ``airfoil_pinn`` импортируют утилиты отсюда напрямую. Корневой
пакет ``utils`` оставлен только как совместимая обёртка для старых сценариев.
"""

from .ui_utils import (
    get_menu_choice,
    safe_input_number,
    confirm_action,
    input_with_default,
    clear_screen,
    print_header,
    pause_and_continue,
    Colors,
)
from .io_utils import (
    save_json,
    load_json,
    sanitize_filename,
    ensure_directory,
)
from .auxiliary_utils import (
    set_seed,
    check_memory_usage,
    safe_grad,
)

__all__ = [
    # UI
    'get_menu_choice',
    'safe_input_number',
    'confirm_action',
    'input_with_default',
    'clear_screen',
    'print_header',
    'pause_and_continue',
    'Colors',
    # I/O
    'save_json',
    'load_json',
    'sanitize_filename',
    'ensure_directory',
    # Вспомогательные
    'set_seed',
    'check_memory_usage',
    'safe_grad',
]
