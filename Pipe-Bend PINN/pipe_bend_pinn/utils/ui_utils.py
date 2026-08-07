#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI-утилиты для Pipe-Bend PINN.

Содержит цвета для консольного вывода и вспомогательные функции навигации.
Адаптировано из airfoil_pinn.utils.ui_utils.
"""

from __future__ import annotations

import sys


class Colors:
    """Цвета для консольного вывода."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    if sys.platform == 'win32':
        GREEN = BLUE = YELLOW = RED = CYAN = MAGENTA = BOLD = UNDERLINE = RESET = ''


def clear_screen() -> None:
    """Очищает экран консоли."""
    import os
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print('\033c')


def get_menu_choice(prompt: str, valid_range: tuple[int, int] | None = None,
                    allow_zero: bool = False, zero_action: str = "назад") -> str:
    """Выводит запрос и возвращает выбор пользователя."""
    suffix = f" (0 - {zero_action})" if allow_zero else ""
    if valid_range:
        prompt = f"{prompt} [{valid_range[0]}-{valid_range[1]}{suffix}]: "
    else:
        prompt = f"{prompt}{suffix}: "
    return input(prompt).strip()


def pause_and_continue(message: str = "Нажмите Enter чтобы продолжить...") -> None:
    """Ожидает нажатия Enter."""
    input(message)
