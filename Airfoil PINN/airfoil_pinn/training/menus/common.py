#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Общие элементы консольных меню обучения.

Функции здесь намеренно простые: они задают единый формат подменю, а конкретные
разделы меню импортируют их без зависимости друг от друга.
"""

__all__ = ["print_submenu_footer", "print_submenu_header"]


def print_submenu_header(title: str) -> None:
    """Выводит заголовок подменю в формате ``Название {``."""
    print(f"\n{title} {{")


def print_submenu_footer() -> None:
    """Выводит закрывающую скобку подменю."""
    print("}")
