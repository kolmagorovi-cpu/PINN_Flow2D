#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивные меню обучения PINN.

Подпакет будет постепенно забирать разделы из ``learning_cli``: архитектуру,
веса, стратегию обучения, диагностику и настройки процесса. На первом шаге
здесь хранится общий вывод заголовков подменю.
"""

from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)

__all__ = ["print_submenu_footer", "print_submenu_header"]
