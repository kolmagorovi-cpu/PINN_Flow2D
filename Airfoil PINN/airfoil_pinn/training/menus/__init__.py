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

__all__ = [
    "architecture_menu",
    "checkpoint_menu",
    "diagnostics_menu",
    "loss_function_menu",
    "output_menu",
    "process_settings",
    "print_submenu_footer",
    "print_submenu_header",
    "training_strategy_menu",
    "weights_menu",
]


def __getattr__(name: str):
    """
    Лениво экспортирует тяжёлые подменю.

    ``architecture_menu`` тянет построение схемы сети и matplotlib. Ленивый
    импорт позволяет общим helper-функциям меню оставаться лёгкими.
    """
    if name == "architecture_menu":
        from airfoil_pinn.training.menus.architecture import architecture_menu

        return architecture_menu
    if name in {"loss_function_menu", "weights_menu"}:
        from airfoil_pinn.training.menus.loss import (
            loss_function_menu,
            weights_menu,
        )

        return {
            "loss_function_menu": loss_function_menu,
            "weights_menu": weights_menu,
        }[name]
    if name == "training_strategy_menu":
        from airfoil_pinn.training.menus.strategy import training_strategy_menu

        return training_strategy_menu
    if name in {"checkpoint_menu", "diagnostics_menu", "output_menu"}:
        from airfoil_pinn.training.menus.results import (
            checkpoint_menu,
            diagnostics_menu,
            output_menu,
        )

        return {
            "checkpoint_menu": checkpoint_menu,
            "diagnostics_menu": diagnostics_menu,
            "output_menu": output_menu,
        }[name]
    if name == "process_settings":
        from airfoil_pinn.training.menus.process import process_settings

        return process_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
