#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Компоненты тренировочного процесса.

Пакет содержит вычисление функции потерь, цикл обучения, метрики, RAR,
оптимизаторы и интерактивные обработчики. Старый путь ``solver_components.*``
остается совместимым через тонкие обертки в корне проекта.
"""

__all__ = [
    "CyclicLR",
    "InteractiveHandler",
    "LocalMinimaDetector",
    "LossComputer",
    "MetricsCollector",
    "RARManager",
    "Trainer",
]


def __getattr__(name: str):
    """
    Лениво отдает публичные компоненты обучения.

    Компоненты имеют разные зависимости: например, ``Trainer`` импортирует
    SDF-диагностику, а ``LossComputer`` нужен в изоляции для тестов формата
    весов. Ленивый экспорт не заставляет точечный импорт загружать весь
    тренировочный стек.
    """
    if name == "InteractiveHandler":
        from airfoil_pinn.training.components.interactive import InteractiveHandler

        return InteractiveHandler

    if name == "LossComputer":
        from airfoil_pinn.training.components.loss_computer import LossComputer

        return LossComputer

    if name == "MetricsCollector":
        from airfoil_pinn.training.components.metrics import MetricsCollector

        return MetricsCollector

    if name in {"CyclicLR", "LocalMinimaDetector"}:
        from airfoil_pinn.training.components.optimizers import CyclicLR, LocalMinimaDetector

        return {"CyclicLR": CyclicLR, "LocalMinimaDetector": LocalMinimaDetector}[name]

    if name == "RARManager":
        from airfoil_pinn.training.components.rar import RARManager

        return RARManager

    if name == "Trainer":
        from airfoil_pinn.training.components.trainer import Trainer

        return Trainer

    raise AttributeError(
        f"module 'airfoil_pinn.training.components' has no attribute {name!r}"
    )
