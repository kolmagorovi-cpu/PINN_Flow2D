#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Слой данных Airfoil PINN.

Пакет объединяет загрузку датасетов, подготовку начальных полей и SDF-
диагностику. Старые модули верхнего уровня остаются тонкими обертками, чтобы
существующие сценарии продолжали работать во время рефакторинга.
"""

__all__ = [
    "DataHandler",
    "Initializer",
    "MeshData",
    "MeshPointAdapter",
    "PINNDataset",
    "initialize_menu",
]


def __getattr__(name: str):
    """
    Лениво экспортирует публичные классы слоя данных.

    ``Initializer`` зависит от консольных утилит и справочников условий. Если
    импортировать его сразу при загрузке пакета, точечные импорты вроде
    ``airfoil_pinn.data.sdf_diagnostics`` начинают подтягивать лишние модули.
    Ленивый экспорт оставляет пакет удобным, но не расширяет граф импортов.
    """
    if name in {"DataHandler", "MeshData", "MeshPointAdapter", "PINNDataset"}:
        from airfoil_pinn.data.handler import (
            DataHandler,
            MeshData,
            MeshPointAdapter,
            PINNDataset,
        )

        exports = {
            "DataHandler": DataHandler,
            "MeshData": MeshData,
            "MeshPointAdapter": MeshPointAdapter,
            "PINNDataset": PINNDataset,
        }
        return exports[name]

    if name in {"Initializer", "initialize_menu"}:
        from airfoil_pinn.data.initializer import Initializer, initialize_menu

        exports = {
            "Initializer": Initializer,
            "initialize_menu": initialize_menu,
        }
        return exports[name]

    raise AttributeError(f"module 'airfoil_pinn.data' has no attribute {name!r}")
