#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Слой данных Airfoil PINN.

Пакет объединяет загрузку датасетов, подготовку начальных полей и SDF-
диагностику. Старые модули верхнего уровня остаются тонкими обертками, чтобы
существующие сценарии продолжали работать во время рефакторинга.
"""

from airfoil_pinn.data.handler import DataHandler, MeshData, MeshPointAdapter, PINNDataset
from airfoil_pinn.data.initializer import Initializer, initialize_menu

__all__ = [
    "DataHandler",
    "Initializer",
    "MeshData",
    "MeshPointAdapter",
    "PINNDataset",
    "initialize_menu",
]
