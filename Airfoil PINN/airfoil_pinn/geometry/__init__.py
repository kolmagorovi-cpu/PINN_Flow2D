#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Геометрический модуль Airfoil PINN.

Пакет отвечает за загрузку профиля, построение расчетной области, сохранение
геометрических артефактов и консольный сценарий G2D. Внешний код должен
постепенно переходить на импорты из ``airfoil_pinn.geometry`` вместо старого
модуля ``Geometry``.
"""

from airfoil_pinn.geometry.core import (
    AirfoilGeometry,
    Geometry2D,
    GeometryConfig,
    g2d,
    load_geometry_from_config,
)

__all__ = [
    "AirfoilGeometry",
    "Geometry2D",
    "GeometryConfig",
    "g2d",
    "load_geometry_from_config",
]
