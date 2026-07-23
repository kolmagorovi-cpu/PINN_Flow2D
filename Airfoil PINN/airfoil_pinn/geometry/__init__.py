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
from airfoil_pinn.geometry.sdf import (
    DistanceFieldDiagnostics,
    PolylineDistanceField,
    differentiable_sdf_for_interior,
)

__all__ = [
    "AirfoilGeometry",
    "DistanceFieldDiagnostics",
    "Geometry2D",
    "GeometryConfig",
    "PolylineDistanceField",
    "differentiable_sdf_for_interior",
    "g2d",
    "load_geometry_from_config",
]
