#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль дискретизации расчетной области.

Пакет отвечает за генерацию точек коллокации, классификацию границ и подготовку
артефактов в ``workspace/sampling``. Точки коллокации и FVM-сетка разделены по
подпапкам, чтобы их проще было искать и очищать независимо.
"""

from airfoil_pinn.sampling.core import (
    BoundaryType,
    CollocationPoint,
    CollocationPoints,
    S2DR,
    SamplerConfig,
)
from airfoil_pinn.sampling.normal_layer import (
    NormalLayerColumnsResult,
    NormalLayerConfig,
    NormalLayerResult,
    generate_normal_layer_columns,
    generate_normal_layer_points,
)

__all__ = [
    "BoundaryType",
    "CollocationPoint",
    "CollocationPoints",
    "S2DR",
    "SamplerConfig",
    "NormalLayerColumnsResult",
    "NormalLayerConfig",
    "NormalLayerResult",
    "generate_normal_layer_columns",
    "generate_normal_layer_points",
]
