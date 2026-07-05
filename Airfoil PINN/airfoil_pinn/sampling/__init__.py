#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль дискретизации расчетной области.

Пакет отвечает за генерацию точек коллокации, классификацию границ и подготовку
артефактов Sampler/Data. Старая точка импорта ``Collocation`` сохранена как
совместимая обертка.
"""

from airfoil_pinn.sampling.core import (
    BoundaryType,
    CollocationPoint,
    CollocationPoints,
    S2DR,
    SamplerConfig,
)

__all__ = [
    "BoundaryType",
    "CollocationPoint",
    "CollocationPoints",
    "S2DR",
    "SamplerConfig",
]
