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

__all__ = [
    "BoundaryType",
    "CollocationPoint",
    "CollocationPoints",
    "S2DR",
    "SamplerConfig",
]
