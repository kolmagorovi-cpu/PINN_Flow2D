#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Слой обучения Airfoil PINN.

Пакет постепенно собирает модели, чекпоинты, эталонные решения, solver и
тренировочные компоненты. На этом этапе перенесены только небольшие support-
модули, чтобы основной цикл обучения остался нетронутым.
"""

from airfoil_pinn.training.checkpoint import CheckpointManager
from airfoil_pinn.training.models import PINNIncompressible, SelfAdaptiveWeights, Sine

__all__ = [
    "CheckpointManager",
    "PINNIncompressible",
    "SelfAdaptiveWeights",
    "Sine",
]
