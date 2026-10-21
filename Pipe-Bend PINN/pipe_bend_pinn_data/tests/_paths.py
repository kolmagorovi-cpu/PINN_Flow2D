#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общие пути и эталонные значения для тестов ``pipe_bend_pinn_data``."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPORT_CSV = os.path.join(PROJECT_ROOT, "docs", "export.csv")
DOMAIN_NPY = os.path.join(
    PROJECT_ROOT, "workspace", "geometry", "domains", "pipe_bend_domain.npy"
)

# Жидкость «Жидкость ламинарная (Re=500)» + условия «Ламинарный_Re500».
LAMINAR_SCALES = dict(
    pipe_width=0.5,
    velocity=1.0,
    density=998.0,
    dynamic_viscosity=0.998,
    static_pressure=100000.0,
)

# Прямоугольный контур, охватывающий все точки export.csv (для механики SDF).
EXPORT_RECTANGLE = [
    [-3.2, -0.4],
    [0.4, -0.4],
    [0.4, 3.2],
    [-3.2, 3.2],
]
