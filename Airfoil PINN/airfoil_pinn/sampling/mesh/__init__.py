#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение FVM-сетки для точек семплинга.

Подпакет отделяет генерацию коллокационных точек от построения
boundary-fitted Mesh2D, которая нужна FVM-невязкам и пересборке сетки после RAR.
"""

from airfoil_pinn.sampling.mesh.core import (
    BoundaryFace,
    InternalFace,
    Mesh2D,
    MeshNode,
    build_dual_face_quadrature,
)

__all__ = [
    "BoundaryFace",
    "InternalFace",
    "Mesh2D",
    "MeshNode",
    "build_dual_face_quadrature",
]
