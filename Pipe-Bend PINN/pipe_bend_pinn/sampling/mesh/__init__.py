#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение FVM-сетки для точек семплинга.

Подпакет отделяет генерацию коллокационных точек от построения
boundary-fitted Mesh2D, которая нужна FVM-невязкам и пересборке сетки после RAR.
"""

from pipe_bend_pinn.sampling.mesh.core import (
    BoundaryFace,
    InternalFace,
    Mesh2D,
    MeshNode,
    build_pipe_bend_mesh,
)

__all__ = ["BoundaryFace", "InternalFace", "Mesh2D", "MeshNode", "build_pipe_bend_mesh"]
