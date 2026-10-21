#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вычисление SDF относительно контура стенки (без autograd).

Для CFD-точек достаточно предвычислить беззнаковое расстояние до полилинии
стенки: внутри канала оно совпадает с внутренней ветвью SDF, на стенке = 0.
Реализация повторяет математику
``pipe_bend_pinn.geometry.sdf.PolylineDistanceField``, но на numpy.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "load_domain_polyline",
    "distance_to_polyline",
    "points_in_polygon",
    "PolylineSDF",
]


def load_domain_polyline(path: str | os.PathLike[str]) -> np.ndarray:
    """Загружает контур области из ``.npy`` (``(N, 2)`` или structured)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл геометрии не найден: {path}")
    domain = np.load(path, allow_pickle=True)
    if domain.ndim == 1 and domain.dtype.names is not None:
        if "x" not in domain.dtype.names or "y" not in domain.dtype.names:
            raise ValueError("structured-геометрия должна содержать поля x и y")
        vertices = np.column_stack((domain["x"], domain["y"]))
    else:
        vertices = np.asarray(domain, dtype=np.float64)
    vertices = vertices.reshape(-1, 2)
    if vertices.shape[0] < 3:
        raise ValueError("контур должен содержать минимум 3 вершины")
    if not np.isfinite(vertices).all():
        raise ValueError("контур содержит NaN/Inf")
    return vertices


def distance_to_polyline(
    points: np.ndarray,
    polyline: np.ndarray,
    *,
    closed: bool = True,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Беззнаковое расстояние от точек до ломаной (векторно, с чанкингом)."""
    query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    vertices = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if vertices.shape[0] < 2:
        raise ValueError("polyline должен содержать минимум 2 вершины")
    if closed and not np.allclose(vertices[0], vertices[-1], atol=1e-12, rtol=0.0):
        vertices = np.vstack([vertices, vertices[0:1]])

    starts = vertices[:-1]
    ends = vertices[1:]
    vectors = ends - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    valid = squared_lengths > 1e-30
    starts, vectors, squared_lengths = starts[valid], vectors[valid], squared_lengths[valid]
    if not np.any(valid):
        raise ValueError("polyline не содержит невырожденных сегментов")

    out = np.empty(query.shape[0], dtype=np.float64)
    for begin in range(0, query.shape[0], chunk_size):
        stop = min(begin + chunk_size, query.shape[0])
        block = query[begin:stop]
        relative = block[:, None, :] - starts[None, :, :]
        projection = np.sum(relative * vectors[None, :, :], axis=-1)
        projection = np.clip(projection / squared_lengths[None, :], 0.0, 1.0)
        closest = starts[None, :, :] + projection[:, :, None] * vectors[None, :, :]
        delta = block[:, None, :] - closest
        out[begin:stop] = np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=1))
    return out


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Маска попадания точек в замкнутый полигон (ray casting)."""
    query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ring = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if not np.allclose(ring[0], ring[-1], atol=1e-12, rtol=0.0):
        ring = np.vstack([ring, ring[0:1]])

    x = query[:, 0]
    y = query[:, 1]
    inside = np.zeros(query.shape[0], dtype=bool)
    for index in range(ring.shape[0] - 1):
        x_i, y_i = ring[index]
        x_j, y_j = ring[index + 1]
        crosses = (y_i > y) != (y_j > y)
        if not np.any(crosses):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x_i + (x_j - x_i) * (y - y_i) / (y_j - y_i)
        inside ^= crosses & (x < x_cross)
    return inside


class PolylineSDF:
    """Беззнаковое расстояние до контура стенки (в тех же единицах, что вход)."""

    def __init__(
        self,
        polyline: np.ndarray,
        *,
        closed: bool = True,
        chunk_size: int = 4096,
    ) -> None:
        self.polyline = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        self.closed = bool(closed)
        self.chunk_size = int(chunk_size)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return distance_to_polyline(
            points,
            self.polyline,
            closed=self.closed,
            chunk_size=self.chunk_size,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        return points_in_polygon(points, self.polyline)

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return (
            f"PolylineSDF(vertices={self.polyline.shape[0]}, "
            f"closed={self.closed}, chunk_size={self.chunk_size})"
        )
