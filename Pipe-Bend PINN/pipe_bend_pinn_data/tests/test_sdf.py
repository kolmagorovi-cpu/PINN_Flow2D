#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты SDF и геометрических проверок."""

from __future__ import annotations

import unittest

import numpy as np

from ._paths import DOMAIN_NPY

from pipe_bend_pinn_data import (
    PolylineSDF,
    distance_to_polyline,
    load_domain_polyline,
    points_in_polygon,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


class SDFTests(unittest.TestCase):
    def test_distance_inside_square(self) -> None:
        # В центре квадрата расстояние до стенки = 0.5.
        distance = distance_to_polyline(np.array([[0.5, 0.5]]), SQUARE)
        self.assertAlmostEqual(float(distance[0]), 0.5, places=12)

    def test_distance_on_wall_is_zero(self) -> None:
        distance = distance_to_polyline(np.array([[0.0, 0.5], [1.0, 0.5]]), SQUARE)
        np.testing.assert_allclose(distance, [0.0, 0.0], atol=1e-12)

    def test_distance_outside_square(self) -> None:
        distance = distance_to_polyline(np.array([[2.0, 0.5]]), SQUARE)
        self.assertAlmostEqual(float(distance[0]), 1.0, places=12)

    def test_points_in_polygon(self) -> None:
        points = np.array([[0.5, 0.5], [2.0, 0.5], [0.0, 0.5]])
        inside = points_in_polygon(points, SQUARE)
        self.assertTrue(bool(inside[0]))
        self.assertFalse(bool(inside[1]))
        # Точка на границе: результат ray casting зависит от ориентации,
        # но не должен падать.
        self.assertIn(bool(inside[2]), (True, False))

    def test_polyline_sdf_wrapper(self) -> None:
        field = PolylineSDF(SQUARE)
        values = field.evaluate(np.array([[0.5, 0.5], [2.0, 0.5]]))
        np.testing.assert_allclose(values, [0.5, 1.0], atol=1e-12)

    def test_load_real_domain(self) -> None:
        vertices = load_domain_polyline(DOMAIN_NPY)
        self.assertGreaterEqual(vertices.shape[0], 3)
        self.assertTrue(np.isfinite(vertices).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
