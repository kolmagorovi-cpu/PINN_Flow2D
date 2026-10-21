#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты разбора CFD-экспорта."""

from __future__ import annotations

import unittest

import numpy as np

from ._paths import EXPORT_CSV

from pipe_bend_pinn_data import UnsetColumnError, load_cfd_export, parse_cfd_export


class CFDLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_cfd_export(EXPORT_CSV)

    def test_point_count(self) -> None:
        self.assertEqual(self.raw.n_points, 3402)

    def test_coordinate_ranges(self) -> None:
        self.assertAlmostEqual(float(self.raw.x.min()), -3.0, places=6)
        self.assertAlmostEqual(float(self.raw.x.max()), 0.25, places=6)
        self.assertAlmostEqual(float(self.raw.y.min()), -0.25, places=6)
        self.assertAlmostEqual(float(self.raw.y.max()), 3.0, places=6)

    def test_pressure_is_absolute(self) -> None:
        self.assertGreater(float(self.raw.p.min()), 9.0e4)
        self.assertLess(float(self.raw.p.max()), 1.1e5)

    def test_velocity_magnitude_matches_components(self) -> None:
        self.assertIsNotNone(self.raw.u_mag)
        mismatch = float(np.max(np.abs(self.raw.u_mag - np.hypot(self.raw.u, self.raw.v))))
        self.assertLess(mismatch, 1e-6)

    def test_plane_is_flat(self) -> None:
        self.assertIsNotNone(self.raw.z)
        self.assertAlmostEqual(float(np.ptp(self.raw.z)), 0.0, places=9)

    def test_plain_csv_with_alias_header(self) -> None:
        text = "x,y,p,u,v\n0,0,100000,1.0,0.0\n1,0,99999,0.9,0.1\n"
        raw = parse_cfd_export(text)
        self.assertEqual(raw.n_points, 2)
        np.testing.assert_allclose(raw.u, [1.0, 0.9])

    def test_missing_required_column_raises(self) -> None:
        text = "x,y,p\n0,0,100000\n"
        with self.assertRaises(UnsetColumnError):
            parse_cfd_export(text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
