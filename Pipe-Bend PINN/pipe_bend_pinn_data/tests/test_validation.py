#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты валидации CFD-данных и привязки к геометрии."""

from __future__ import annotations

import unittest

import numpy as np

from ._paths import DOMAIN_NPY, EXPORT_CSV, EXPORT_RECTANGLE, LAMINAR_SCALES

from pipe_bend_pinn_data import (
    PolylineSDF,
    ReferenceScales,
    load_cfd_export,
    load_domain_polyline,
    normalize_cfd,
    validate_alignment,
    validate_normalized,
    validate_raw_cfd,
)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_cfd_export(EXPORT_CSV)
        cls.scales = ReferenceScales(**LAMINAR_SCALES)
        cls.points_norm = np.column_stack(
            (cls.raw.x / cls.scales.pipe_width, cls.raw.y / cls.scales.pipe_width)
        )

    def test_raw_cfd_is_clean(self) -> None:
        report = validate_raw_cfd(self.raw)
        self.assertTrue(report.ok, msg=report.summary())
        self.assertAlmostEqual(report.metrics["z_span"], 0.0, places=9)

    def test_normalized_is_clean(self) -> None:
        norm = normalize_cfd(self.raw, self.scales, U_in=1.0)
        report = validate_normalized(norm)
        self.assertTrue(report.ok, msg=report.summary())

    def test_alignment_with_rectangle_passes(self) -> None:
        polyline = np.asarray(EXPORT_RECTANGLE) / self.scales.pipe_width
        sdf = PolylineSDF(polyline).evaluate(self.points_norm)
        report = validate_alignment(self.points_norm, sdf, polyline)
        self.assertTrue(report.ok, msg=report.summary())
        self.assertAlmostEqual(report.metrics["coverage"], 1.0, places=9)

    def test_alignment_with_real_domain_flags_mismatch(self) -> None:
        vertices = load_domain_polyline(DOMAIN_NPY)
        polyline = vertices / self.scales.pipe_width
        sdf = PolylineSDF(polyline).evaluate(self.points_norm)
        report = validate_alignment(self.points_norm, sdf, polyline)
        # docs/export.csv экспортирован с другой геометрии — это должно ловиться.
        self.assertFalse(report.ok)
        self.assertIn("outside_domain", {issue.code for issue in report.errors})
        self.assertLess(report.metrics["coverage"], 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
