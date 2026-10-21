#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Интеграционные тесты CFDataHandler."""

from __future__ import annotations

import unittest

from ._paths import DOMAIN_NPY, EXPORT_CSV, EXPORT_RECTANGLE, LAMINAR_SCALES

from pipe_bend_pinn_data import (
    CFDCase,
    CFDataHandler,
    CFDManifest,
    GeometrySpec,
    ReferenceScales,
)


def _manifest(geometry: GeometrySpec) -> CFDManifest:
    return CFDManifest(
        cases=[CFDCase(file=EXPORT_CSV, U_in=1.0, name="Re500")],
        ref_scales=ReferenceScales(**LAMINAR_SCALES),
        geometry=geometry,
    )


class HandlerTests(unittest.TestCase):
    def test_load_with_rectangle_geometry(self) -> None:
        handler = CFDataHandler(
            _manifest(GeometrySpec(vertices=EXPORT_RECTANGLE)), strict=True
        ).load()
        self.assertEqual(len(handler), 3402)
        summary = handler.summary()
        self.assertEqual(summary["n_cases"], 1)
        self.assertEqual(summary["cases"][0]["name"], "Re500")
        self.assertAlmostEqual(summary["cases"][0]["Re"], 500.0)
        self.assertGreaterEqual(summary["cases"][0]["sdf_norm"][0], 0.0)

    def test_dataloader_batch(self) -> None:
        handler = CFDataHandler(
            _manifest(GeometrySpec(vertices=EXPORT_RECTANGLE)), strict=True
        ).load()
        loader = handler.dataloader(batch_size=128, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["x"].shape), (128, 1))
        self.assertEqual(tuple(batch["re_norm"].shape), (128, 1))
        self.assertEqual(tuple(batch["case_id"].shape), (128,))

    def test_mismatched_geometry_is_reported(self) -> None:
        handler = CFDataHandler(
            _manifest(GeometrySpec(domain_file=DOMAIN_NPY)), strict=False
        ).load()
        codes = {
            issue.code
            for report in handler.reports
            for issue in report.issues
        }
        self.assertIn("outside_domain", codes)

    def test_strict_mode_raises_on_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            CFDataHandler(
                _manifest(GeometrySpec(domain_file=DOMAIN_NPY)), strict=True
            ).load()


if __name__ == "__main__":
    unittest.main(verbosity=2)
