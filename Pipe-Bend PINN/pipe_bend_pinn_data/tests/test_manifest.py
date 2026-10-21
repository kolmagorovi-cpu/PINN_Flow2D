#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты манифеста CFD-кейсов."""

from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from ._paths import DOMAIN_NPY, EXPORT_CSV, LAMINAR_SCALES

from pipe_bend_pinn_data import load_manifest


class ManifestTests(unittest.TestCase):
    def _write(self, directory: str, payload: dict) -> str:
        path = os.path.join(directory, "cfd_manifest.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True)
        return path

    def test_list_form(self) -> None:
        payload = {
            "ref_scales": dict(LAMINAR_SCALES),
            "geometry": {"domain_file": DOMAIN_NPY, "closed": True},
            "cases": [
                {"file": EXPORT_CSV, "U_in": 1.0, "name": "Re500"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = load_manifest(self._write(directory, payload))
        self.assertEqual(manifest.n_cases, 1)
        self.assertAlmostEqual(manifest.cases[0].U_in, 1.0)
        self.assertEqual(manifest.cases[0].name, "Re500")
        self.assertAlmostEqual(manifest.ref_scales.velocity, 1.0)
        self.assertAlmostEqual(manifest.ref_scales.reynolds, 500.0)
        self.assertTrue(manifest.geometry.is_defined)

    def test_mapping_form(self) -> None:
        payload = {
            "ref_scales": dict(LAMINAR_SCALES),
            "geometry": {"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            "cases": {EXPORT_CSV: 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = load_manifest(self._write(directory, payload))
        self.assertEqual(manifest.n_cases, 1)
        self.assertAlmostEqual(manifest.cases[0].U_in, 1.0)

    def test_missing_ref_scales_raises(self) -> None:
        payload = {"cases": {EXPORT_CSV: 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_duplicate_file_raises(self) -> None:
        payload = {
            "ref_scales": dict(LAMINAR_SCALES),
            "geometry": {"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            "cases": [
                {"file": EXPORT_CSV, "U_in": 1.0},
                {"file": EXPORT_CSV, "U_in": 2.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, payload)
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
