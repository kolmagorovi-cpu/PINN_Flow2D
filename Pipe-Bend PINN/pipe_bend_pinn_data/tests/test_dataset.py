#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты torch-датасета нормализованных CFD-точек."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from ._paths import EXPORT_CSV, EXPORT_RECTANGLE, LAMINAR_SCALES

from pipe_bend_pinn_data import (
    ParametricCFDDataset,
    PolylineSDF,
    ReferenceScales,
    load_cfd_export,
    normalize_cfd,
)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_cfd_export(EXPORT_CSV)
        cls.scales = ReferenceScales(**LAMINAR_SCALES)
        polyline = np.asarray(EXPORT_RECTANGLE) / cls.scales.pipe_width
        points = np.column_stack(
            (cls.raw.x / cls.scales.pipe_width, cls.raw.y / cls.scales.pipe_width)
        )
        sdf = PolylineSDF(polyline).evaluate(points)
        cls.norm = normalize_cfd(
            cls.raw, cls.scales, U_in=1.0, case_name="rect", sdf_norm=sdf
        )

    def test_length(self) -> None:
        dataset = ParametricCFDDataset(self.norm)
        self.assertEqual(len(dataset), self.raw.n_points)

    def test_item_shapes(self) -> None:
        dataset = ParametricCFDDataset(self.norm)
        item = dataset[0]
        for key in ("x", "y", "sdf", "re_norm", "u", "v", "p"):
            self.assertEqual(tuple(item[key].shape), (1,), msg=key)
        self.assertEqual(tuple(item["case_id"].shape), ())

    def test_structured_array_fields(self) -> None:
        dataset = ParametricCFDDataset(self.norm)
        array = dataset.to_structured_array()
        expected = {
            "x_norm", "y_norm", "sdf_norm", "p_norm", "u_norm", "v_norm",
            "re", "re_norm", "U_in", "case_id", "boundary_type", "on_boundary",
        }
        self.assertEqual(set(array.dtype.names), expected)
        self.assertTrue(np.all(array["boundary_type"] == "data"))
        self.assertFalse(bool(array["on_boundary"].any()))

    def test_save_npy_roundtrip(self) -> None:
        dataset = ParametricCFDDataset(self.norm)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cfd_points.npy")
            dataset.save_npy(path)
            loaded = np.load(path)
        self.assertEqual(loaded.shape, (len(dataset),))
        np.testing.assert_allclose(loaded["x_norm"], self.norm.x_norm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
