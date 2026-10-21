#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты нормализации и параметризации по Re."""

from __future__ import annotations

import unittest

import numpy as np

from ._paths import LAMINAR_SCALES

from pipe_bend_pinn_data import (
    RawCFD,
    ReferenceScales,
    concat_normalized,
    normalize_cfd,
)


def _scales(**overrides) -> ReferenceScales:
    values = dict(LAMINAR_SCALES)
    values.update(overrides)
    return ReferenceScales(**values)


class NormalizationTests(unittest.TestCase):
    def test_reference_reynolds_is_500(self) -> None:
        self.assertAlmostEqual(_scales().reynolds, 500.0, places=9)

    def test_reynolds_scales_with_velocity(self) -> None:
        scales = _scales()
        self.assertAlmostEqual(scales.reynolds_for(2.0), 1000.0, places=9)

    def test_invalid_scale_raises(self) -> None:
        with self.assertRaises(ValueError):
            _scales(pipe_width=-1.0)

    def test_normalization_fields(self) -> None:
        scales = _scales()
        raw = RawCFD(
            x=np.array([-2.0, 0.0]),
            y=np.array([0.5, 1.0]),
            p=np.array([100000.0, 100500.0]),
            u=np.array([1.0, 0.0]),
            v=np.array([0.0, 0.5]),
        )
        norm = normalize_cfd(raw, scales, U_in=1.0, case_name="unit")

        np.testing.assert_allclose(norm.x_norm, [-4.0, 0.0])
        np.testing.assert_allclose(norm.y_norm, [1.0, 2.0])
        np.testing.assert_allclose(norm.u_norm, [1.0, 0.0])
        np.testing.assert_allclose(norm.v_norm, [0.0, 0.5])
        # q = 0.5 * 998 * 1^2 = 499 Па; p_ref = 100000 Па.
        np.testing.assert_allclose(norm.p_norm, [0.0, 500.0 / 499.0])
        self.assertAlmostEqual(float(norm.re[0]), 500.0, places=9)
        self.assertAlmostEqual(float(norm.re_norm[0]), 1.0, places=12)

    def test_re_norm_tracks_velocity_ratio(self) -> None:
        scales = _scales()
        raw = RawCFD(
            x=np.zeros(3), y=np.zeros(3), p=np.zeros(3),
            u=np.ones(3), v=np.zeros(3),
        )
        norm = normalize_cfd(raw, scales, U_in=1.5)
        self.assertAlmostEqual(float(norm.re_norm[0]), 1.5, places=12)
        self.assertAlmostEqual(float(norm.re[0]), 750.0, places=9)

    def test_concat_renumbers_cases(self) -> None:
        scales = _scales()
        raw = RawCFD(x=np.zeros(2), y=np.zeros(2), p=np.zeros(2),
                     u=np.ones(2), v=np.zeros(2))
        first = normalize_cfd(raw, scales, 1.0, case_name="a", case_id=0)
        second = normalize_cfd(raw, scales, 2.0, case_name="b", case_id=0)
        merged = concat_normalized([first, second])
        self.assertEqual(merged.n_cases, 2)
        self.assertEqual(merged.case_id.tolist(), [0, 0, 1, 1])
        self.assertEqual(merged.case_names, ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
