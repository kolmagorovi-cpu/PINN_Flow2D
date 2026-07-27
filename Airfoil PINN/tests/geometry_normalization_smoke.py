"""Smoke tests for one-source geometry nondimensionalization."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.data.initializer import Initializer
from airfoil_pinn.data.normalization import (
    build_geometry_reference,
    extract_geometry_reference,
)
from airfoil_pinn.data.sdf_diagnostics import estimate_boundary_layer_delta
from airfoil_pinn.geometry import Geometry2D, GeometryConfig
from airfoil_pinn.sampling.core import CollocationPoints, SamplerConfig
from airfoil_pinn.sampling.normal_layer import estimate_boundary_layer_thickness


class GeometryNormalizationTests(unittest.TestCase):
    """Verify current and legacy metadata resolve to one physical chord."""

    def test_geometry_reference_round_trip(self) -> None:
        foil = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]])
        reference = build_geometry_reference(foil, geom_name="test")

        self.assertIsNotNone(reference)
        resolved = extract_geometry_reference({"geometry_reference": reference})
        self.assertEqual(resolved["geom_name"], "test")
        self.assertAlmostEqual(resolved["chord"], 1.0)

    def test_initializer_prefers_matching_sampler_metadata(self) -> None:
        initializer = Initializer()
        initializer.points = np.zeros(3, dtype=[("x", "f8"), ("y", "f8")])
        initializer.wall_points = np.array([[0.1, 0.0], [0.9, 0.0]])
        initializer.sampler_metadata = {
            "num_points": 3,
            "geometry_reference": {"chord": 1.0, "geom_name": "sampled"},
        }

        self.assertAlmostEqual(initializer._load_chord_from_metadata(), 1.0)
        self.assertEqual(initializer.geometry_reference["geom_name"], "sampled")

    def test_initializer_loads_only_sampler_linked_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            domains_dir = os.path.join(directory, "domains")
            metadata_dir = os.path.join(directory, "metadata")
            os.makedirs(domains_dir)
            os.makedirs(metadata_dir)

            active_domain = np.array(
                [[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, -1.0]]
            )
            decoy_domain = np.array(
                [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, -10.0]]
            )
            active_foil = np.array([[0.0, 0.0], [1.0, 0.0]])
            np.save(os.path.join(domains_dir, "active_domain.npy"), active_domain)
            np.save(os.path.join(domains_dir, "newer_decoy_domain.npy"), decoy_domain)
            np.save(os.path.join(metadata_dir, "active_foil.npy"), active_foil)

            initializer = Initializer()
            initializer.geometry_dir = domains_dir
            initializer.sampler_metadata = {
                "geom_name": "active",
                "geometry_reference": {"geom_name": "active", "chord": 1.0},
            }
            with patch(
                "airfoil_pinn.data.initializer.GEOMETRY_METADATA_DIR",
                metadata_dir,
            ):
                np.testing.assert_allclose(
                    initializer._load_geometry_coords("domain"),
                    active_domain,
                )
                np.testing.assert_allclose(
                    initializer._load_geometry_coords("foil"),
                    active_foil,
                )

    def test_initializer_reconstructs_domain_from_sampler_bounds(self) -> None:
        initializer = Initializer()
        initializer.sampler_metadata = {
            "bounds": {
                "x_min": -1.5,
                "x_max": 6.0,
                "y_min": -1.0,
                "y_max": 0.5,
            }
        }
        initializer.load_domain_coords()

        expected = np.array(
            [
                [-1.5, -1.0],
                [6.0, -1.0],
                [6.0, 0.5],
                [-1.5, 0.5],
                [-1.5, -1.0],
            ]
        )
        np.testing.assert_allclose(initializer.domain_coords, expected)

    def test_sampler_requires_explicit_geometry_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SamplerConfig(
                geometry_dir=directory,
                output_dir=directory,
                geom_name=None,
            )
            sampler = CollocationPoints(config)
            with self.assertRaisesRegex(ValueError, "geom_name"):
                sampler.load_geometries()

    def test_geometry2d_loads_an_explicit_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            foil_path = os.path.join(directory, "foil.txt")
            with open(foil_path, "w", encoding="utf-8") as handle:
                handle.write("0.0 0.0\n1.0 0.0\n")

            geometry = Geometry2D(
                GeometryConfig(output_dir=directory, verbose=False)
            )
            self.assertTrue(geometry.load_from_file(foil_path))
            self.assertAlmostEqual(geometry.chord, 1.0)
            np.testing.assert_allclose(
                geometry.geometry_coords,
                [[0.0, 0.0], [1.0, 0.0]],
            )

    def test_delta_reference_uses_one_model(self) -> None:
        delta, delta_norm = estimate_boundary_layer_delta(
            1.5,
            625_000.0,
            model="turbulent",
        )
        sampler_delta = estimate_boundary_layer_thickness(
            1.5,
            625_000.0,
            cf_model="turbulent",
        )

        self.assertAlmostEqual(delta, sampler_delta)
        self.assertAlmostEqual(delta_norm, delta / 1.5)

    def test_data_handler_repairs_legacy_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initial_path = os.path.join(directory, "initial.npy")
            sampler_path = os.path.join(directory, "collocation_points.npy")
            self._write_legacy_initial(initial_path)
            self._write_sampler(sampler_path)

            with patch("airfoil_pinn.data.handler.SAMPLING_COLLOCATION_DIR", directory):
                handler = DataHandler(initial_path, batch_size=2, shuffle=False)

            np.testing.assert_allclose(handler.dataset.x.numpy().ravel(), [0.0, 1.0])
            np.testing.assert_allclose(handler.dataset.sdf.numpy().ravel(), [0.2, 0.4])
            self.assertAlmostEqual(handler.chord, 1.0)
            self.assertAlmostEqual(handler.Re, 10.0)
            self.assertAlmostEqual(handler.delta_bl_99_est_norm, 5.0 / np.sqrt(10.0))

    @staticmethod
    def _write_legacy_initial(path: str) -> None:
        """Write fields normalized by an intentionally stale two-metre chord."""
        dtype = [
            ("x_norm", "f8"), ("y_norm", "f8"), ("sdf_norm", "f8"),
            ("p_norm", "f8"), ("u_norm", "f8"), ("v_norm", "f8"),
            ("boundary_type", "U20"), ("on_boundary", "bool"),
        ]
        points = np.zeros(2, dtype=dtype)
        points["x_norm"] = [0.0, 0.5]
        points["sdf_norm"] = [0.1, 0.2]
        points["boundary_type"] = ["wall", "wall"]
        points["on_boundary"] = [True, True]
        np.save(path, points)
        np.save(path.replace(".npy", "_metadata.npy"), {
            "chord": 2.0,
            "sdf_norm_factor": 2.0,
            "U_inf": 1.0,
            "rho": 1.0,
            "mu": 0.1,
            "Re": 20.0,
        })

    @staticmethod
    def _write_sampler(path: str) -> None:
        """Write matching physical points and their authoritative geometry scale."""
        dtype = [
            ("x", "f8"), ("y", "f8"), ("on_boundary", "bool"),
            ("on_airfoil", "bool"), ("segment_id", "i4"),
        ]
        points = np.zeros(2, dtype=dtype)
        points["x"] = [0.0, 1.0]
        points["on_boundary"] = [True, True]
        points["on_airfoil"] = [True, True]
        points["segment_id"] = [-1, -1]
        np.save(path, points)
        np.save(path.replace(".npy", "_metadata.npy"), {
            "num_points": 2,
            "foil_closed": False,
            "geometry_reference": {"chord": 1.0, "geom_name": "sampled"},
        })


if __name__ == "__main__":
    unittest.main()
