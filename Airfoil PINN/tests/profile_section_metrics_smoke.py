#!/usr/bin/env python3
"""Regression tests for exact FVM section profiles and flux diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from airfoil_pinn.data.handler import DataHandler, MeshData
from airfoil_pinn.training.components.metrics import MetricsCollector
from airfoil_pinn.training.telemetry import RunTelemetry


def _ring_mesh() -> MeshData:
    """Create a square fluid domain with a square solid hole."""
    coords = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-0.237, -0.237],
            [0.237, -0.237],
            [0.237, 0.237],
            [-0.237, 0.237],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    empty_i = np.empty(0, dtype=np.int32)
    empty_vec = np.empty((0, 2), dtype=np.float64)
    return MeshData(
        coords=coords,
        triangles=triangles,
        cv_volumes=np.ones(coords.shape[0], dtype=np.float64),
        int_face_node_i=empty_i,
        int_face_node_j=empty_i,
        int_face_area=empty_vec,
        bnd_face_node=empty_i,
        bnd_face_neighbor=empty_i,
        bnd_face_type=np.empty(0, dtype=str),
        bnd_face_area=empty_vec,
        node_type=np.full(coords.shape[0], "interior", dtype=object),
        node_bnd_type=np.full(coords.shape[0], "interior", dtype=object),
    )


def _handler() -> DataHandler:
    handler = DataHandler.__new__(DataHandler)
    handler.chord = 1.0
    handler.mesh_data = _ring_mesh()
    coords = handler.mesh_data.coords
    permutation = np.array([3, 5, 1, 7, 0, 6, 2, 4])
    shuffled = coords[permutation]
    handler.dataset = SimpleNamespace(
        n_points=len(shuffled),
        x=torch.tensor(shuffled[:, 0], dtype=torch.float64).view(-1, 1),
        y=torch.tensor(shuffled[:, 1], dtype=torch.float64).view(-1, 1),
        mesh_x_phys=torch.tensor(shuffled[:, 0], dtype=torch.float64),
        mesh_y_phys=torch.tensor(shuffled[:, 1], dtype=torch.float64),
        case_id=torch.zeros(len(shuffled), dtype=torch.long),
        u_target=torch.zeros(len(shuffled), 1, dtype=torch.float64),
    )
    return handler


class ExactProfileTests(unittest.TestCase):
    def test_barycentric_profile_uses_exact_x(self) -> None:
        handler = _handler()
        coords = handler.mesh_data.coords
        node_values = 2.0 * coords[:, 0] - 3.0 * coords[:, 1] + 4.0
        y_query = np.linspace(-0.9, 0.9, 37)

        y, sampled = handler.interpolate_fvm_profile(
            0.5,
            node_values,
            y_grid=y_query,
        )

        np.testing.assert_allclose(y, y_query)
        np.testing.assert_allclose(
            sampled,
            2.0 * 0.5 - 3.0 * y_query + 4.0,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_solid_hole_and_outside_domain_are_not_extrapolated(self) -> None:
        handler = _handler()
        node_values = np.ones(handler.mesh_data.n_nodes)
        y_query = np.array([-0.75, 0.0, 0.75])

        _, through_hole = handler.interpolate_fvm_profile(
            0.0,
            node_values,
            y_grid=y_query,
        )
        _, outside = handler.interpolate_fvm_profile(
            2.0,
            node_values,
            y_grid=y_query,
        )

        np.testing.assert_allclose(
            through_hole[[0, 2]],
            np.ones(2),
        )
        self.assertTrue(np.isnan(through_hole[1]))
        self.assertTrue(bool(np.isnan(outside).all()))

    def test_dataset_values_are_mapped_to_mesh_order(self) -> None:
        handler = _handler()
        dataset_coords = np.column_stack(
            (
                handler.dataset.mesh_x_phys.numpy(),
                handler.dataset.mesh_y_phys.numpy(),
            )
        )
        values = torch.tensor(
            1.5 * dataset_coords[:, 0] + dataset_coords[:, 1],
            dtype=torch.float64,
        )
        y_query = np.linspace(-0.8, 0.8, 9)

        y, sampled = handler.interpolate_profile(
            0.5,
            u_vals=values,
            y_grid=y_query,
        )

        np.testing.assert_allclose(y, y_query)
        np.testing.assert_allclose(
            sampled,
            1.5 * 0.5 + y_query,
            rtol=1e-13,
            atol=1e-13,
        )


class SectionFlowMetricTests(unittest.TestCase):
    def test_section_flux_does_not_bridge_solid_gap(self) -> None:
        collector = MetricsCollector.__new__(MetricsCollector)
        collector.data_handler = _handler()
        collector.section_flow_enabled = True
        collector.section_flow_sections = (-0.5, 0.0, 0.5)
        velocity = torch.zeros(
            collector.data_handler.mesh_data.n_nodes,
            2,
            dtype=torch.float64,
        )
        velocity[:, 0] = 1.0

        metrics = collector._compute_section_flow_metrics(
            velocity,
            reference_flux=2.0,
        )

        self.assertAlmostEqual(
            metrics["section_volume_flux"]["-0.5"],
            2.0,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["section_volume_flux"]["0"],
            1.526,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["section_volume_flux"]["0.5"],
            2.0,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["section_volume_flux_relative_spread"],
            0.237,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["section_volume_flux_max_relative_error"],
            0.237,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["section_volume_flux_relative_deviation"]["0"],
            -0.237,
            places=12,
        )

    def test_unavailable_section_and_text_formatter_are_explicit(self) -> None:
        collector = MetricsCollector.__new__(MetricsCollector)
        collector.data_handler = _handler()
        collector.section_flow_enabled = True
        collector.section_flow_sections = (0.5, 2.0)
        velocity = torch.zeros(
            collector.data_handler.mesh_data.n_nodes,
            2,
            dtype=torch.float64,
        )
        velocity[:, 0] = 1.0

        metrics = collector._compute_section_flow_metrics(
            velocity,
            reference_flux=2.0,
        )
        line = collector.format_section_flow_log(metrics)

        self.assertEqual(
            metrics["section_volume_flux_unavailable"],
            ["2"],
        )
        self.assertIsNotNone(line)
        self.assertIn("Q*(x/c=0.5)=2", line)
        self.assertIn("unavailable x/c=2", line)

    def test_section_metrics_survive_cache_and_strict_jsonl(self) -> None:
        collector = MetricsCollector.__new__(MetricsCollector)
        collector._last_mass_epoch = -1
        collector._cached_mass_metrics = {
            "section_volume_flux": {},
        }
        payload = {
            "mass_flux_in": 2.0,
            "mass_flux_out": 2.0,
            "mass_balance_error": 0.0,
            "section_volume_flux": {"-1": 2.0, "1.2": 1.8},
            "section_volume_flux_relative_spread": 0.1,
        }
        calls = []

        def compute():
            calls.append(1)
            return payload

        collector._compute_mass_flow_metrics = compute
        first = collector.get_mass_flow_metrics(2, interval=2)
        cached = collector.get_mass_flow_metrics(3, interval=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(cached, first)
        self.assertEqual(cached["section_volume_flux"]["1.2"], 1.8)

        with tempfile.TemporaryDirectory() as directory:
            telemetry = RunTelemetry(directory)
            telemetry.append_epoch({"diagnostics": cached})
            record = json.loads(
                Path(telemetry.epochs_path).read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(
            record["diagnostics"]["section_volume_flux"],
            payload["section_volume_flux"],
        )
        self.assertEqual(
            record["diagnostics"][
                "section_volume_flux_relative_spread"
            ],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
