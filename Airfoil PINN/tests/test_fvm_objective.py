"""Analytical and contract tests for the finite-volume continuity objective."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.data.flow import FlowCase
from airfoil_pinn.data.normalization import ReferenceScales
from airfoil_pinn.physics.fvm import FVMResiduals
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.contract import (
    build_training_contract,
    validate_checkpoint_contract,
)
from airfoil_pinn.training.input_features import InputFeatureAssembler
from airfoil_pinn.training.solver import PINNSolver


def _structured_cell_mesh(n: int) -> tuple[SimpleNamespace, np.ndarray]:
    """Build an n-by-n conservative cell-centred mesh on the unit square."""
    h = 1.0 / n
    coordinates = np.array(
        [
            [(i + 0.5) * h, (j + 0.5) * h]
            for j in range(n)
            for i in range(n)
        ],
        dtype=np.float64,
    )

    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_s: list[list[float]] = []
    bnd_node: list[int] = []
    bnd_s: list[list[float]] = []
    int_quad_coords: list[list[float]] = []
    bnd_quad_coords: list[list[float]] = []

    def node(i: int, j: int) -> int:
        return j * n + i

    for j in range(n):
        for i in range(n):
            current = node(i, j)
            if i + 1 < n:
                edge_i.append(current)
                edge_j.append(node(i + 1, j))
                edge_s.append([h, 0.0])
                int_quad_coords.append([(i + 1.0) * h, (j + 0.5) * h])
            if j + 1 < n:
                edge_i.append(current)
                edge_j.append(node(i, j + 1))
                edge_s.append([0.0, h])
                int_quad_coords.append([(i + 0.5) * h, (j + 1.0) * h])
            if i == 0:
                bnd_node.append(current)
                bnd_s.append([-h, 0.0])
                bnd_quad_coords.append([0.0, (j + 0.5) * h])
            if i == n - 1:
                bnd_node.append(current)
                bnd_s.append([h, 0.0])
                bnd_quad_coords.append([1.0, (j + 0.5) * h])
            if j == 0:
                bnd_node.append(current)
                bnd_s.append([0.0, -h])
                bnd_quad_coords.append([(i + 0.5) * h, 0.0])
            if j == n - 1:
                bnd_node.append(current)
                bnd_s.append([0.0, h])
                bnd_quad_coords.append([(i + 0.5) * h, 1.0])

    mesh = SimpleNamespace(
        cv_volumes=np.full(n * n, h * h, dtype=np.float64),
        int_face_node_i=np.asarray(edge_i, dtype=np.int64),
        int_face_node_j=np.asarray(edge_j, dtype=np.int64),
        int_face_area=np.asarray(edge_s, dtype=np.float64).reshape(-1, 2),
        bnd_face_node=np.asarray(bnd_node, dtype=np.int64),
        bnd_face_area=np.asarray(bnd_s, dtype=np.float64).reshape(-1, 2),
        bnd_face_type=np.full(len(bnd_node), "outlet"),
        int_quad_node_i=np.asarray(edge_i, dtype=np.int64),
        int_quad_node_j=np.asarray(edge_j, dtype=np.int64),
        int_quad_area=np.asarray(edge_s, dtype=np.float64).reshape(-1, 2),
        int_quad_coords=np.asarray(int_quad_coords, dtype=np.float64),
        bnd_quad_coords=np.asarray(bnd_quad_coords, dtype=np.float64),
    )
    return mesh, coordinates


def _one_cell_mesh() -> SimpleNamespace:
    return SimpleNamespace(
        cv_volumes=np.array([1.0], dtype=np.float64),
        int_face_node_i=np.empty(0, dtype=np.int64),
        int_face_node_j=np.empty(0, dtype=np.int64),
        int_face_area=np.empty((0, 2), dtype=np.float64),
        bnd_face_node=np.array([0, 0, 0, 0], dtype=np.int64),
        bnd_face_area=np.array(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        bnd_face_type=np.full(4, "outlet"),
    )


class FVMObjectiveTests(unittest.TestCase):
    """Verify analytical scaling, differentiability and strict validation."""

    def test_local_flux_imbalance_is_divided_by_control_volume(self) -> None:
        """The local residual is divergence ``Phi_i / V_i``, not raw flux."""

        residuals = []
        for volume in (1.0, 0.01):
            mesh = _one_cell_mesh()
            mesh.cv_volumes[0] = volume
            mesh.bnd_face_type[0] = "inlet"
            fvm = FVMResiduals(mesh)
            fvm.set_inlet_velocity(
                torch.tensor([1.0, 0.0], dtype=torch.float64)
            )
            residual = fvm.compute_residual(
                torch.zeros((1, 2), dtype=torch.float64)
            )
            residuals.append(residual)
            torch.testing.assert_close(
                residual,
                torch.tensor([-1.0 / volume], dtype=torch.float64),
            )

        torch.testing.assert_close(residuals[1], 100.0 * residuals[0])

    def test_constant_field_is_discretely_divergence_free(self) -> None:
        mesh, _ = _structured_cell_mesh(8)
        fvm = FVMResiduals(mesh)
        velocity = torch.tensor(
            [[2.5, -0.75]] * fvm.N,
            dtype=torch.float64,
        )
        residual = fvm.compute_residual(velocity)
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=2e-14,
            rtol=0.0,
        )
        self.assertLess(
            fvm.mass_loss(
                velocity,
                reduction="divergence_l2_volume",
            ).item(),
            1e-26,
        )

    def test_known_divergence_converges_to_four_without_h2_collapse(self) -> None:
        physical_losses = []
        legacy_losses = []
        for n in (8, 16, 32):
            mesh, coordinates = _structured_cell_mesh(n)
            fvm = FVMResiduals(mesh)
            velocity = torch.as_tensor(coordinates, dtype=torch.float64)
            physical_losses.append(
                fvm.mass_loss(
                    velocity,
                    reduction="divergence_l2_volume",
                ).item()
            )
            legacy_losses.append(
                fvm.mass_loss(
                    velocity,
                    reduction="volume_weighted",
                ).item()
            )

        self.assertTrue(
            all(
                abs(4.0 - fine) < abs(4.0 - coarse)
                for coarse, fine in zip(
                    physical_losses,
                    physical_losses[1:],
                )
            )
        )
        self.assertGreater(physical_losses[-1], 3.7)
        self.assertLess(legacy_losses[-1], legacy_losses[0] / 10.0)

    def test_fvm_loss_preserves_nonzero_finite_autograd(self) -> None:
        mesh, coordinates = _structured_cell_mesh(8)
        fvm = FVMResiduals(mesh)
        scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
        velocity = scale * torch.as_tensor(
            coordinates,
            dtype=torch.float64,
        )
        loss = fvm.mass_loss(
            velocity,
            reduction="divergence_l2_volume",
        )
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))
        self.assertGreater(abs(scale.grad.item()), 0.0)

    def test_direct_face_predictions_reproduce_linear_divergence(self) -> None:
        """Segment midpoint PINN values avoid all nodal interpolation."""
        mesh, _ = _structured_cell_mesh(8)
        fvm = FVMResiduals(mesh)
        internal_velocity = torch.as_tensor(
            mesh.int_quad_coords,
            dtype=torch.float64,
        )
        boundary_velocity = torch.as_tensor(
            mesh.bnd_quad_coords,
            dtype=torch.float64,
        )
        residual = fvm.compute_residual_quadrature(
            internal_velocity,
            boundary_velocity,
        )
        torch.testing.assert_close(
            residual,
            torch.full_like(residual, 2.0),
            atol=2e-14,
            rtol=0.0,
        )

    def test_direct_face_fvm_preserves_parameter_gradients(self) -> None:
        mesh, _ = _structured_cell_mesh(4)
        fvm = FVMResiduals(mesh)
        scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
        internal_velocity = scale * torch.as_tensor(
            mesh.int_quad_coords,
            dtype=torch.float64,
        )
        boundary_velocity = scale * torch.as_tensor(
            mesh.bnd_quad_coords,
            dtype=torch.float64,
        )
        loss = fvm.mass_loss_quadrature(
            internal_velocity,
            boundary_velocity,
        )
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))
        self.assertGreater(abs(scale.grad.item()), 0.0)

    def test_integral_balance_uses_boundary_midpoint_predictions(self) -> None:
        mesh, _ = _structured_cell_mesh(4)
        boundary_xy = mesh.bnd_quad_coords
        mesh.bnd_face_type = np.where(
            np.isclose(boundary_xy[:, 0], 0.0),
            "inlet",
            np.where(
                np.isclose(boundary_xy[:, 0], 1.0),
                "outlet",
                "farfield",
            ),
        )
        fvm = FVMResiduals(mesh)
        loss_computer = LossComputer.__new__(LossComputer)
        loss_computer.integral_condition = True
        loss_computer.integral_target = 0.0
        loss_computer.flow_type = "2d"
        loss_computer.device = torch.device("cpu")
        loss_computer.dtype = torch.float64
        velocity = torch.zeros((fvm.B, 2), dtype=torch.float64)
        velocity[:, 0] = 1.0
        loss = loss_computer.compute_boundary_quadrature_integral_loss(
            velocity,
            fvm.bnd_S,
            fvm.bnd_type,
        )
        torch.testing.assert_close(
            loss,
            torch.zeros_like(loss),
            atol=2e-14,
            rtol=0.0,
        )

        # The top farfield has unit length. A unit outward velocity there
        # creates total signed flux 1 and therefore a unit squared loss.
        velocity.zero_()
        top_farfield = torch.as_tensor(
            np.isclose(boundary_xy[:, 1], 1.0),
            dtype=torch.bool,
        )
        velocity[top_farfield, 1] = 1.0
        loss = loss_computer.compute_boundary_quadrature_integral_loss(
            velocity,
            fvm.bnd_S,
            fvm.bnd_type,
        )
        torch.testing.assert_close(
            loss,
            torch.ones_like(loss),
            atol=2e-14,
            rtol=0.0,
        )

    def test_raw_weighted_and_rms_metrics_are_separate(self) -> None:
        mesh, coordinates = _structured_cell_mesh(8)
        loss_computer = LossComputer.__new__(LossComputer)
        loss_computer.device = torch.device("cpu")
        loss_computer.dtype = torch.float64
        loss_computer.fvm = FVMResiduals(mesh)
        loss_computer.fvm_weight = 25.0
        loss_computer.fvm_loss_reduction = "divergence_l2_volume"
        metrics = loss_computer.compute_fvm_metrics(
            torch.as_tensor(coordinates, dtype=torch.float64)
        )
        torch.testing.assert_close(
            metrics["weighted"],
            25.0 * metrics["raw"],
        )
        torch.testing.assert_close(metrics["mse"], metrics["raw"])
        torch.testing.assert_close(
            metrics["rms"].square(),
            metrics["mse"],
        )

    def test_invalid_volumes_are_rejected_instead_of_clamped(self) -> None:
        for bad_value in (float("nan"), 0.0, -1.0):
            with self.subTest(value=bad_value):
                mesh = _one_cell_mesh()
                mesh.cv_volumes[0] = bad_value
                with self.assertRaisesRegex(
                    ValueError,
                    "NaN or Inf|strictly positive",
                ):
                    FVMResiduals(mesh)

    def test_rebuild_revalidates_geometry(self) -> None:
        fvm = FVMResiduals(_one_cell_mesh())
        invalid = _one_cell_mesh()
        invalid.cv_volumes[0] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            fvm.rebuild_from_mesh(invalid)
        self.assertEqual(fvm.N, 1)
        self.assertGreater(fvm.V.item(), 0.0)

    def test_invalid_faces_closure_and_boundary_types_are_rejected(self) -> None:
        bad_index = _one_cell_mesh()
        bad_index.bnd_face_node[0] = 1
        with self.assertRaisesRegex(ValueError, "indices"):
            FVMResiduals(bad_index)

        bad_area = _one_cell_mesh()
        bad_area.bnd_face_area[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "area vectors"):
            FVMResiduals(bad_area)

        bad_closure = _one_cell_mesh()
        bad_closure.bnd_face_area[0, 0] = -0.9
        with self.assertRaisesRegex(ValueError, "do not close"):
            FVMResiduals(bad_closure)

        bad_type = _one_cell_mesh()
        bad_type.bnd_face_type[0] = "mystery"
        with self.assertRaisesRegex(ValueError, "unsupported types"):
            FVMResiduals(bad_type)

    def test_velocity_node_count_and_mesh_fingerprint_are_strict(self) -> None:
        fvm = FVMResiduals(_one_cell_mesh())
        with self.assertRaisesRegex(ValueError, r"\(1, 2\)"):
            fvm.compute_residual(torch.zeros((2, 2), dtype=torch.float64))

        changed = _one_cell_mesh()
        changed.cv_volumes[0] = 2.0
        self.assertNotEqual(
            fvm.summary()["mesh_fingerprint"],
            FVMResiduals(changed).summary()["mesh_fingerprint"],
        )


class FVMScheduleTests(unittest.TestCase):
    """Check optimizer-step scheduling and inverse-probability correction."""

    @staticmethod
    def _trainer(
        schedule: str,
        interval: int,
        accumulation_steps: int = 1,
    ) -> Trainer:
        trainer = Trainer.__new__(Trainer)
        trainer.fvm_schedule = schedule
        trainer.fvm_batch_interval = interval
        trainer.accum_steps = accumulation_steps
        trainer._fvm_optimizer_step_counter = 0
        return trainer

    def test_full_schedule_runs_once_per_optimizer_step(self) -> None:
        trainer = self._trainer(
            "every_optimizer_step",
            interval=1,
            accumulation_steps=4,
        )
        self.assertEqual(
            trainer._scheduled_fvm_batches(10, True),
            {3: 1.0, 7: 1.0, 9: 1.0},
        )

    def test_stochastic_gradient_matches_full_schedule_in_expectation(self) -> None:
        random.seed(937)
        sample_count = 50_000
        trainer = self._trainer(
            "stochastic_optimizer_step",
            interval=5,
        )
        corrections = trainer._scheduled_fvm_batches(
            sample_count,
            True,
        )
        theta = 1.75
        full_gradient = 2.0 * theta
        sampled_gradient = (
            2.0 * theta * sum(corrections.values()) / sample_count
        )
        self.assertTrue(
            math.isclose(
                sampled_gradient,
                full_gradient,
                rel_tol=0.025,
            )
        )


class FVMConfigurationContractTests(unittest.TestCase):
    """Ensure active configs and checkpoints expose the complete FVM contract."""

    def test_active_fvm_configs_use_the_physical_baseline(self) -> None:
        config_dir = Path(__file__).parents[1] / "workspace" / "configs"
        paths = sorted(config_dir.glob("FVM*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                fvm = config["loss_weights"]["fvm"]
                self.assertTrue(fvm["enabled"])
                self.assertEqual(fvm["weight"], 25.0)
                self.assertEqual(fvm["reduction"], "divergence_l2_volume")
                self.assertEqual(fvm["schedule"], "every_optimizer_step")
                self.assertEqual(fvm["batch_interval"], 1)

    def test_enabled_fvm_without_mesh_fails_fast(self) -> None:
        solver = PINNSolver.__new__(PINNSolver)
        solver.config = {
            "loss_weights": {
                "fvm": {"enabled": True, "weight": 25.0},
                "integral_condition": False,
            }
        }
        solver.loss_computer = SimpleNamespace(
            flow_type="2d",
            fvm=None,
        )
        solver.data_handler = SimpleNamespace(mesh_data=None)
        with self.assertRaisesRegex(RuntimeError, "no valid mesh"):
            solver._setup_fvm_residuals()

    def test_checkpoint_fingerprint_changes_with_fvm_contract(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["loss_weights"]["fvm"] = {
            "enabled": True,
            "weight": 25.0,
            "reduction": "divergence_l2_volume",
            "schedule": "every_optimizer_step",
            "batch_interval": 1,
        }
        assembler = InputFeatureAssembler(config)
        handler = SimpleNamespace(
            reference_scales=ReferenceScales(
                chord=1.0,
                velocity=1.0,
                density=1.0,
                dynamic_viscosity=1e-3,
                static_pressure=101_325.0,
            ),
            flow_cases={0: FlowCase(0, reynolds=1000.0)},
            boundary_type_names=["interior"],
            fvm_sdf_geometry_fingerprint="sdf-mesh",
            cached_sdf_geometry_fingerprint="sdf-mesh",
            cached_sdf_method="exact_polyline_segments",
        )
        operator = SimpleNamespace(
            mesh_fingerprint="mesh-a",
            _geometry_is_dimensionless=True,
            _geometry_length_scale=1.0,
        )
        loss = SimpleNamespace(
            feature_manifest=assembler.manifest,
            sdf_geometry_fingerprint="sdf-mesh",
            sdf_float64_consistency_tolerance=1e-12,
            fvm=operator,
        )
        current = build_training_contract(config, handler, loss)
        self.assertEqual(
            current.payload["fvm"],
            {
                "enabled": True,
                "velocity_sampling": "pinn_segment_midpoint",
                "quadrature_rule": "midpoint_per_dual_segment",
                "reduction": "divergence_l2_volume",
                "weight": 25.0,
                "schedule": "every_optimizer_step",
                "interval": 1,
                "inclusion_probability": 1.0,
                "mesh_identifier": config["data"].get("mesh_file"),
                "mesh_fingerprint": "mesh-a",
                "internal_quadrature_points": None,
                "boundary_quadrature_points": None,
                "normalization": "face_vectors/c;control_volumes/c^2",
                "geometry_is_dimensionless": True,
                "geometry_length_scale": 1.0,
            },
        )

        changed_config = copy.deepcopy(config)
        changed_config["loss_weights"]["fvm"]["reduction"] = "volume_weighted"
        changed = build_training_contract(changed_config, handler, loss)
        self.assertNotEqual(current.fingerprint(), changed.fingerprint())
        with self.assertRaisesRegex(RuntimeError, "fvm"):
            validate_checkpoint_contract(current.to_metadata(), changed)

        operator.mesh_fingerprint = "mesh-b"
        changed_mesh = build_training_contract(config, handler, loss)
        self.assertNotEqual(current.fingerprint(), changed_mesh.fingerprint())
        with self.assertRaisesRegex(RuntimeError, "fvm"):
            validate_checkpoint_contract(
                current.to_metadata(),
                changed_mesh,
            )


if __name__ == "__main__":
    unittest.main()
