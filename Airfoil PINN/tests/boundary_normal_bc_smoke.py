"""Rotation-invariant boundary-condition tests using geometric normals."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from airfoil_pinn.data.flow import FlowCase
from airfoil_pinn.data.handler import DataHandler, PINNDataset
from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.sampling.core import CollocationPoints
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.components.trainer import Trainer


DTYPE = torch.float64


def _loss_computer(boundary_name: str) -> LossComputer:
    config = {
        "input_params": {
            "coordinates": True,
            "sdf": True,
            "viscosity": False,
            "inlet_velocity": False,
        },
        "pde": {
            "flow_type": "2d",
            "system": "incompressible_navier_stokes",
        },
        "boundary_conditions": {
            boundary_name: {
                "kind": boundary_name,
                "enforcement": "soft",
            },
        },
        "loss_weights": {
            "pde_weights": {
                "cont": 1.0,
                "mom_x": 1.0,
                "mom_y": 1.0,
            },
            "bc_weights": {
                boundary_name: 1.0,
                "symmetry_grad": 1.0,
            },
            "use_sa_weights": False,
            "sa_apply_to_bc": False,
            "adaptive_bc": False,
            "integral_condition": False,
            "integral_weight": 0.0,
            "fvm": {"enabled": False, "weight": 0.0},
        },
        "diagnostics": {"strict_autograd": True},
    }
    handler = SimpleNamespace(
        boundary_type_names=[boundary_name],
        mesh_airfoil_coords=None,
        chord=1.0,
        flow_cases={
            0: FlowCase(
                case_id=0,
                reynolds=1.0e5,
                angle_degrees=0.0,
            )
        },
        dataset=SimpleNamespace(type_to_idx={boundary_name: 0}),
    )
    handler.get_flow_case_batch = MethodType(
        DataHandler.get_flow_case_batch,
        handler,
    )
    return LossComputer(
        config=config,
        data_handler=handler,
        rho_norm=torch.tensor(1.0, dtype=DTYPE),
        mu_norm=torch.tensor(1.0e-5, dtype=DTYPE),
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )


def _rotation(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=DTYPE,
    )


class PressureBoundaryTests(unittest.TestCase):
    def test_pressure_penalty_and_gradients_are_boundary_local(self) -> None:
        """Opposite boundary pressures cannot cancel or affect interior Cp."""
        for name in (
            "outlet", "farfield", "inlet", "velocity_dirichlet", "wall", "interior",
        ):
            with self.subTest(boundary=name):
                loss = _loss_computer(name)
                u = torch.ones((3, 1), dtype=DTYPE)
                zero = torch.zeros_like(u)
                pressure = torch.tensor(
                    [[2.0], [-2.0], [100.0]], dtype=DTYPE, requires_grad=True,
                )
                # The third point is outside the boundary, regardless of Cp.
                boundary = torch.tensor([[1.0], [1.0], [0.0]], dtype=DTYPE)
                baseline, _ = loss.compute_bc_loss(u, zero, zero, boundary, zero, zero)
                actual, _ = loss.compute_bc_loss(u, zero, pressure, boundary, zero, zero)
                active = name in ("outlet", "farfield")
                delta = actual[name] - baseline[name]
                self.assertEqual(float(delta.detach()), 4.0 if active else 0.0)
                if active:
                    gradient, = torch.autograd.grad(delta, pressure)
                    torch.testing.assert_close(
                        gradient, torch.tensor([[2.0], [-2.0], [0.0]], dtype=DTYPE),
                    )

    def test_adaptive_weights_and_rar_use_the_same_pressure_boundaries(self) -> None:
        for name in ("outlet", "farfield", "inlet", "velocity_dirichlet", "interior"):
            with self.subTest(boundary=name):
                loss = _loss_computer(name)
                loss.loss_weights[name] = 3.0
                trainer = Trainer.__new__(Trainer)
                trainer.loss_computer = loss
                trainer.data_handler = loss.data_handler
                trainer.device = torch.device("cpu")
                trainer.dtype = DTYPE
                trainer.sa_weights = None
                u = torch.ones((3, 1), dtype=DTYPE)
                zero = torch.zeros_like(u)
                pressure = torch.tensor([[2.0], [-2.0], [100.0]], dtype=DTYPE)
                boundary = torch.tensor([[1.0], [1.0], [0.0]], dtype=DTYPE)
                weights = torch.tensor([[2.0], [4.0], [8.0]], dtype=DTYPE)
                args = dict(u=u, v=zero, x=zero, y=zero, boundary_types=boundary)
                baseline = trainer._compute_sa_bc_loss(p=zero, w_bc=weights, **args)
                actual = trainer._compute_sa_bc_loss(p=pressure, w_bc=weights, **args)
                active = name in ("outlet", "farfield")
                self.assertEqual(float(actual - baseline), 36.0 if active else 0.0)

                for adaptive in (False, True):
                    trainer.sa_weights = object() if adaptive else None
                    loss.sa_apply_to_bc = adaptive
                    base_score = trainer._compute_bc_point_score(p=zero, w_bc=weights, **args)
                    score = trainer._compute_bc_point_score(p=pressure, w_bc=weights, **args)
                    expected = torch.zeros(3, dtype=DTYPE)
                    if active:
                        expected[:2] = torch.tensor(
                            [24.0, 48.0] if adaptive else [12.0, 12.0], dtype=DTYPE,
                        )
                    torch.testing.assert_close(score - base_score, expected)


class SlipWallNormalTests(unittest.TestCase):
    def test_horizontal_vertical_and_inclined_boundaries(self) -> None:
        loss = _loss_computer("slip_wall")
        for normal in ((0.0, 1.0), (1.0, 0.0), (0.6, 0.8)):
            nx = torch.full((3, 1), normal[0], dtype=DTYPE)
            ny = torch.full((3, 1), normal[1], dtype=DTYPE)
            tangent = torch.tensor(
                [[-normal[1], normal[0]]],
                dtype=DTYPE,
            )
            velocity = tangent.repeat(3, 1) * torch.tensor(
                [[1.0], [2.0], [-3.0]],
                dtype=DTYPE,
            )
            boundary = torch.ones((3, 1), dtype=DTYPE)
            components, extra = loss.compute_bc_loss(
                velocity[:, 0:1],
                velocity[:, 1:2],
                torch.zeros((3, 1), dtype=DTYPE),
                boundary,
                torch.zeros((3, 1), dtype=DTYPE),
                torch.zeros((3, 1), dtype=DTYPE),
                normal_x=nx,
                normal_y=ny,
            )
            torch.testing.assert_close(
                components["slip_wall"],
                torch.zeros((), dtype=DTYPE),
                atol=1.0e-15,
                rtol=0.0,
            )
            self.assertEqual(float(extra), 0.0)

    def test_curved_boundary_uses_pointwise_normals(self) -> None:
        loss = _loss_computer("slip_wall")
        theta = torch.linspace(0.0, 1.5 * math.pi, 7, dtype=DTYPE)
        nx = torch.cos(theta).view(-1, 1)
        ny = torch.sin(theta).view(-1, 1)
        u = -ny
        v = nx
        boundary = torch.ones_like(nx)

        components, _ = loss.compute_bc_loss(
            u,
            v,
            torch.zeros_like(u),
            boundary,
            nx.clone(),
            ny.clone(),
            normal_x=nx,
            normal_y=ny,
        )
        torch.testing.assert_close(
            components["slip_wall"],
            torch.zeros((), dtype=DTYPE),
            atol=1.0e-15,
            rtol=0.0,
        )

    def test_missing_normals_fail_instead_of_guessing_a_side(self) -> None:
        loss = _loss_computer("slip_wall")
        with self.assertRaisesRegex(RuntimeError, "requires stored normal_x"):
            loss.compute_bc_loss(
                torch.ones((1, 1), dtype=DTYPE),
                torch.zeros((1, 1), dtype=DTYPE),
                torch.zeros((1, 1), dtype=DTYPE),
                torch.ones((1, 1), dtype=DTYPE),
                torch.zeros((1, 1), dtype=DTYPE),
                torch.zeros((1, 1), dtype=DTYPE),
            )


class SymmetryNormalTests(unittest.TestCase):
    @staticmethod
    def _linear_velocity(
        coordinates: torch.Tensor,
        offset: torch.Tensor,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = coordinates @ gradient.T + offset
        return velocity[:, 0:1], velocity[:, 1:2]

    def _evaluate(
        self,
        coordinates: torch.Tensor,
        normals: torch.Tensor,
        offset: torch.Tensor,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = _loss_computer("symmetry")
        x = coordinates[:, 0:1].clone().detach().requires_grad_(True)
        y = coordinates[:, 1:2].clone().detach().requires_grad_(True)
        xy = torch.cat((x, y), dim=1)
        u, v = self._linear_velocity(xy, offset, gradient)
        components, extra = loss.compute_bc_loss(
            u,
            v,
            torch.zeros_like(u),
            torch.ones_like(u),
            x,
            y,
            normal_x=normals[:, 0:1],
            normal_y=normals[:, 1:2],
        )
        return components["symmetry"], extra

    def test_symmetry_enforces_normal_velocity_and_tangent_normal_gradient(
        self,
    ) -> None:
        normal = torch.tensor([0.6, 0.8], dtype=DTYPE)
        tangent = torch.tensor([-0.8, 0.6], dtype=DTYPE)
        coordinates = torch.stack(
            (tangent * -1.0, tangent * 0.5, tangent * 2.0),
        )
        normals = normal.repeat(3, 1)

        normal_loss, gradient_loss = self._evaluate(
            coordinates,
            normals,
            tangent,
            torch.zeros((2, 2), dtype=DTYPE),
        )
        torch.testing.assert_close(
            normal_loss,
            torch.zeros((), dtype=DTYPE),
            atol=1.0e-15,
            rtol=0.0,
        )
        torch.testing.assert_close(
            gradient_loss,
            torch.zeros((), dtype=DTYPE),
            atol=1.0e-15,
            rtol=0.0,
        )

        # q(r) = t * (n dot r) has du_t/dn = 1 and u_n = 0.
        gradient = torch.outer(tangent, normal)
        normal_loss, gradient_loss = self._evaluate(
            coordinates,
            normals,
            torch.zeros(2, dtype=DTYPE),
            gradient,
        )
        torch.testing.assert_close(
            normal_loss,
            torch.zeros((), dtype=DTYPE),
            atol=1.0e-15,
            rtol=0.0,
        )
        torch.testing.assert_close(
            gradient_loss,
            torch.ones((), dtype=DTYPE),
            atol=1.0e-14,
            rtol=0.0,
        )

    def test_boundary_loss_is_invariant_under_rotation(self) -> None:
        coordinates = torch.tensor(
            [[-0.8, 0.3], [0.2, -0.5], [1.1, 0.7]],
            dtype=DTYPE,
        )
        normal = torch.tensor([0.6, 0.8], dtype=DTYPE)
        normals = normal.repeat(3, 1)
        offset = torch.tensor([0.4, -0.7], dtype=DTYPE)
        gradient = torch.tensor(
            [[0.3, -0.2], [0.5, 0.9]],
            dtype=DTYPE,
        )
        original = self._evaluate(
            coordinates,
            normals,
            offset,
            gradient,
        )

        rotation = _rotation(0.73)
        rotated = self._evaluate(
            coordinates @ rotation.T,
            normals @ rotation.T,
            offset @ rotation.T,
            rotation @ gradient @ rotation.T,
        )
        torch.testing.assert_close(original[0], rotated[0])
        torch.testing.assert_close(original[1], rotated[1])


class BoundaryNormalDataTests(unittest.TestCase):
    def test_default_config_separates_boundary_physics(self) -> None:
        config = ConfigManager().config
        boundary = config["boundary_conditions"]
        self.assertEqual(boundary["slip_wall"]["kind"], "slip_wall")
        self.assertEqual(boundary["symmetry"]["kind"], "symmetry")
        self.assertEqual(boundary["farfield"]["kind"], "freestream")
        self.assertNotIn("slip_tangent_grad", config["loss_weights"]["bc_weights"])
        self.assertNotIn("slip_velocity", config["loss_weights"]["bc_weights"])

    def test_sampler_computes_unit_normals_for_curved_polyline(self) -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 65)
        reference = np.column_stack((np.cos(theta), np.sin(theta)))
        points = 0.5 * (reference[:-1] + reference[1:])
        normals = CollocationPoints._polyline_normals(
            points,
            reference,
            closed=True,
        )
        tangents = reference[1:] - reference[:-1]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        np.testing.assert_allclose(
            np.linalg.norm(normals, axis=1),
            1.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            np.sum(normals * tangents, axis=1),
            0.0,
            atol=1.0e-14,
        )

    def test_dataset_preserves_boundary_normals_in_named_batch(self) -> None:
        dtype = np.dtype(
            [
                ("x_norm", "f8"),
                ("y_norm", "f8"),
                ("sdf_norm", "f8"),
                ("p_norm", "f8"),
                ("u_norm", "f8"),
                ("v_norm", "f8"),
                ("boundary_type", "U20"),
                ("on_boundary", "bool"),
                ("normal_x", "f8"),
                ("normal_y", "f8"),
            ]
        )
        points = np.zeros(1, dtype=dtype)
        points["boundary_type"] = "slip_wall"
        points["on_boundary"] = True
        points["normal_x"] = 0.6
        points["normal_y"] = 0.8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.npy"
            np.save(path, points)
            sample = PINNDataset(str(path))[0]

        torch.testing.assert_close(
            sample.normal_x,
            torch.tensor([0.6], dtype=torch.float32),
        )
        torch.testing.assert_close(
            sample.normal_y,
            torch.tensor([0.8], dtype=torch.float32),
        )
        self.assertEqual(len(tuple(sample)), 11)


if __name__ == "__main__":
    unittest.main()
