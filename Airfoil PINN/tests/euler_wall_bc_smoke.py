"""Regression tests for PDE-dependent wall boundary conditions."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import (
    ConfigurationError,
    validate_static_config,
)
from airfoil_pinn.data.flow import FlowCase
from airfoil_pinn.data.handler import DataHandler, PINNDataset
from airfoil_pinn.geometry.sdf import PolylineDistanceField
from airfoil_pinn.training.components.loss_computer import LossComputer


def _loss_config(pde_system: str) -> dict:
    config = copy.deepcopy(ConfigManager().config)
    config["pde"]["system"] = pde_system
    config["boundary_conditions"]["type"] = "soft"
    return config


def _wall_loss_computer(pde_system: str) -> LossComputer:
    handler = SimpleNamespace(
        boundary_type_names=["wall"],
        mesh_airfoil_coords=np.asarray(
            [[-1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        ),
        chord=1.0,
        sampler_metadata={"foil_closed": False},
        flow_cases={
            0: FlowCase(
                case_id=0,
                reynolds=1.0e5,
                angle_degrees=0.0,
            )
        },
        dataset=SimpleNamespace(type_to_idx={"wall": 0}),
    )
    handler.get_flow_case_batch = MethodType(
        DataHandler.get_flow_case_batch,
        handler,
    )
    return LossComputer(
        config=_loss_config(pde_system),
        data_handler=handler,
        rho_norm=torch.tensor(1.0, dtype=torch.float64),
        mu_norm=torch.tensor(1.0e-5, dtype=torch.float64),
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


class EulerWallLossTests(unittest.TestCase):
    """Euler walls constrain only velocity normal to the SDF surface."""

    def test_euler_leaves_tangential_velocity_unconstrained(self) -> None:
        loss = _wall_loss_computer("euler")
        x = torch.tensor([[-0.5], [0.5]], dtype=torch.float64)
        y = torch.zeros_like(x)
        u = torch.tensor(
            [[2.0], [0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        v = torch.tensor(
            [[0.0], [3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        boundary = torch.ones((2, 1), dtype=torch.float64)

        components, _ = loss.compute_bc_loss(
            u,
            v,
            torch.zeros_like(u),
            boundary,
            x,
            y,
        )

        torch.testing.assert_close(
            components["wall"],
            torch.tensor(4.5, dtype=torch.float64),
        )
        components["wall"].backward()
        torch.testing.assert_close(
            u.grad,
            torch.zeros_like(u),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            v.grad,
            torch.tensor([[0.0], [3.0]], dtype=torch.float64),
        )

    def test_navier_stokes_retains_no_slip_wall(self) -> None:
        loss = _wall_loss_computer("incompressible_navier_stokes")
        x = torch.tensor([[-0.5], [0.5]], dtype=torch.float64)
        y = torch.zeros_like(x)
        u = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
        v = torch.tensor([[0.0], [3.0]], dtype=torch.float64)
        boundary = torch.ones((2, 1), dtype=torch.float64)

        components, _ = loss.compute_bc_loss(
            u,
            v,
            torch.zeros_like(u),
            boundary,
            x,
            y,
        )

        torch.testing.assert_close(
            components["wall"],
            torch.tensor(6.5, dtype=torch.float64),
        )

    def test_surface_normal_is_sdf_segment_normal(self) -> None:
        field = PolylineDistanceField(
            np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
            closed=False,
        )
        normals = field.surface_normals(
            torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        )
        tangent = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=1, keepdim=True)

        torch.testing.assert_close(
            torch.sum(normals * tangent, dim=1),
            torch.zeros(1, dtype=torch.float64),
            atol=1.0e-15,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(normals, dim=1),
            torch.ones(1, dtype=torch.float64),
        )


class EulerHardBoundaryTests(unittest.TestCase):
    """Independent component masks must not silently create Euler no-slip."""

    @staticmethod
    def _write_points(path: Path) -> None:
        dtype = np.dtype(
            [
                ("x_norm", "f8"),
                ("y_norm", "f8"),
                ("sdf_norm", "f8"),
                ("p_norm", "f8"),
                ("u_norm", "f8"),
                ("v_norm", "f8"),
                ("boundary_type", "U16"),
            ]
        )
        points = np.zeros(1, dtype=dtype)
        points["boundary_type"] = "wall"
        np.save(path, points)

    def test_euler_wall_does_not_fix_velocity_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.npy"
            self._write_points(path)
            dataset = PINNDataset(str(path), pde_system="euler")

        self.assertFalse(bool(dataset.mask_u_fixed[0]))
        self.assertFalse(bool(dataset.mask_v_fixed[0]))

    def test_navier_stokes_wall_still_fixes_both_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.npy"
            self._write_points(path)
            dataset = PINNDataset(
                str(path),
                pde_system="incompressible_navier_stokes",
            )

        self.assertTrue(bool(dataset.mask_u_fixed[0]))
        self.assertTrue(bool(dataset.mask_v_fixed[0]))

    def test_euler_hard_bc_configuration_is_rejected(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["pde"]["system"] = "euler"
        config["boundary_conditions"]["type"] = "hard"

        with self.assertRaisesRegex(
            ConfigurationError,
            "requires boundary_conditions.type=soft",
        ):
            validate_static_config(config)


if __name__ == "__main__":
    unittest.main()
