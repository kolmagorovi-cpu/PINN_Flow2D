"""Analytical checks for case-aware PDE, boundary conditions and SA-PINN."""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
import unittest

import torch

from airfoil_pinn.data.flow import FlowCase
from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.physics.pde import PDEResiduals
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.components.rar import RARManager
from airfoil_pinn.training.models import SelfAdaptiveWeights
from airfoil_pinn.utils import safe_grad


def _grad(output: torch.Tensor, variable: torch.Tensor, **_: object) -> torch.Tensor:
    """Differentiate while allowing analytically constant derivatives."""
    return safe_grad(
        output,
        variable,
        create_graph=True,
        silent=True,
        strict=False,
    )


class DimensionlessPDETests(unittest.TestCase):
    """Verify exact coefficients and parameter gradients in the NS operator."""

    def test_pointwise_reynolds_controls_viscous_residual(self) -> None:
        """Two equal coordinates with different Re produce analytical values."""
        x = torch.ones((2, 1), dtype=torch.float64, requires_grad=True)
        y = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
        reynolds = torch.tensor(
            [[10.0], [100.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        u = x.square() + 0.0 * y
        v = 0.0 * x + 0.0 * y
        pressure = 0.0 * x + 0.0 * y

        residuals = PDEResiduals.compute_residuals_2d(
            u,
            v,
            pressure,
            x,
            y,
            reynolds.reciprocal(),
            _grad,
        )

        expected = torch.tensor(
            [[(2.0 - 0.2) ** 2], [(2.0 - 0.02) ** 2]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(residuals["mom_x"], expected)
        parameter_gradient = torch.autograd.grad(
            residuals["mom_x"].sum(),
            reynolds,
        )[0]
        self.assertTrue(bool(torch.isfinite(parameter_gradient).all()))
        self.assertTrue(bool((parameter_gradient > 0.0).all()))

    def test_cp_pressure_gradient_has_half_coefficient(self) -> None:
        """Cp normalization yields ``(0.5 * dCp/dx)^2`` in momentum."""
        x = torch.tensor(
            [[0.2], [0.7]],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torch.zeros_like(x, requires_grad=True)
        zero = 0.0 * x + 0.0 * y
        residuals = PDEResiduals.compute_residuals_2d(
            zero,
            zero,
            x + 0.0 * y,
            x,
            y,
            torch.tensor(1e-5, dtype=torch.float64),
            _grad,
        )
        torch.testing.assert_close(
            residuals["mom_x"],
            torch.full_like(x, 0.25),
        )

    def test_inverse_reynolds_vector_cannot_broadcast_to_matrix(self) -> None:
        """A per-point vector is reshaped to a column, never an N-by-N field."""
        reference = torch.zeros((3, 1), dtype=torch.float64)
        coefficient = PDEResiduals._inverse_reynolds_like(
            torch.tensor([0.1, 0.01, 0.001], dtype=torch.float64),
            reference,
        )
        self.assertEqual(coefficient.shape, reference.shape)


def _loss_config() -> dict:
    """Return a complete minimal configuration for case-aware loss tests."""
    return {
        "input_params": {
            "coordinates": True,
            "sdf": True,
            "boundary_types": True,
            "viscosity": True,
            "inlet_velocity": True,
        },
        "pde": {
            "flow_type": "2d",
            "system": "incompressible_navier_stokes",
        },
        "boundary_conditions": {"type": "soft"},
        "loss_weights": {
            "pde_weights": {
                "cont": 1.0,
                "mom_x": 1.0,
                "mom_y": 1.0,
            },
            "bc_weights": {
                "velocity_dirichlet": 1.0,
                "interior": 0.0,
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


def _loss_computer() -> LossComputer:
    """Construct a two-case LossComputer without filesystem artifacts."""
    handler = SimpleNamespace(
        boundary_type_names=["interior", "velocity_dirichlet"],
        inlet_u_norm=1.0,
        inlet_v_norm=0.0,
        mesh_airfoil_coords=None,
        chord=1.0,
        flow_cases={
            0: FlowCase(
                case_id=0,
                reynolds=10.0,
                angle_degrees=0.0,
            ),
            1: FlowCase(
                case_id=1,
                reynolds=100.0,
                angle_degrees=30.0,
            ),
        },
        dataset=SimpleNamespace(
            type_to_idx={"interior": 0, "velocity_dirichlet": 1}
        ),
    )
    handler.get_flow_case_batch = MethodType(
        DataHandler.get_flow_case_batch,
        handler,
    )
    return LossComputer(
        config=_loss_config(),
        data_handler=handler,
        rho_norm=torch.tensor(1.0, dtype=torch.float64),
        mu_norm=torch.tensor(0.1, dtype=torch.float64),
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


class CaseAwareLossTests(unittest.TestCase):
    """Verify that the same case IDs drive inputs, physics and BC targets."""

    def test_soft_boundary_targets_use_sin_cos_alpha(self) -> None:
        """Exact freestream vectors for alpha=0 and 30 degrees give zero."""
        loss = _loss_computer()
        x = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
        y = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        case_ids = torch.tensor([0, 1])
        u = torch.tensor(
            [[1.0], [math.cos(math.radians(30.0))]],
            dtype=torch.float64,
        )
        v = torch.tensor(
            [[0.0], [math.sin(math.radians(30.0))]],
            dtype=torch.float64,
        )
        boundary = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        components, _ = loss.compute_bc_loss(
            u,
            v,
            torch.zeros_like(u),
            boundary,
            x,
            y,
            case_ids=case_ids,
        )
        torch.testing.assert_close(
            components["velocity_dirichlet"],
            torch.zeros((), dtype=torch.float64),
            atol=1e-15,
            rtol=0.0,
        )

    def test_hard_velocity_constraint_is_case_specific(self) -> None:
        """Hard Dirichlet values cannot silently fall back to case zero."""
        loss = _loss_computer()
        boundary = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        u, v = loss.apply_velocity_dirichlet_constraint(
            torch.zeros((2, 1), dtype=torch.float64),
            torch.zeros((2, 1), dtype=torch.float64),
            boundary,
            case_ids=torch.tensor([0, 1]),
        )
        torch.testing.assert_close(
            u,
            torch.tensor(
                [[1.0], [math.cos(math.radians(30.0))]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            v,
            torch.tensor([[0.0], [0.5]], dtype=torch.float64),
        )

    def test_rar_deduplication_keeps_distinct_case_associations(self) -> None:
        """The same coordinate remains valid once for every flow case."""
        x, y, sdf, case_ids = RARManager._deduplicate(
            x=torch.tensor([0.2, 0.2, 0.2]).numpy(),
            y=torch.tensor([0.3, 0.3, 0.3]).numpy(),
            sdf=torch.tensor([0.1, 0.1, 0.1]).numpy(),
            case_id=torch.tensor([0, 0, 1]).numpy(),
        )
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(len(sdf), 2)
        self.assertEqual(case_ids.tolist(), [0, 1])


class SelfAdaptiveWeightTests(unittest.TestCase):
    """Verify the maximizing side of the SA-PINN saddle objective."""

    def test_attention_increases_for_positive_loss(self) -> None:
        """A descent optimizer performs ascent on gradient-reversed weights."""
        weights = SelfAdaptiveWeights(2, 1, init_val=1.0)
        optimizer = torch.optim.SGD(weights.parameters(), lr=0.1)
        before = weights().detach().clone()
        weights().sum().backward()
        optimizer.step()
        weights.post_step_()
        after = weights().detach()
        self.assertTrue(bool((after > before).all()))

    def test_rar_extension_preserves_gradient_reversal(self) -> None:
        """Replacing the parameter during RAR keeps its ascent hook active."""
        weights = SelfAdaptiveWeights(1, 1, init_val=1.0)
        weights.extend(2)
        optimizer = torch.optim.SGD(weights.parameters(), lr=0.1)
        before = weights().detach().clone()
        weights().sum().backward()
        optimizer.step()
        self.assertEqual(tuple(weights.log_weights.shape), (3, 1))
        self.assertTrue(bool((weights().detach() > before).all()))


if __name__ == "__main__":
    unittest.main()
