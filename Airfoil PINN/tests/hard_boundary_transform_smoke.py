"""Tests for distance-based hard no-slip and impermeability operators."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import (
    ConfigurationError,
    validate_static_config,
)
from airfoil_pinn.geometry.sdf import PolylineDistanceField
from airfoil_pinn.training.boundary import (
    BoundaryOutputTransform,
    distance_mask,
)
from airfoil_pinn.training.components.loss_computer import LossComputer


def _line_field() -> PolylineDistanceField:
    return PolylineDistanceField(
        np.array([[-2.0, 0.0], [2.0, 0.0]]),
        closed=False,
    )


def _hard_transform(
    *,
    dtype: torch.dtype = torch.float64,
    scale: float = 0.2,
) -> BoundaryOutputTransform:
    return BoundaryOutputTransform(
        _line_field(),
        wall_kind="no_slip",
        wall_enforcement="hard",
        distance_scale=scale,
    ).to(dtype=dtype)


class BoundaryOutputTransformTests(unittest.TestCase):
    """Verify exact values, trace continuity and PDE autograd derivatives."""

    def test_unsampled_wall_points_ignore_raw_velocity(self) -> None:
        transform = _hard_transform()
        x = torch.linspace(-1.9, 1.9, 37, dtype=torch.float64).reshape(-1, 1)
        y = torch.zeros_like(x)
        raw_u = torch.randn_like(x)
        raw_v = torch.randn_like(x)
        raw_p = torch.randn_like(x)

        u, v, p = transform(raw_u, raw_v, raw_p, x, y)

        torch.testing.assert_close(u, torch.zeros_like(u), atol=1e-12, rtol=0.0)
        torch.testing.assert_close(v, torch.zeros_like(v), atol=1e-12, rtol=0.0)
        self.assertIs(p, raw_p)

    def test_constraint_needs_no_boundary_labels_or_dataset_masks(self) -> None:
        transform = _hard_transform()
        x = torch.tensor([[-0.37], [0.73]], dtype=torch.float64)
        y = torch.zeros_like(x)
        u, v, _ = transform(
            torch.ones_like(x),
            -torch.ones_like(x),
            torch.zeros_like(x),
            x,
            y,
        )
        self.assertLess(float(u.abs().max()), 1e-12)
        self.assertLess(float(v.abs().max()), 1e-12)

    def test_near_wall_trace_tends_continuously_to_zero(self) -> None:
        transform = _hard_transform(scale=0.02)
        errors = []
        for epsilon in (1e-2, 1e-3, 1e-4, 1e-5):
            x = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1)
            y = torch.full_like(x, epsilon)
            u, v, _ = transform(
                torch.ones_like(x),
                2.0 * torch.ones_like(x),
                torch.zeros_like(x),
                x,
                y,
            )
            errors.append(torch.sqrt(u.square() + v.square()).mean())

        self.assertTrue(
            all(
                float(later) < float(earlier)
                for earlier, later in zip(errors, errors[1:])
            )
        )

    def test_manufactured_first_and_second_derivatives(self) -> None:
        scale = 0.3
        transform = _hard_transform(scale=scale)
        x = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.7]], dtype=torch.float64, requires_grad=True)
        raw_u = torch.sin(x)
        u, _, _ = transform(
            raw_u,
            torch.zeros_like(raw_u),
            torch.zeros_like(raw_u),
            x,
            y,
        )
        u_x, u_y = torch.autograd.grad(
            u,
            (x, y),
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u_y,
            y,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
        )[0]

        mask = y / (y + scale)
        torch.testing.assert_close(u, mask * torch.sin(x))
        torch.testing.assert_close(u_x, mask * torch.cos(x))
        torch.testing.assert_close(u_xx, -mask * torch.sin(x))
        torch.testing.assert_close(
            u_y,
            scale * torch.sin(x) / (y + scale).square(),
        )
        torch.testing.assert_close(
            u_yy,
            -2.0 * scale * torch.sin(x) / (y + scale).pow(3),
        )
        for value in (u, u_x, u_y, u_xx, u_yy):
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_piecewise_sdf_derivatives_are_finite_away_from_ties(self) -> None:
        field = PolylineDistanceField(
            np.array(
                [
                    [-1.0, 0.0],
                    [0.0, 0.15],
                    [1.0, 0.0],
                    [0.0, -0.15],
                ]
            ),
            closed=True,
        )
        transform = BoundaryOutputTransform(
            field,
            wall_kind="no_slip",
            wall_enforcement="hard",
            distance_scale=0.02,
        )
        points = torch.tensor(
            [
                [-0.999, 0.02],
                [0.999, 0.02],
                [-0.4, 0.25],
                [0.4, -0.25],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        nearest, _ = field.nearest_two(points)
        self.assertFalse(bool(field.tie_mask(points).any()))
        raw_u = torch.sin(points[:, 0:1]) + points[:, 1:2].square()
        u, _, _ = transform(
            raw_u,
            raw_u.cos(),
            raw_u,
            points[:, 0:1],
            points[:, 1:2],
            sdf=nearest,
        )
        first = torch.autograd.grad(
            u,
            points,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        second_derivative = torch.autograd.grad(
            first.sum(),
            points,
            create_graph=False,
        )[0]
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second_derivative).all()))

        switch = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(bool(field.tie_mask(switch).item()))

    def test_soft_transform_is_identity(self) -> None:
        transform = BoundaryOutputTransform(
            None,
            wall_kind="no_slip",
            wall_enforcement="soft",
            distance_scale=0.02,
        )
        values = [torch.randn(4, 1) for _ in range(3)]
        result = transform(
            *values,
            torch.zeros(4, 1),
            torch.zeros(4, 1),
        )
        self.assertTrue(all(actual is expected for actual, expected in zip(result, values)))

    def test_distance_scale_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            distance_mask(torch.ones(1, 1), 0.0)


class BoundaryConfigurationTests(unittest.TestCase):
    """Validate both equation-dependent hard wall operators."""

    def test_supported_hard_wall_configuration_is_valid(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["boundary_conditions"]["wall"]["enforcement"] = "hard"
        validate_static_config(config)

    def test_euler_overrides_stale_hard_wall_configuration(self) -> None:
        manager = ConfigManager()
        config = manager.config
        config["pde"]["system"] = "euler"
        config["boundary_conditions"]["wall"].update(
            {"kind": "inviscid_wall", "enforcement": "hard"}
        )
        manager.validate()
        self.assertEqual(config["boundary_conditions"]["wall"]["kind"], "slip_wall")
        self.assertEqual(config["boundary_conditions"]["wall"]["enforcement"], "hard")

    def test_global_hard_configuration_is_rejected(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["boundary_conditions"]["type"] = "hard"
        with self.assertRaisesRegex(
            ConfigurationError,
            "Global boundary_conditions.type=hard is deprecated",
        ):
            validate_static_config(config)


class HybridBoundaryLossTests(unittest.TestCase):
    """A hard wall must not turn off unrelated soft boundary residuals."""

    def test_hard_wall_is_skipped_but_outlet_pressure_remains_soft(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["boundary_conditions"]["wall"]["enforcement"] = "hard"
        config["loss_weights"]["bc_weights"]["outlet_grad"] = 0.0
        dataset = SimpleNamespace(
            type_to_idx={"wall": 0, "outlet": 1},
        )

        def get_flow_case_batch(case_ids, *, device, dtype):
            shape = (case_ids.numel(), 1)
            return SimpleNamespace(
                cos_alpha=torch.ones(shape, device=device, dtype=dtype),
                sin_alpha=torch.zeros(shape, device=device, dtype=dtype),
            )

        handler = SimpleNamespace(
            boundary_type_names=["wall", "outlet"],
            mesh_airfoil_coords=np.array([[-1.0, 0.0], [1.0, 0.0]]),
            chord=1.0,
            sampler_metadata={"foil_closed": False},
            dataset=dataset,
            get_flow_case_batch=get_flow_case_batch,
        )

        loss = LossComputer(
            config,
            handler,
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1e-3, dtype=torch.float64),
            -2.0,
            2.0,
            -1.0,
            1.0,
            torch.device("cpu"),
            torch.float64,
        )
        x = torch.tensor([[0.0], [1.5]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.0], [0.5]], dtype=torch.float64, requires_grad=True)
        boundary_types = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        components, _ = loss.compute_bc_loss(
            u=torch.tensor([[7.0], [0.0]], dtype=torch.float64),
            v=torch.tensor([[9.0], [0.0]], dtype=torch.float64),
            p=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
            boundary_types=boundary_types,
            x=x,
            y=y,
            normal_x=torch.tensor(
                [[float("nan")], [1.0]],
                dtype=torch.float64,
            ),
            normal_y=torch.tensor(
                [[float("nan")], [0.0]],
                dtype=torch.float64,
            ),
        )

        self.assertEqual(float(components["wall"]), 0.0)
        self.assertEqual(float(components["outlet"]), 4.0)


if __name__ == "__main__":
    unittest.main()
