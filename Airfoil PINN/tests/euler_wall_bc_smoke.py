"""Regression tests for PDE-dependent wall boundary conditions."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

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
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.boundary import resolve_boundary_conditions
from airfoil_pinn.training.menus.loss import loss_function_menu


def _loss_config(pde_system: str) -> dict:
    config = copy.deepcopy(ConfigManager().config)
    config["pde"]["system"] = pde_system
    if pde_system == "euler":
        config["boundary_conditions"]["wall"]["kind"] = "slip_wall"
    config["boundary_conditions"]["type"] = "soft"
    return config


def _wall_loss_computer(pde_system: str, wall: dict | None = None) -> LossComputer:
    config = _loss_config(pde_system)
    if wall is not None:
        config["boundary_conditions"]["wall"].update(wall)
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
        config=config,
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

    def test_direct_loss_construction_overrides_stale_wall_settings(self) -> None:
        for system, stale, expected in (
            ("euler", "no_slip", "slip_wall"),
            ("incompressible_navier_stokes", "slip_wall", "no_slip"),
        ):
            with self.subTest(system=system):
                loss = _wall_loss_computer(
                    system, {"kind": stale, "enforcement": "hard"}
                )
                self.assertEqual(loss.boundary_conditions["wall"]["kind"], expected)
                self.assertEqual(
                    loss.boundary_enforcement_contract["wall"]["kind"], expected
                )
                self.assertEqual(
                    loss.config["boundary_conditions"]["wall"]["kind"], expected
                )
                x = torch.tensor([[-0.5], [0.5]], dtype=torch.float64)
                y = torch.zeros_like(x)
                u, v, p = torch.ones_like(x), 2 * torch.ones_like(x), torch.zeros_like(x)
                actual_u, actual_v, _ = loss.transform_outputs(u, v, p, x, y)
                self.assertTrue(loss.boundary_transform.enabled)
                if system == "euler":
                    torch.testing.assert_close(actual_u, u)
                    torch.testing.assert_close(actual_v, torch.zeros_like(v))
                else:
                    self.assertTrue(loss.boundary_transform.enabled)
                    torch.testing.assert_close(actual_u, torch.zeros_like(u))
                    torch.testing.assert_close(actual_v, torch.zeros_like(v))

    def test_pde_specific_wall_in_all_loss_paths(self) -> None:
        for system in ("euler", "incompressible_navier_stokes"):
            with self.subTest(system=system):
                stale_kind = "no_slip" if system == "euler" else "slip_wall"
                loss = _wall_loss_computer(system, {"kind": stale_kind})
                loss.data_handler.boundary_type_names = ["wall", "interior"]
                # An oblique surface ensures neither Cartesian component
                # individually vanishes for a purely tangential flow.
                loss.sdf_module = PolylineDistanceField(
                    np.asarray([[-1.0, -1.0], [1.0, 1.0]]), closed=False
                )
                x = torch.tensor([[-0.5], [0.5], [0.0]], dtype=torch.float64)
                y = torch.tensor([[-0.5], [0.5], [1.0]], dtype=torch.float64)
                u = torch.tensor([[2.0], [2.0], [100.0]], dtype=torch.float64,
                                 requires_grad=True)
                v = torch.tensor([[2.0], [-2.0], [-100.0]], dtype=torch.float64,
                                 requires_grad=True)
                p = torch.zeros_like(u)
                boundary = torch.tensor(
                    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
                )
                weights = torch.tensor(
                    [[2.0, 1.0], [3.0, 1.0], [10.0, 1.0]], dtype=torch.float64
                )
                mask = boundary[:, 0:1] > 0.5
                trainer = SimpleNamespace(
                    loss_computer=loss, data_handler=loss.data_handler,
                    device=loss.device, dtype=loss.dtype,
                    sa_weights=None,
                )
                components, _ = loss.compute_bc_loss(u, v, p, boundary, x, y)
                sa = Trainer._compute_sa_bc_loss(
                    trainer, u, v, p, x, y, boundary, weights
                )
                rar = Trainer._compute_bc_point_score(
                    trainer, u, v, p, x, y, boundary, None
                )
                expected_points = torch.tensor(
                    [0.0, 8.0, 0.0] if system == "euler" else [8.0, 8.0, 0.0],
                    dtype=torch.float64,
                )
                expected = expected_points[:2].mean()
                torch.testing.assert_close(components["wall"], expected)
                torch.testing.assert_close(
                    sa, (expected_points[:2] * weights[:2, 0]).mean()
                )
                torch.testing.assert_close(rar, expected_points)
                torch.testing.assert_close(
                    loss.compute_wall_point_loss(u, v, x, y, mask).flatten(),
                    expected_points,
                )
                trainer.sa_weights = object()
                loss.sa_apply_to_bc = True
                rar_sa = Trainer._compute_bc_point_score(
                    trainer, u, v, p, x, y, boundary, weights
                )
                torch.testing.assert_close(rar_sa, expected_points * weights[:, 0])
                for objective, scale in (
                    (components["wall"], 1.0),
                    (sa, weights[:, 0:1]),
                    (rar.sum(), 2.0),
                    (rar_sa.sum(), 2.0 * weights[:, 0:1]),
                ):
                    du, dv = torch.autograd.grad(
                        objective, (u, v), retain_graph=True
                    )
                    if system == "euler":
                        torch.testing.assert_close(du + dv, torch.zeros_like(u))
                    else:
                        torch.testing.assert_close(du, scale * u * mask)
                        torch.testing.assert_close(dv, scale * v * mask)
                    torch.testing.assert_close(du[~mask], torch.zeros_like(du[~mask]))
                    torch.testing.assert_close(dv[~mask], torch.zeros_like(dv[~mask]))
                transformed = loss.boundary_transform(u, v, p, x, y)
                torch.testing.assert_close(transformed[0], u)
                torch.testing.assert_close(transformed[1], v)

    def test_hard_wall_is_excluded_from_sa_and_rar_but_inlet_is_soft(self) -> None:
        loss = _wall_loss_computer(
            "incompressible_navier_stokes", {"enforcement": "hard"}
        )
        loss.data_handler.boundary_type_names = ["wall", "inlet"]
        loss.sa_apply_to_bc = True
        trainer = SimpleNamespace(
            loss_computer=loss, data_handler=loss.data_handler,
            device=loss.device, dtype=loss.dtype, sa_weights=object(),
        )
        x = torch.tensor([[-0.5], [0.5], [-1.0]], dtype=torch.float64)
        y = torch.zeros_like(x)
        # Deliberately nonzero wall values detect accidental soft penalties,
        # even though the production hard transform already zeros them.
        u = torch.tensor([[7.0], [9.0], [3.0]], dtype=torch.float64,
                         requires_grad=True)
        v = torch.tensor([[4.0], [5.0], [4.0]], dtype=torch.float64,
                         requires_grad=True)
        p = torch.zeros_like(u)
        boundary = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        weights = torch.full_like(boundary, 2.0)
        components, _ = loss.compute_bc_loss(u, v, p, boundary, x, y)
        sa = Trainer._compute_sa_bc_loss(
            trainer, u, v, p, x, y, boundary, weights
        )
        rar = Trainer._compute_bc_point_score(
            trainer, u, v, p, x, y, boundary, weights
        )
        self.assertEqual(components["wall"].item(), 0.0)
        self.assertEqual(components["inlet"].item(), 20.0)
        self.assertEqual(sa.item(), 40.0)
        torch.testing.assert_close(
            rar, torch.tensor([0.0, 0.0, 40.0], dtype=torch.float64)
        )
        for objective in (components["inlet"], sa, rar.sum()):
            du, dv = torch.autograd.grad(objective, (u, v), retain_graph=True)
            torch.testing.assert_close(du[:2], torch.zeros_like(du[:2]))
            torch.testing.assert_close(dv[:2], torch.zeros_like(dv[:2]))

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

    def test_yaml_loading_resolves_pde_specific_wall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            manager = ConfigManager()
            for system, kind in (
                ("euler", "slip_wall"),
                ("incompressible_navier_stokes", "no_slip"),
            ):
                for wall_section in (
                    "",
                    "boundary_conditions:\n  wall:\n    enforcement: soft\n",
                    f"boundary_conditions:\n  wall:\n    kind: {kind}\n",
                ):
                    with self.subTest(system=system, wall_section=wall_section):
                        path.write_text(
                            f"pde:\n  system: {system}\n{wall_section}",
                            encoding="utf-8",
                        )
                        manager.load_from_file(str(path))
                        self.assertEqual(
                            manager.config["boundary_conditions"]["wall"]["kind"],
                            kind,
                        )

    def test_yaml_loading_overrides_conflicting_wall_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for system, wrong_kind, expected_kind in (
                ("euler", "no_slip", "slip_wall"),
                ("incompressible_navier_stokes", "slip_wall", "no_slip"),
            ):
                with self.subTest(system=system):
                    path.write_text(
                        f"pde:\n  system: {system}\n"
                        f"boundary_conditions:\n  wall:\n    kind: {wrong_kind}\n",
                        encoding="utf-8",
                    )
                    manager = ConfigManager()
                    manager.load_from_file(str(path))
                    self.assertEqual(
                        manager.config["boundary_conditions"]["wall"]["kind"],
                        expected_kind,
                    )

    def test_manual_wall_kind_is_ignored(self) -> None:
        config = _loss_config("incompressible_navier_stokes")
        config["boundary_conditions"]["wall"]["kind"] = "slip_typo"
        validate_static_config(config)
        self.assertEqual(
            resolve_boundary_conditions(config)["wall"]["kind"], "no_slip"
        )

    def test_explicit_euler_no_slip_is_overridden(self) -> None:
        config = _loss_config("euler")
        config["boundary_conditions"]["wall"]["kind"] = "no_slip"
        validate_static_config(config)
        self.assertEqual(
            resolve_boundary_conditions(config)["wall"]["kind"], "slip_wall"
        )

    def test_equation_switch_is_normalized_and_saved(self) -> None:
        manager = ConfigManager()
        manager.config["boundary_conditions"]["wall"]["enforcement"] = "hard"
        with tempfile.TemporaryDirectory() as directory:
            manager.CONFIG_DIR = directory
            for system, expected in (
                ("euler", "slip_wall"),
                ("incompressible_navier_stokes", "no_slip"),
            ):
                with self.subTest(system=system):
                    manager.config["pde"]["system"] = system
                    manager.save("switched")
                    wall = manager.config["boundary_conditions"]["wall"]
                    self.assertEqual(wall["kind"], expected)
                    self.assertEqual(wall["enforcement"], "hard")
                    loaded = ConfigManager()
                    loaded.load_from_file(str(Path(directory) / "switched.yaml"))
                    self.assertEqual(loaded.config["boundary_conditions"]["wall"], wall)

    def test_menu_equation_switch_updates_wall_immediately(self) -> None:
        manager = ConfigManager()
        manager.config["boundary_conditions"]["wall"]["enforcement"] = "hard"
        with patch("airfoil_pinn.training.menus.loss.clear_screen"), patch("builtins.print"):
            for choice, expected in (("2", "slip_wall"), ("1", "no_slip")):
                with self.subTest(choice=choice), patch(
                    "builtins.input", side_effect=["1", choice, "0"]
                ):
                    loss_function_menu(manager)
                    wall = manager.config["boundary_conditions"]["wall"]
                    self.assertEqual(wall["kind"], expected)
                    self.assertEqual(wall["enforcement"], "hard")

    def test_explicit_euler_slip_wall_is_valid(self) -> None:
        validate_static_config(_loss_config("euler"))

    def test_implicit_euler_wall_is_impermeable(self) -> None:
        config = _loss_config("euler")
        del config["boundary_conditions"]["wall"]["kind"]
        validate_static_config(config)
        self.assertEqual(
            resolve_boundary_conditions(config)["wall"]["kind"], "slip_wall"
        )

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

    def test_euler_legacy_global_hard_bc_configuration_is_rejected(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["pde"]["system"] = "euler"
        config["boundary_conditions"]["type"] = "hard"

        with self.assertRaisesRegex(
            ConfigurationError,
            "Global boundary_conditions.type=hard is deprecated",
        ):
            validate_static_config(config)


if __name__ == "__main__":
    unittest.main()
