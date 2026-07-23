"""Smoke tests for model smoothness, numerical guards and run telemetry."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import (
    ConfigurationError,
    validate_static_config,
)
from airfoil_pinn.data.handler import (
    DataHandler,
    _seed_dataloader_worker,
)
from airfoil_pinn.training.models import PINNIncompressible, Sine
from airfoil_pinn.training.network_diagram import _input_labels
from airfoil_pinn.training.numerics import (
    NumericalTrainingError,
    check_gradient_health,
    ensure_finite_loss,
)
from airfoil_pinn.training.progress import build_epoch_progress
from airfoil_pinn.training.telemetry import RunTelemetry


class SirenContractTests(unittest.TestCase):
    """Verify frequency scaling, initialization and coordinate derivatives."""

    def test_sine_applies_configured_frequency(self) -> None:
        activation = Sine(omega_0=7.0)
        value = torch.tensor([[0.25]], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(activation(value), torch.sin(7.0 * value))
        )

    def test_siren_layer_bounds_match_paper_initialization(self) -> None:
        torch.manual_seed(12)
        model = PINNIncompressible(
            input_dim=3,
            layers=[8, 8],
            activation="siren",
            output_dim=3,
            siren_first_omega_0=20.0,
            siren_hidden_omega_0=25.0,
        )
        first_bound = 1.0 / 3.0
        hidden_bound = math.sqrt(6.0 / 8.0) / 25.0
        self.assertLessEqual(
            float(model.hidden_layers[0].weight.detach().abs().max()),
            first_bound,
        )
        self.assertLessEqual(
            float(model.hidden_layers[1].weight.detach().abs().max()),
            hidden_bound,
        )
        self.assertLessEqual(
            float(model.output_layer.weight.detach().abs().max()),
            hidden_bound,
        )

    def test_siren_has_finite_second_coordinate_derivatives(self) -> None:
        model = PINNIncompressible(
            input_dim=3,
            layers=[12, 12],
            activation="siren",
            output_dim=3,
        ).double()
        inputs = torch.tensor(
            [[0.1, -0.2, 0.05], [0.3, 0.4, 0.12]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = model(inputs)[:, 0]
        first = torch.autograd.grad(
            output.sum(),
            inputs,
            create_graph=True,
        )[0]
        second = torch.autograd.grad(
            first[:, :2].sum(),
            inputs,
            create_graph=True,
        )[0]
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(torch.isfinite(second).all()))


class StaticRobustnessValidationTests(unittest.TestCase):
    """Check activation smoothness and numerical configuration rules."""

    def test_relu_is_rejected_for_viscous_navier_stokes(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["architecture"]["activation"] = "relu"
        with self.assertRaisesRegex(
            ConfigurationError,
            "twice-differentiable",
        ):
            validate_static_config(config)

    def test_relu_remains_available_for_first_order_euler(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["architecture"]["activation"] = "relu"
        config["pde"]["system"] = "euler"
        validate_static_config(config)

    def test_direct_float16_is_rejected(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["process"]["dtype"] = "float16"
        with self.assertRaisesRegex(ConfigurationError, "float16"):
            validate_static_config(config)

    def test_siren_frequencies_must_be_positive(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["architecture"]["siren_hidden_omega_0"] = 0.0
        with self.assertRaisesRegex(
            ConfigurationError,
            "siren_hidden_omega_0",
        ):
            validate_static_config(config)


class NumericalGuardTests(unittest.TestCase):
    """Ensure non-finite losses and gradients fail before optimizer steps."""

    def test_nonfinite_loss_reports_components(self) -> None:
        with self.assertRaisesRegex(
            NumericalTrainingError,
            "pde=1.000000e\\+00",
        ):
            ensure_finite_loss(
                torch.tensor(float("nan")),
                stage="test",
                components={"pde": 1.0},
            )

    def test_gradient_health_reports_norm_and_nan_parameter(self) -> None:
        good = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        good.grad = torch.tensor([3.0, 4.0])
        health = check_gradient_health(
            [("good", good)],
            stage="test",
        )
        self.assertAlmostEqual(health.l2_norm, 5.0)
        self.assertAlmostEqual(health.max_abs, 4.0)

        bad = torch.nn.Parameter(torch.tensor([1.0]))
        bad.grad = torch.tensor([float("nan")])
        with self.assertRaisesRegex(
            NumericalTrainingError,
            "bad",
        ):
            check_gradient_health([("bad", bad)], stage="test")


class OperationalRobustnessTests(unittest.TestCase):
    """Verify telemetry, progress fallback and worker reproducibility."""

    def test_telemetry_is_strict_json_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            telemetry = RunTelemetry(directory)
            telemetry.write_manifest({"tensor": torch.tensor(2.0)})
            telemetry.append_epoch({"epoch": 1, "loss": np.float64(0.5)})
            manifest = json.loads(
                telemetry.manifest_path.read_text(encoding="utf-8")
            )
            epoch = json.loads(
                telemetry.epochs_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["tensor"], 2.0)
            self.assertEqual(epoch["epoch"], 1)
            with self.assertRaises(ValueError):
                telemetry.append_epoch({"loss": float("nan")})
            self.assertFalse(
                any(Path(directory).glob("*.tmp"))
            )

    def test_progress_has_plain_iterable_fallback(self) -> None:
        epochs = [1, 2, 3]
        self.assertIs(
            build_epoch_progress(epochs, enabled=False),
            epochs,
        )

    def test_network_diagram_uses_modern_manifest_labels(self) -> None:
        config = copy.deepcopy(ConfigManager().config)
        config["data"]["points_file"] = "missing-for-diagram-test.npy"
        config["input_features"] = {
            "coordinates": {"enabled": True},
            "sdf": {"enabled": True},
            "reynolds": {
                "enabled": True,
                "range": [1e4, 1e6],
            },
            "angle_of_attack": {"enabled": True},
            "boundary_types": {"enabled": False},
            "geometry_parameters": {
                "enabled": True,
                "names": ["thickness_over_chord"],
            },
        }
        self.assertEqual(
            _input_labels(config),
            [
                "x/c",
                "y/c",
                "SDF/c",
                "scaled log10(Re_c)",
                "cos(alpha)",
                "sin(alpha)",
                "thickness_over_chord",
            ],
        )

    def test_worker_seeding_is_reproducible(self) -> None:
        torch.manual_seed(123)
        _seed_dataloader_worker(0)
        first = np.random.random(4)
        torch.manual_seed(123)
        _seed_dataloader_worker(0)
        second = np.random.random(4)
        self.assertTrue(np.array_equal(first, second))

    def test_spawn_safe_dataloader_worker(self) -> None:
        config = ConfigManager().config
        handler = DataHandler(
            points_file=config["data"]["points_file"],
            batch_size=32,
            mesh_file=None,
            num_workers=1,
            persistent_workers=False,
            seed=19,
        )
        iterator = iter(handler.get_dataloader())
        batch = next(iterator)
        self.assertEqual(batch.x.shape[0], 32)
        del iterator


if __name__ == "__main__":
    unittest.main()
