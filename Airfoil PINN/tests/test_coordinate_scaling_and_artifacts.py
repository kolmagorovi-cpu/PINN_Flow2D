"""Regression tests for network-space scaling and terminal loss plots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import validate_static_config
from airfoil_pinn.training.models import PINNIncompressible
from airfoil_pinn.training.solver import PINNSolver
from airfoil_pinn.training.visualizer import Visualizer


class CoordinateScalingTests(unittest.TestCase):
    """Coordinate transforms must preserve physical autograd semantics."""

    def test_dataset_minmax_maps_xy_only_and_preserves_gradients(self) -> None:
        model = PINNIncompressible(
            input_dim=3,
            layers=[4],
            activation="siren",
            output_dim=3,
            output_bias=(0.0, 0.0, 0.0),
            coordinate_scaling={
                "enabled": True,
                "mode": "dataset_minmax",
                "coord_dim": 2,
                "resolved_min": [-6.0, -2.0],
                "resolved_max": [10.0, 2.0],
            },
            siren_first_omega_0=10.0,
            siren_hidden_omega_0=10.0,
        ).to(dtype=torch.float64)
        inputs = torch.tensor(
            [[-6.0, -2.0, 0.25], [2.0, 0.0, -0.5], [10.0, 2.0, 1.5]],
            dtype=torch.float64,
            requires_grad=True,
        )

        scaled = model._scale_model_inputs(inputs)

        expected = torch.tensor(
            [[-1.0, -1.0, 0.25], [0.0, 0.0, -0.5], [1.0, 1.0, 1.5]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(scaled, expected)
        scaled.sum().backward()
        torch.testing.assert_close(
            inputs.grad,
            torch.tensor(
                [[0.125, 0.5, 1.0]] * 3,
                dtype=torch.float64,
            ),
        )

    def test_test_hp_configuration_has_requested_schedule(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "workspace"
            / "configs"
            / "test_hp.yaml"
        )
        manager = ConfigManager()
        manager.load_from_file(str(config_path))
        config = manager.config
        validate_static_config(config)

        self.assertEqual(config["architecture"]["hidden_layers"], [64] * 4)
        self.assertEqual(config["architecture"]["activation"], "siren")
        self.assertEqual(config["architecture"]["siren_first_omega_0"], 10.0)
        self.assertEqual(config["architecture"]["siren_hidden_omega_0"], 10.0)
        self.assertEqual(
            config["architecture"]["coordinate_scaling"]["mode"],
            "dataset_minmax",
        )
        self.assertEqual(config["optimizer_params"]["lr"], 1e-4)
        self.assertEqual(config["training"]["epochs"], 2560)
        self.assertEqual(config["training"]["switch_to_lbfgs_epoch"], 2048)
        self.assertTrue(config["training"]["full_epoch_optimizer_step"])
        self.assertEqual(config["process"]["device"], "cpu")


class TerminalArtifactTests(unittest.TestCase):
    """Final loss plots must not depend on an enabled live window."""

    def test_loss_history_is_saved_in_console_only_mode(self) -> None:
        visualizer = Visualizer(
            plot_frequency=None,
            config_name="headless_test",
        )
        history = {
            "loss": [4.0, 2.0, 1.0],
            "pde_mse": {"cont": [1.0, 0.5, 0.25]},
            "bc_mse": {"wall": [0.8, 0.4, 0.2]},
            "total_pde": [2.0, 1.0, 0.5],
            "total_bc": [1.0, 0.5, 0.25],
            "integral_mse": [],
            "fvm_mse": [0.4, 0.3, 0.2],
            "pressure_gauge_mse": [0.1, 0.05, 0.025],
        }
        with tempfile.TemporaryDirectory() as directory:
            visualizer.save_loss_history(history, directory, "error")
            output = Path(directory, "loss_history_epoch_error.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_field_saving_continues_if_live_plot_saving_fails(self) -> None:
        visualizer = MagicMock()
        visualizer.save_plots.side_effect = RuntimeError("closed live window")
        solver = SimpleNamespace(
            visualizer=visualizer,
            metrics=SimpleNamespace(history={"loss": [1.0]}),
            _get_plots_dir=MagicMock(return_value="plots"),
            _save_checkpoint_flow_images=MagicMock(),
        )

        PINNSolver._save_terminal_artifacts(solver, "error")

        visualizer.save_loss_history.assert_called_once_with(
            solver.metrics.history,
            "plots",
            "error",
        )
        solver._save_checkpoint_flow_images.assert_called_once_with(
            "error",
            force=True,
            save_npy=True,
        )


if __name__ == "__main__":
    unittest.main()
