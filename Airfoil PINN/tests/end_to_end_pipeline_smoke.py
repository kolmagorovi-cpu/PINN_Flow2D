"""End-to-end save/resume validation for the modern PINN pipeline."""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.solver import PINNSolver


def _small_solver_config() -> dict:
    """Return a CPU-only graph using the real normalized collocation data."""
    config = copy.deepcopy(ConfigManager().config)
    config["project_name"] = "end_to_end_smoke"
    config["architecture"]["hidden_layers"] = [8, 8]
    config["architecture"]["activation"] = "tanh"
    config["architecture"]["coordinate_scaling"] = {
        "enabled": True,
        "scale_factor": 1.5,
    }
    # Modern input_features must take precedence over contradictory legacy
    # flags throughout validation, SDF construction and model assembly.
    config["input_params"]["coordinates"] = False
    config["input_params"]["sdf"] = False
    config["input_features"] = {
        "coordinates": {"enabled": True},
        "sdf": {"enabled": True},
        "boundary_types": {"enabled": True},
    }
    config["training"]["batch_size"] = 128
    config["process"]["device"] = "cpu"
    config["data"]["mesh_file"] = None
    config["loss_weights"]["fvm"]["enabled"] = False
    config["loss_weights"]["fvm"]["weight"] = 0.0
    config["checkpoint"]["save_flow_images"] = False
    config["output"]["progress_bar"] = False
    config["enable_key_listener"] = False
    return config


class EndToEndResumeTests(unittest.TestCase):
    """Exercise graph construction, autodiff, optimizer and exact resume."""

    def test_checkpoint_restores_model_optimizer_data_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_directory = Path(directory, "first")
            config = _small_solver_config()
            solver = PINNSolver(
                config,
                experiment_dir=str(first_directory),
            )
            batch = next(iter(solver.data_handler.get_dataloader()))
            result = solver.trainer.train_batch_adam(
                batch,
                accum_step=True,
            )
            self.assertTrue(np.isfinite(result["total"]))
            self.assertGreater(result["grad_norm"], 0.0)

            solver._save_checkpoint(1)
            checkpoint_path = (
                first_directory / "checkpoints" / "pinn_1.pt"
            )
            self.assertTrue(checkpoint_path.exists())

            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(4)
            expected_loader = torch.rand(
                4,
                generator=solver.data_handler._loader_generator,
            )
            expected_model = {
                name: value.detach().clone()
                for name, value in solver._unwrap_model_for_state(
                    solver.model
                ).state_dict().items()
            }

            resumed_config = _small_solver_config()
            resumed_config["process"]["load_model_path"] = str(
                checkpoint_path
            )
            resumed = PINNSolver(
                resumed_config,
                experiment_dir=str(Path(directory, "resumed")),
            )

            self.assertEqual(resumed.start_epoch, 1)
            resumed_model = resumed._unwrap_model_for_state(resumed.model)
            for name, value in resumed_model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, expected_model[name]),
                    msg=name,
                )
            self.assertTrue(resumed.optimizer.state)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(
                torch.equal(
                    torch.rand(
                        4,
                        generator=resumed.data_handler._loader_generator,
                    ),
                    expected_loader,
                )
            )

            manifest = json.loads(
                resumed.telemetry.manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["runtime"]["start_epoch"], 1)
            self.assertEqual(
                manifest["training_contract"]["fingerprint"],
                resumed.training_contract.fingerprint(),
            )
            self.assertFalse(
                checkpoint_path.with_suffix(".pt.tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()
