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

    def test_raw_pressure_shift_changes_total_loss(self) -> None:
        """The case-aware pressure gauge removes the PDE pressure nullspace."""
        with tempfile.TemporaryDirectory() as directory:
            solver = PINNSolver(
                _small_solver_config(),
                experiment_dir=str(Path(directory, "pressure_gauge")),
            )
            batch = next(iter(solver.data_handler.get_dataloader()))
            baseline = solver.trainer.forward_pass(batch)
            model = solver._unwrap_model_for_state(solver.model)
            with torch.no_grad():
                model.output_layer.bias[2].add_(10.0)
            shifted = solver.trainer.forward_pass(batch)

            self.assertNotEqual(
                float(baseline["total"].detach()),
                float(shifted["total"].detach()),
            )
            self.assertNotEqual(
                baseline["pressure_gauge_loss"],
                shifted["pressure_gauge_loss"],
            )

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

            # SDF is a derived cache.  A stale checkpoint column must not
            # survive resume even when its stored provenance claims the
            # current geometry.
            solver.data_handler.dataset.sdf.fill_(42.0)
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
            resumed_points = torch.cat(
                (
                    resumed.data_handler.dataset.x,
                    resumed.data_handler.dataset.y,
                ),
                dim=1,
            ).double().numpy()
            resumed_canonical_sdf = (
                resumed.loss_computer.evaluate_sdf_numpy(resumed_points)
            )
            resumed_sdf_error = float(
                np.max(
                    np.abs(
                        resumed.data_handler.dataset.sdf.numpy().reshape(-1)
                        - resumed_canonical_sdf
                    )
                )
            )
            self.assertLessEqual(
                resumed_sdf_error,
                resumed.loss_computer.sdf_float64_consistency_tolerance,
            )
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

    def test_hard_wall_trains_and_generalizes_to_new_surface_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _small_solver_config()
            config["boundary_conditions"]["wall"]["enforcement"] = "hard"
            solver = PINNSolver(
                config,
                experiment_dir=str(Path(directory, "hard_wall")),
            )

            batch = next(iter(solver.data_handler.get_dataloader()))
            result = solver.trainer.train_batch_adam(
                batch,
                accum_step=True,
            )
            self.assertTrue(np.isfinite(result["total"]))

            field = solver.loss_computer.sdf_module
            self.assertIsNotNone(field)
            starts = field.segment_starts[:16]
            ends = field.segment_ends[:16]
            points = starts + 0.371 * (ends - starts)
            x = (
                points[:, 0:1]
                .detach()
                .clone()
                .to(dtype=solver.dtype)
            )
            y = (
                points[:, 1:2]
                .detach()
                .clone()
                .to(dtype=solver.dtype)
            )
            phi = solver.loss_computer.boundary_transform.wall_distance(x, y)
            inputs = solver.loss_computer.build_input_tensor(x, y, phi)
            raw = solver.model(inputs)
            u, v, p = solver.loss_computer.transform_outputs(
                raw[:, 0:1],
                raw[:, 1:2],
                raw[:, 2:3],
                x,
                y,
            )

            self.assertLess(float(u.detach().abs().max()), 1e-5)
            self.assertLess(float(v.detach().abs().max()), 1e-5)
            torch.testing.assert_close(p, raw[:, 2:3])

    def test_boundary_transform_change_requires_finetune_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory, "source")
            source = PINNSolver(
                _small_solver_config(),
                experiment_dir=str(source_dir),
            )
            source._save_checkpoint(1)
            checkpoint_path = source_dir / "checkpoints" / "pinn_1.pt"

            incompatible = _small_solver_config()
            incompatible["boundary_conditions"]["wall"]["enforcement"] = "hard"
            incompatible["process"]["load_model_path"] = str(checkpoint_path)
            with self.assertRaisesRegex(
                RuntimeError,
                "exact resume is invalid",
            ):
                PINNSolver(
                    incompatible,
                    experiment_dir=str(Path(directory, "invalid_resume")),
                )

            finetune = copy.deepcopy(incompatible)
            finetune["checkpoint"]["load_mode"] = "finetune"
            restored = PINNSolver(
                finetune,
                experiment_dir=str(Path(directory, "finetune")),
            )
            self.assertEqual(restored.start_epoch, 0)
            self.assertFalse(restored.optimizer.state)


if __name__ == "__main__":
    unittest.main()
