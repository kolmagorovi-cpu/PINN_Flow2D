"""Regression tests for explicit MSE/RMS training telemetry."""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from airfoil_pinn.training.components.metrics import MetricsCollector
from airfoil_pinn.training.components.trainer import Trainer


def _loss_computer() -> SimpleNamespace:
    return SimpleNamespace(
        pde_components=["cont", "mom_x", "mom_y"],
        bc_type="soft",
        use_sa_weights=False,
        integral_condition=False,
        flow_type="planar",
        fvm_weight=25.0,
        loss_weights={
            "cont": 2.0,
            "mom_x": 3.0,
            "mom_y": 4.0,
            "inlet": 5.0,
        },
    )


def _data_handler() -> SimpleNamespace:
    return SimpleNamespace(boundary_type_names=["inlet"])


class LossMetricLoggingTests(unittest.TestCase):
    def test_history_stores_mse_rms_and_max_abs_separately(self) -> None:
        with tempfile.TemporaryDirectory() as logs_dir:
            metrics = MetricsCollector(
                model=torch.nn.Linear(1, 1),
                data_handler=_data_handler(),
                loss_computer=_loss_computer(),
                config={},
                device=torch.device("cpu"),
                dtype=torch.float32,
                logs_dir=logs_dir,
            )
            metrics.update_history(
                loss=10.0,
                pde_raw={"cont": 4.0, "mom_x": 9.0, "mom_y": 16.0},
                bc_comp={"inlet": 25.0},
                integral_loss=0.0,
                wall_speed=0.0,
                pde_max={"cont": 49.0, "mom_x": 64.0, "mom_y": 81.0},
                pde_rate={"cont": 1.0, "mom_x": 1.0, "mom_y": 1.0},
                grad_norm=0.0,
            )

        self.assertEqual(metrics.history["pde_mse"]["cont"][-1], 4.0)
        self.assertEqual(metrics.history["pde_rms"]["cont"][-1], 2.0)
        self.assertEqual(metrics.history["bc_mse"]["inlet"][-1], 25.0)
        self.assertEqual(metrics.history["bc_rms"]["inlet"][-1], 5.0)
        self.assertEqual(metrics.history["max_abs_res"]["cont"][-1], 7.0)

    def test_epoch_table_has_distinct_mse_and_rms_columns(self) -> None:
        metrics = MetricsCollector.__new__(MetricsCollector)
        metrics.loss_computer = _loss_computer()
        metrics.data_handler = _data_handler()
        metrics.history = {
            "loss": [],
            "pde": {"cont": [], "mom_x": [], "mom_y": []},
            "bc": {"inlet": []},
            "total_bc": [],
            "total_pde": [],
            "fvm_loss": [],
        }

        output = io.StringIO()
        with redirect_stdout(output):
            metrics.print_epoch_table(
                epoch=1,
                total_epochs=1,
                loss=10.0,
                pde_raw={"cont": 4.0, "mom_x": 9.0, "mom_y": 16.0},
                bc_comp={"inlet": 25.0},
                integral_loss=0.0,
                pde_max={"cont": 49.0, "mom_x": 64.0, "mom_y": 81.0},
                epoch_time=1.0,
                total_time=1.0,
                fvm_loss=100.0,
                fvm_raw_loss=4.0,
                fvm_rms=2.0,
                optimizer_name="Adam",
                learning_rate=1e-3,
                grad_norm=0.25,
                pde_multiplier=1.0,
                eta_seconds=60.0,
                avg_u_in=0.99,
                avg_u_out=0.97,
                mass_balance_error=0.02,
            )

        table = output.getvalue()
        self.assertIn("Epoch 1/1 [100.0%] · Adam", table)
        self.assertIn("ETA 01:00", table)
        self.assertIn("LR 1.00e-03 · Grad 2.50e-01", table)
        self.assertIn("│ PDE", table)
        self.assertIn("│ BOUNDARY", table)
        self.assertIn("│ AUXILIARY", table)
        self.assertIn("MSE", table)
        self.assertIn("RMS", table)
        self.assertIn("Max Abs", table)
        self.assertRegex(
            table,
            r"Continuity\s+│\s+4\.00e\+00\s+│\s+"
            r"2\.00e\+00\s+│\s+7\.00e\+00",
        )
        self.assertRegex(
            table,
            r"FVM Divergence\s+│\s+4\.00e\+00\s+│\s+"
            r"2\.00e\+00",
        )
        self.assertIn(
            "Flow: u_in=0.9900 · u_out=0.9700 · mass error=2.000e-02",
            table,
        )

    def test_epoch_mse_is_weighted_by_active_point_count(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.data_handler = SimpleNamespace(
            boundary_type_names=["inlet"],
            get_dataloader=lambda: [object(), object()],
        )
        trainer.loss_computer = SimpleNamespace(
            pde_components=["cont", "mom_x", "mom_y"],
            bc_type="soft",
            fvm=None,
            fvm_weight=0.0,
            integral_condition=False,
            integral_weight=0.0,
        )
        trainer.accum_steps = 1
        trainer.loss_ma = {
            "cont": [],
            "mom_x": [],
            "mom_y": [],
            "inlet": [],
        }
        trainer._fvm_epoch_counter = 0
        trainer._last_fvm_loss = 0.0
        trainer._last_fvm_raw_loss = 0.0
        trainer._last_fvm_rms = 0.0
        trainer._last_fvm_evaluations = 0
        trainer._last_integral_loss = 0.0

        def batch(
            pde_mse: float,
            pde_count: int,
            pde_max: float,
            bc_mse: float,
            bc_count: int,
        ) -> dict:
            return {
                "total": pde_mse + bc_mse,
                "pde_raw": {
                    "cont": pde_mse,
                    "mom_x": pde_mse,
                    "mom_y": pde_mse,
                },
                "pde_counts": {
                    "cont": pde_count,
                    "mom_x": pde_count,
                    "mom_y": pde_count,
                },
                "pde_max": {
                    "cont": pde_max,
                    "mom_x": pde_max,
                    "mom_y": pde_max,
                },
                "bc_comp": {"inlet": bc_mse},
                "bc_counts": {"inlet": bc_count},
                "extra_grad": 0.0,
                "integral_loss": 0.0,
                "wall_speed": None,
                "grad_norm": 0.0,
            }

        responses = iter(
            [
                batch(4.0, 1, 4.0, 9.0, 1),
                batch(1.0, 3, 16.0, 1.0, 3),
            ]
        )
        trainer.train_batch_adam = lambda *args, **kwargs: next(responses)

        (
            _,
            pde_mse,
            bc_mse,
            _,
            _,
            _,
            pde_max,
            _,
            _,
        ) = trainer.train_epoch_adam()

        self.assertAlmostEqual(pde_mse["cont"], 7.0 / 4.0)
        self.assertAlmostEqual(math.sqrt(pde_mse["cont"]), math.sqrt(7.0 / 4.0))
        self.assertAlmostEqual(bc_mse["inlet"], 12.0 / 4.0)
        self.assertEqual(pde_max["cont"], 16.0)


if __name__ == "__main__":
    unittest.main()
