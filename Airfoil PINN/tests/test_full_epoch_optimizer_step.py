"""Regression tests for the optional full-epoch Adam update."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import torch

from airfoil_pinn.training.components.trainer import Trainer


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))


def _batch_result(total: torch.Tensor) -> dict:
    value = float(total.detach())
    zeros = {name: 0.0 for name in ("cont", "mom_x", "mom_y")}
    counts = {name: 1 for name in zeros}
    return {
        "total": total,
        "pde_raw": dict(zeros),
        "pde_mse": dict(zeros),
        "pde_max": dict(zeros),
        "pde_counts": counts,
        "bc_comp": {},
        "bc_mse": {},
        "bc_counts": {},
        "extra_grad": 0.0,
        "integral_loss": 0.0,
        "fvm_loss": 0.0,
        "fvm_raw_loss": 0.0,
        "fvm_mse": 0.0,
        "fvm_weighted_loss": 0.0,
        "fvm_rms": 0.0,
        "pressure_gauge_loss": 0.0,
        "pressure_gauge_raw": 0.0,
        "pde_coverage": {},
        "wall_speed": None,
        "test_value": value,
    }


class FullEpochOptimizerStepTests(unittest.TestCase):
    def _trainer(self, *, full_epoch: bool) -> tuple[Trainer, _ScalarModel]:
        model = _ScalarModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.data_handler = SimpleNamespace(
            boundary_type_names=[],
            get_dataloader=lambda: [
                torch.tensor(1.0),
                torch.tensor(3.0),
            ],
        )
        trainer.loss_computer = SimpleNamespace(
            pde_components=["cont", "mom_x", "mom_y"],
            bc_type="hard",
            fvm=None,
            fvm_weight=0.0,
            integral_condition=False,
            integral_weight=0.0,
        )
        trainer.device = torch.device("cpu")
        trainer.dtype = torch.float32
        trainer.accum_steps = 1
        trainer.full_epoch_optimizer_step = full_epoch
        trainer.use_amp = False
        trainer.scaler = None
        trainer.sa_weights = None
        trainer.grad_clip = None
        trainer.loss_ma = {name: [] for name in trainer.loss_computer.pde_components}
        trainer._fvm_epoch_counter = 0
        trainer._fvm_optimizer_step_counter = 0
        trainer._last_fvm_loss = 0.0
        trainer._last_fvm_raw_loss = 0.0
        trainer._last_fvm_rms = 0.0
        trainer._last_fvm_evaluations = 0
        trainer._last_integral_loss = 0.0
        trainer._last_pressure_gauge_mse = 0.0
        trainer._last_pressure_gauge_loss = 0.0

        observed_values: list[float] = []

        def forward_pass(
            self: Trainer,
            target: torch.Tensor,
            *args,
            **kwargs,
        ) -> dict:
            del args, kwargs
            observed_values.append(float(model.value.detach()))
            return _batch_result((model.value - target).square())

        trainer.forward_pass = MethodType(forward_pass, trainer)
        trainer.observed_values = observed_values
        return trainer, model

    def test_full_epoch_mode_freezes_weights_until_last_batch(self) -> None:
        trainer, model = self._trainer(full_epoch=True)
        step_count = 0
        original_step = trainer.optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counted_step
        trainer.train_epoch_adam()

        self.assertEqual(step_count, 1)
        self.assertEqual(trainer.observed_values, [0.0, 0.0])
        self.assertAlmostEqual(float(model.value.detach()), 0.4, places=6)

    def test_regular_mode_still_updates_each_batch(self) -> None:
        trainer, _ = self._trainer(full_epoch=False)
        trainer.train_epoch_adam()

        self.assertEqual(len(trainer.observed_values), 2)
        self.assertEqual(trainer.observed_values[0], 0.0)
        self.assertNotEqual(trainer.observed_values[1], 0.0)

    def test_full_epoch_fvm_schedule_targets_only_final_batch(self) -> None:
        trainer, _ = self._trainer(full_epoch=True)
        trainer.fvm_schedule = "every_optimizer_step"
        trainer.fvm_batch_interval = 1
        self.assertEqual(trainer._scheduled_fvm_batches(5, True), {4: 1.0})


if __name__ == "__main__":
    unittest.main()
