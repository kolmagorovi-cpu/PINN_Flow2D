"""SA min/max updates, fixed BC factors, RAR state and exact resume."""

from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.data.batch import collate_pinn_batches
from airfoil_pinn.training.components.metrics import MetricsCollector
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.models import SelfAdaptiveWeights
from airfoil_pinn.training.solver import PINNSolver


def _trainer(*, amsgrad: bool = False) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device('cpu')
    trainer.dtype = torch.float32
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(2.0)
    trainer.sa_weights = SelfAdaptiveWeights(3, 5)
    trainer.optimizer = torch.optim.Adam(
        [
            {'params': trainer.model.parameters()},
            {'params': trainer.sa_weights.parameters()},
        ], lr=0.01, amsgrad=amsgrad,
    )
    trainer.data_handler = SimpleNamespace(
        boundary_type_names=['inlet', 'outlet'],
        dataset=SimpleNamespace(
            type_to_idx={}, x=torch.zeros(3, 1), y=torch.zeros(3, 1),
            sdf=torch.ones(3, 1),
            boundary_types=torch.tensor([[1., 0.], [0., 1.], [0., 1.]]),
        ),
    )
    trainer.loss_computer = SimpleNamespace(
        loss_weights={'inlet': 7.0, 'outlet': 3.0},
        sa_apply_to_bc=True, bc_type='soft', pde_on_boundary=True,
        pde_components=['cont', 'mom_x', 'mom_y'],
        is_hard_boundary=lambda name: False,
        get_freestream_targets=lambda u, cases: (torch.ones_like(u), torch.zeros_like(u)),
        prepare_sdf_for_pde=lambda x, y, sdf, bt: (sdf, torch.ones_like(sdf, dtype=torch.bool)),
    )
    return trainer


def _step(trainer: Trainer) -> None:
    trainer.optimizer.zero_grad()
    (trainer.sa_weights() * trainer.model.weight.square()).mean().backward()
    trainer.optimizer.step()
    trainer.sa_weights.post_step_()


class SelfAdaptiveTests(unittest.TestCase):
    def test_network_descends_and_attention_ascends(self) -> None:
        trainer = _trainer()
        before = trainer.sa_weights().detach().clone()
        _step(trainer)
        self.assertLess(trainer.model.weight.item(), 2.0)
        self.assertTrue(bool((trainer.sa_weights() > before).all()))

    def test_bc_coefficients_multiply_attention_and_rar_scores(self) -> None:
        trainer = _trainer()
        u = torch.tensor([[3.], [0.], [0.]])
        p = torch.tensor([[0.], [2.], [4.]])
        zero = torch.zeros_like(u)
        weights = torch.tensor([[2., 9.], [9., 3.], [9., 5.]], requires_grad=True)
        args = dict(u=u, v=zero, p=p, x=zero, y=zero,
                    boundary_types=trainer.data_handler.dataset.boundary_types, w_bc=weights)
        loss = trainer._compute_sa_bc_loss(**args)
        # inlet: 7 * (2 * 4); outlet: 3 * mean(3 * 4, 5 * 16)
        self.assertEqual(loss.item(), 194.0)
        loss.backward()
        torch.testing.assert_close(weights.grad, torch.tensor([[28., 0.], [0., 6.], [0., 24.]]))
        torch.testing.assert_close(trainer._compute_bc_point_score(**args), torch.tensor([56., 36., 240.]))
        trainer.loss_computer.loss_weights['inlet'] = 0.0
        self.assertEqual(trainer._compute_sa_bc_loss(**args).item(), 138.0)
        trainer.loss_computer.is_hard_boundary = lambda name: name == 'outlet'
        self.assertEqual(trainer._compute_sa_bc_loss(**args).item(), 0.0)

    def test_rar_preserves_network_and_attention_adam_state(self) -> None:
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                trainer = _trainer(amsgrad=amsgrad)
                _step(trainer)
                optimizer = trainer.optimizer
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
                scheduler.step(1.0)
                scheduler.step(2.0)
                old = trainer.sa_weights.log_weights
                state = copy.deepcopy(optimizer.state[old])
                model_state = optimizer.state[trainer.model.weight]
                before = old.detach().clone()
                trainer._extend_sa_weights(2)
                new = trainer.sa_weights.log_weights
                self.assertIs(trainer.optimizer, optimizer)
                self.assertIs(scheduler.optimizer, optimizer)
                self.assertIs(optimizer.state[trainer.model.weight], model_state)
                self.assertNotIn(old, optimizer.state)
                self.assertIs(optimizer.param_groups[1]['params'][0], new)
                self.assertEqual(optimizer.param_groups[1]['lr'], 0.001)
                torch.testing.assert_close(new[:3], before)
                torch.testing.assert_close(trainer.sa_weights()[3:], torch.ones(2, 5))
                for key, value in state.items():
                    if value.shape == old.shape:
                        torch.testing.assert_close(optimizer.state[new][key][:3], value)
                        self.assertEqual(torch.count_nonzero(optimizer.state[new][key][3:]).item(), 0)
                    else:
                        torch.testing.assert_close(optimizer.state[new][key], value)
                before = trainer.sa_weights().detach().clone()
                _step(trainer)
                self.assertTrue(bool((trainer.sa_weights() > before).all()))

    def test_rar_before_first_step_creates_valid_adam_state(self) -> None:
        trainer = _trainer()
        trainer._extend_sa_weights(2)
        self.assertEqual(len(trainer.optimizer.state), 0)
        _step(trainer)
        self.assertEqual(trainer.optimizer.state[trainer.sa_weights.log_weights]['exp_avg'].shape, (5, 5))

    def test_diagnostics_exclude_unrelated_and_disabled_bc_rows(self) -> None:
        trainer = _trainer()
        with torch.no_grad():
            trainer.sa_weights.log_weights[1:, 3].fill_(20.)
        report = trainer.compute_sa_diagnostics()
        self.assertEqual(report['bc/inlet']['count'], 1)
        self.assertAlmostEqual(report['bc/inlet']['mean'], 1.0)
        self.assertEqual(report['bc/outlet']['count'], 2)
        self.assertEqual(report['pde/cont']['count'], 3)
        self.assertEqual(sum(report['bc/outlet']['histogram']['counts']), 2)
        self.assertEqual(len(report['pde/mom_x']['quantiles']), 5)
        trainer.loss_computer.loss_weights['outlet'] = 0.0
        self.assertEqual(trainer.compute_sa_diagnostics()['bc/outlet'], {'count': 0})

    def test_convergence_history_and_best_restore_after_rar(self) -> None:
        trainer = _trainer()
        solver = PINNSolver.__new__(PINNSolver)
        solver.model, solver.sa_weights, solver.device = trainer.model, trainer.sa_weights, trainer.device
        solver.early_stopping_enabled = True
        solver.early_stopping_restore_best = True
        solver.early_stopping_min_delta = 0.0
        solver.early_stopping_patience = 1
        solver._early_best_loss = float('inf')
        physical = MetricsCollector.convergence_metric
        self.assertFalse(solver._update_early_stopping(1, physical({'cont': 2.0}, {'inlet': 3.0})))
        _step(trainer)
        # Physical error improves even as attention grows.
        self.assertFalse(solver._update_early_stopping(2, physical({'cont': 1.0}, {'inlet': 2.0})))
        best = trainer.model.weight.detach().clone()
        trainer._extend_sa_weights(2)
        _step(trainer)
        self.assertTrue(solver._update_early_stopping(3, 4.0))
        self.assertEqual(solver._early_best_epoch, 2)
        torch.testing.assert_close(trainer.model.weight, best)
        self.assertEqual(solver.sa_weights.log_weights.shape, (5, 5))
        torch.testing.assert_close(solver.sa_weights()[3:], torch.ones(2, 5))
        metrics = MetricsCollector.__new__(MetricsCollector)
        metrics.loss_computer = trainer.loss_computer
        metrics.loss_computer.integral_condition = False
        metrics.data_handler = trainer.data_handler
        metrics.history = {'loss': [1., 10.], 'pde': {'cont': [2., 1.]}, 'bc': {'inlet': [3., 2.]}}
        metrics._ensure_history_schema()
        self.assertEqual(metrics.history['convergence_loss'], [5., 3.])

    def test_optimizer_uses_equal_learning_rates(self) -> None:
        trainer = _trainer()
        solver = PINNSolver.__new__(PINNSolver)
        solver.model, solver.sa_weights = trainer.model, trainer.sa_weights
        solver.config = {'optimizer_params': {'lr': 0.007}, 'sa_lr': 0.1}
        optimizer = solver._init_optimizer()
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [0.007, 0.007])


class SAResumeTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        config = copy.deepcopy(ConfigManager().config)
        config['architecture']['hidden_layers'] = [8, 8]
        config['architecture']['activation'] = 'tanh'
        config['training']['batch_size'] = 32
        config['training']['early_stopping']['enabled'] = True
        config['process']['device'] = 'cpu'
        config['data']['mesh_file'] = None
        config['loss_weights']['use_sa_weights'] = True
        config['loss_weights']['sa_apply_to_bc'] = True
        config['loss_weights']['fvm']['enabled'] = False
        config['loss_weights']['fvm']['weight'] = 0.0
        config['checkpoint']['save_flow_images'] = False
        config['output']['progress_bar'] = False
        config['enable_key_listener'] = False
        return config

    def test_checkpoint_after_rar_continues_same_adam_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            solver = PINNSolver(self._config(), experiment_dir=str(Path(directory, 'first')))
            batch = next(iter(solver.data_handler.get_dataloader()))
            baseline = solver.trainer.forward_pass(batch)
            with torch.no_grad():
                solver.sa_weights.log_weights.add_(1.0)
            reweighted = solver.trainer.forward_pass(batch)
            self.assertGreater(reweighted['total'].item(), baseline['total'].item())
            self.assertEqual(reweighted['pde_raw'], baseline['pde_raw'])
            for name in baseline['bc_comp']:
                self.assertEqual(reweighted['bc_comp'][name], baseline['bc_comp'][name])
            solver.trainer.train_batch_adam(batch, accum_step=True)
            # Exercise real RAR insertion, not only attention resizing.
            solver.trainer.rar_enable = True
            solver.trainer.rar_num_points = 32
            added = solver.trainer.apply_rar()
            self.assertGreater(added, 0)
            solver._update_early_stopping(1, 3.0)
            solver._save_checkpoint(1)
            checkpoint = Path(directory, 'first', 'checkpoints', 'pinn_1.pt')
            config = self._config()
            config['process']['load_model_path'] = str(checkpoint)
            resumed = PINNSolver(config, experiment_dir=str(Path(directory, 'resumed')))
            self.assertEqual(resumed._early_best_loss, 3.0)
            self.assertEqual(resumed._early_best_epoch, 1)
            self.assertEqual(resumed.data_handler.n_points, solver.data_handler.n_points)
            self.assertIs(resumed.interactive.sa_weights, resumed.sa_weights)
            torch.testing.assert_close(resumed.sa_weights.log_weights, solver.sa_weights.log_weights)
            # Compare a deterministic continuation including the new RAR rows.
            dataset = solver.data_handler.dataset
            batch = collate_pinn_batches([
                dataset[index]
                for index in list(range(8)) + list(range(len(dataset) - added, len(dataset)))
            ])
            solver.trainer.train_batch_adam(batch, accum_step=True)
            resumed.trainer.train_batch_adam(batch, accum_step=True)
            for left, right in zip(solver.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(solver.sa_weights.log_weights, resumed.sa_weights.log_weights, rtol=0, atol=0)
            for left, right in zip(solver.optimizer.state.values(), resumed.optimizer.state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_training_loop_monitors_physics_when_weighted_loss_grows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config()
            config['checkpoint']['interval'] = 0
            solver = PINNSolver(config, experiment_dir=directory)
            solver.trainer.rar_enable = False
            solver.trainer.reset_interval = 0
            solver.early_stopping_patience = 1
            solver.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                solver.optimizer, patience=0,
            )
            initial_lr = solver.optimizer.param_groups[0]['lr']
            results = []
            for weighted, raw in [(1.0, 2.0), (10.0, 1.0), (20.0, 0.5)]:
                pde = {name: raw for name in solver.loss_computer.pde_components}
                bc = {name: 0.0 for name in solver.data_handler.boundary_type_names}
                results.append((weighted, pde, bc, 0., 0., 0., pde, pde, 0.))
            mass = {'mass_balance_error': 0., 'mass_flux_in': 0., 'mass_flux_out': 0.}
            with ExitStack() as stack:
                stack.callback(solver.metrics.close)
                stack.enter_context(patch.object(solver.trainer, 'train_epoch_adam', side_effect=results))
                stack.enter_context(patch.object(solver.metrics, 'get_boundary_avg_velocities', return_value=(0., 0.)))
                stack.enter_context(patch.object(solver.metrics, 'get_mass_flow_metrics', return_value=mass))
                for name in ('_save_checkpoint', '_save_terminal_artifacts', '_finalize'):
                    stack.enter_context(patch.object(solver, name))
                solver.train(epochs=3)
            self.assertEqual(solver.metrics.history['loss'], [1., 10., 20.])
            self.assertEqual(solver.metrics.history['convergence_loss'], [6., 3., 1.5])
            self.assertEqual(solver._early_best_epoch, 3)
            self.assertEqual(solver.scheduler.best, 1.5)
            self.assertEqual(solver.optimizer.param_groups[0]['lr'], initial_lr)
            with patch.object(solver.metrics, '_compute_mass_flow_metrics', return_value=mass):
                final = solver.metrics.compute_final_metrics()
            self.assertEqual(final['best_epoch'], 3)
            self.assertEqual(final['best_loss'], 1.5)


if __name__ == '__main__':
    unittest.main()
