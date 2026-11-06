"""Gradient diagnostics must preserve training and measure weighted vectors."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

import test_full_epoch_optimizer_step as fixture
from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import ConfigurationError, validate_static_config
from airfoil_pinn.training.components.metrics import MetricsCollector
from airfoil_pinn.training.gradient_contributions import (
    COSINE_PAIRS, GradientContributions, format_gradient_contributions,
)
from airfoil_pinn.training.menus.results import diagnostics_menu
from airfoil_pinn.training.telemetry import RunTelemetry


def make_trainer(enabled=True, full_epoch=True):
    trainer, model = fixture.FullEpochOptimizerStepTests()._trainer(full_epoch=full_epoch)
    trainer.loss_computer.fvm = object()
    trainer.loss_computer.fvm_weight = 3.0
    trainer.loss_computer.integral_condition = True
    trainer.loss_computer.integral_weight = 5.0
    trainer.fvm_schedule = 'every_optimizer_step'
    trainer.fvm_batch_interval = 1
    trainer.diag_config = {
        'gradient_report_interval': 0,
        'gradient_contributions_enabled': enabled,
        'compute_fvm_pde_cosine': enabled,
        'gradient_contributions_interval': 2,
    }

    def forward(self, target, pde_mult=1.0, **kw):
        value = model.value
        zero = torch.zeros(())
        losses = {
            'pde': pde_mult * (value - target).square(),
            'bc': 0.5 * (value + 2).square(),
            'fvm': 3 * (value - 4).square() * kw['fvm_scale'] if kw['add_fvm'] else zero,
            'integral': 5 * (value + 1).square() * kw['integral_scale'] if kw['add_integral'] else zero,
        }
        result = fixture._batch_result(sum(losses.values()))
        result['weighted_loss_components'] = losses
        return result

    trainer.forward_pass = MethodType(forward, trainer)
    trainer.begin_gradient_report(2)
    return trainer, model


class GradientContributionTests(unittest.TestCase):
    def test_real_pinn_forward_gradient_sum_matches_backward(self):
        from end_to_end_pipeline_smoke import _small_solver_config
        from airfoil_pinn.training.solver import PINNSolver

        config = _small_solver_config()
        config['boundary_conditions']['type'] = 'soft'
        config['output']['console_only'] = True
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            solver = PINNSolver(config, experiment_dir=directory)
            batch = next(iter(solver.data_handler.get_dataloader()))
            forward = solver.trainer.forward_pass(batch, pde_mult=3.0)
            losses = forward['weighted_loss_components']
            torch.testing.assert_close(sum(losses.values()), forward['total'])
            collector = GradientContributions(solver.model.parameters())
            collector.accumulate(losses)
            forward['total'].backward()
            report = collector.finish_step(stage='real PINN')
            actual = sum(
                float(p.grad.double().square().sum())
                for p in solver.model.parameters() if p.grad is not None
            ) ** 0.5
            self.assertAlmostEqual(report['total_l2'] / actual, 1.0, places=5)
            self.assertGreater(report['components']['pde']['l2'], 0.)
            self.assertEqual(report['components']['fvm']['l2'], 0.)

    def test_accumulated_weighted_norms_and_update_invariance(self):
        plain, plain_model = make_trainer(False)
        measured, measured_model = make_trainer()
        plain.grad_clip = measured.grad_clip = 0.5
        plain.train_epoch_adam()
        measured.train_epoch_adam()
        self.assertTrue(torch.equal(plain_model.value, measured_model.value))
        report = measured.gradient_contributions_report
        self.assertEqual(report['evaluations'], 1)
        for name, norm in dict(pde=4.0, bc=2.0, fvm=24.0, integral=10.0).items():
            self.assertAlmostEqual(report['components'][name]['l2'], norm)
        self.assertAlmostEqual(report['total_l2'], 16.0)
        self.assertAlmostEqual(report['components']['fvm']['share_percent'], 60.0)
        self.assertAlmostEqual(report['cosine_fvm_pde'], 1.0)

    def test_partial_accumulation_window_and_multiple_steps(self):
        plain, plain_model = make_trainer(False, False)
        measured, measured_model = make_trainer(True, False)
        for trainer in (plain, measured):
            trainer.accum_steps = 2
            trainer.data_handler.get_dataloader = lambda: [torch.tensor(1.0)] * 3
            trainer.train_epoch_adam()
        self.assertTrue(torch.equal(plain_model.value, measured_model.value))
        self.assertEqual(measured.gradient_contributions_report['evaluations'], 2)

    def test_lbfgs_last_closure_and_update_invariance(self):
        plain, plain_model = make_trainer(False)
        measured, measured_model = make_trainer()
        for trainer in (plain, measured):
            trainer.optimizer = torch.optim.LBFGS(
                trainer.model.parameters(), lr=0.2, max_iter=3,
            )
            trainer.train_epoch_lbfgs()
        self.assertTrue(torch.equal(plain_model.value, measured_model.value))
        report = measured.gradient_contributions_report
        self.assertEqual(report['evaluations'], 1)
        self.assertIn('last closure', report['stage'])
        self.assertAlmostEqual(report['total_l2'], measured_model.value.grad.norm().item(), places=5)

    def test_interval_and_disabled_skip_autograd_work(self):
        trainer, _ = make_trainer()
        trainer.begin_gradient_report(3)
        self.assertIsNone(trainer.gradient_contributions_report)
        with patch('torch.autograd.grad', side_effect=AssertionError('unexpected diagnostic')):
            trainer.train_epoch_adam()
        trainer.diag_config['gradient_contributions_enabled'] = False
        trainer.diag_config['compute_fvm_pde_cosine'] = False
        trainer.begin_gradient_report(4)
        self.assertIsNone(trainer._gradient_contributions)

    def test_cancellation_amp_scale_unused_and_frozen_parameters(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        unused = torch.nn.Parameter(torch.zeros(1))
        frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        parameter.grad = torch.tensor([7., 8.])
        collector = GradientContributions([parameter, unused, frozen])
        pde = (parameter * torch.tensor([3., 4.])).sum()
        losses = dict(pde=pde, bc=-pde, fvm=torch.tensor(0.), integral=torch.tensor(0.))
        collector.accumulate(losses, amp_scale=1024.)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([7., 8.])))
        self.assertIsNone(unused.grad)
        report = collector.finish_step(stage='test')
        self.assertEqual(report['components']['pde']['l2'], 5.)
        self.assertEqual(report['components']['bc']['share_percent'], 50.)
        self.assertEqual(report['components']['fvm']['l2'], 0.)
        self.assertEqual(report['total_l2'], 0.)
        sum(losses.values()).backward()  # Original graph is still usable.

    def test_vector_accumulation_precedes_norm(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        collector = GradientContributions([parameter])
        for sign in (1., -1.):
            losses = dict.fromkeys(('bc', 'fvm', 'integral'), torch.tensor(0.))
            losses['pde'] = sign * parameter.sum()
            collector.accumulate(losses, divisor=2)
        report = collector.finish_step(stage='test')
        self.assertEqual(report['components']['pde']['l2'], 0.)
        self.assertEqual(report['components']['pde']['share_percent'], 0.)

    def test_console_text_and_log_serialization(self):
        trainer, _ = make_trainer()
        trainer.train_epoch_adam()
        report = trainer.gradient_contributions_report
        text = format_gradient_contributions(report, 2)
        self.assertIn('cosine(FVM, PDE)=1.000000', text)
        self.assertIn('cosine(FVM, BC)=-1.000000', text)
        self.assertIn('cosine(PDE, BC)=-1.000000', text)
        for name in ('PDE', 'BC', 'FVM', 'INTEGRAL'):
            self.assertIn(name, text)
        with tempfile.TemporaryDirectory() as directory:
            metrics = MetricsCollector.__new__(MetricsCollector)
            metrics.log_file = None
            metrics.logs_dir = directory
            metrics.config_name = 'test'
            try:
                metrics.save_log(text)
                log_path = next(Path(directory).glob('*.log'))
                self.assertEqual(log_path.read_text(encoding='utf-8'), text + '\n')
            finally:
                metrics.log_file.close()
            telemetry = RunTelemetry(directory)
            telemetry.append_epoch({'epoch': 2, 'diagnostics': {'gradient_contributions': report}})
            stored = json.loads(telemetry.epochs_path.read_text(encoding='utf-8'))
            self.assertEqual(stored['diagnostics']['gradient_contributions'], report)

    def test_pairwise_cosine_directions_and_undefined(self):
        cases = (
            ([2., 0.], 1.), ([-2., 0.], -1.), ([0., 2.], 0.),
            ([3., 4.], 0.6), ([0., 0.], None), ([float('inf'), 0.], None),
        )
        for left, right in COSINE_PAIRS:
            for direction, expected in cases:
                with self.subTest(pair=(left, right), direction=direction):
                    parameter = torch.nn.Parameter(torch.ones(2))
                    collector = GradientContributions([parameter])
                    losses = dict.fromkeys(('pde', 'bc', 'fvm', 'integral'), torch.tensor(0.))
                    losses[left] = parameter[0]
                    losses[right] = (parameter * torch.tensor(direction)).sum()
                    collector.accumulate(losses, amp_scale=1024.)
                    report = collector.finish_step(stage='test')
                    key = f'cosine_{left}_{right}'
                    if expected is None:
                        self.assertIsNone(report[key])
                        self.assertIn(
                            f'cosine({left.upper()}, {right.upper()})=n/a',
                            format_gradient_contributions(report, 1),
                        )
                    else:
                        self.assertAlmostEqual(report[key], expected)
                    self.assertIsNone(parameter.grad)

    def test_cosine_pair_counts_are_independent(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        collector = GradientContributions([parameter])
        for fvm, bc in ((parameter[0], parameter[1]), (torch.tensor(0.), -parameter[0])):
            collector.accumulate(dict(pde=parameter[0], fvm=fvm, bc=bc, integral=torch.tensor(0.)))
            report = collector.finish_step(stage='test')
        for key, expected, count in (
            ('cosine_fvm_pde', 1., 1), ('cosine_fvm_bc', 0., 1), ('cosine_pde_bc', -0.5, 2),
        ):
            self.assertAlmostEqual(report[key], expected)
            self.assertEqual(report[f'{key}_evaluations'], count)
        collector.reset()
        report = collector.finish_step(stage='test')
        for left, right in COSINE_PAIRS:
            self.assertIsNone(report[f'cosine_{left}_{right}'])
            self.assertEqual(report[f'cosine_{left}_{right}_evaluations'], 0)

    def test_cosine_uses_accumulated_vectors_and_skips_missing_steps(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        collector = GradientContributions([parameter])
        for direction in ([1., 1.], [1., -1.]):
            collector.accumulate(dict(
                pde=(parameter * torch.tensor(direction)).sum(), fvm=parameter[0],
                bc=torch.tensor(0.), integral=torch.tensor(0.),
            ), divisor=2)
        report = collector.finish_step(stage='test')
        self.assertAlmostEqual(report['cosine_fvm_pde'], 1.)
        collector.accumulate(dict(
            pde=parameter[0], fvm=-parameter[0],
            bc=torch.tensor(0.), integral=torch.tensor(0.),
        ))
        collector.finish_step(stage='test')
        collector.accumulate(dict(
            pde=parameter[0], fvm=torch.tensor(0.),
            bc=torch.tensor(0.), integral=torch.tensor(0.),
        ))
        report = collector.finish_step(stage='test')
        self.assertEqual(report['cosine_fvm_pde'], 0.)
        self.assertEqual(report['cosine_fvm_pde_evaluations'], 2)
        collector.reset()
        self.assertIsNone(collector.finish_step(stage='test')['cosine_fvm_pde'])

    def test_configuration_validation(self):
        config = ConfigManager().config
        validate_static_config(config)
        for key, values in {
            'gradient_contributions_interval': (0, -1, True, 1.5, '2'),
            'gradient_contributions_enabled': ('true', 1, None),
            'compute_fvm_pde_cosine': ('true', 1, None),
        }.items():
            original = config['diagnostics'][key]
            for value in values:
                config['diagnostics'][key] = value
                with self.assertRaisesRegex(ConfigurationError, key):
                    validate_static_config(config)
            config['diagnostics'][key] = original

    def test_menu_enable_interval_invalid_and_disable(self):
        cfg = SimpleNamespace(config={'diagnostics': {}})
        answers = ['16', 'y', '17', '0', '', '17', '3', '0']
        with patch('builtins.input', side_effect=answers), patch(
            'airfoil_pinn.training.menus.results.clear_screen'
        ), redirect_stdout(io.StringIO()):
            diagnostics_menu(cfg)
        self.assertTrue(cfg.config['diagnostics']['gradient_contributions_enabled'])
        self.assertEqual(cfg.config['diagnostics']['gradient_contributions_interval'], 3)
        with patch('builtins.input', side_effect=['16', 'n', '0']), patch(
            'airfoil_pinn.training.menus.results.clear_screen'
        ), redirect_stdout(io.StringIO()):
            diagnostics_menu(cfg)
        self.assertFalse(cfg.config['diagnostics']['gradient_contributions_enabled'])

    def test_menu_cosine_toggle_and_independent_output(self):
        trainer, _ = make_trainer(False)
        cfg = SimpleNamespace(config={'diagnostics': trainer.diag_config})
        for enabled, answer in ((True, 'y'), (False, 'n')):
            with patch('builtins.input', side_effect=['18', answer, '0']), patch(
                'airfoil_pinn.training.menus.results.clear_screen'
            ), redirect_stdout(io.StringIO()) as output:
                diagnostics_menu(cfg)
            self.assertIn('18. Вычисление и вывод cosine(FVM, PDE)', output.getvalue())
            self.assertEqual(trainer.diag_config['compute_fvm_pde_cosine'], enabled)
            self.assertFalse(trainer.diag_config['gradient_contributions_enabled'])
            trainer.begin_gradient_report(2)
            if enabled:
                trainer.train_epoch_adam()
                text = format_gradient_contributions(trainer.gradient_contributions_report, 2)
                self.assertIn('cosine(FVM, PDE)=1.000000', text)
                self.assertIn('cosine(FVM, BC)=-1.000000', text)
                self.assertIn('cosine(PDE, BC)=-1.000000', text)
                self.assertNotIn('share=', text)
                trainer.begin_gradient_report(3)
            self.assertIsNone(trainer._gradient_contributions)

    def test_contributions_can_disable_cosine(self):
        trainer, _ = make_trainer()
        trainer.diag_config['compute_fvm_pde_cosine'] = False
        trainer.begin_gradient_report(2)
        trainer.train_epoch_adam()
        report = trainer.gradient_contributions_report
        for left, right in COSINE_PAIRS:
            self.assertEqual(report[f'cosine_{left}_{right}_evaluations'], 0)
        text = format_gradient_contributions(report, 2)
        self.assertIn('share=', text)
        self.assertNotIn('cosine(', text)


if __name__ == '__main__':
    unittest.main()
