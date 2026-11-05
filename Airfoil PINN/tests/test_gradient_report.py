"""Gradient reporting must observe, never change, optimizer behavior."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

import torch
from test_full_epoch_optimizer_step import FullEpochOptimizerStepTests
from airfoil_pinn.training.gradient_report import collect_gradients, print_gradient_report
from airfoil_pinn.training.menus.results import diagnostics_menu


class GradientReportTests(unittest.TestCase):
    def test_report_preserves_update_and_captures_clipping(self):
        fixture = FullEpochOptimizerStepTests()
        for full_epoch in (True, False):
            plain, plain_model = fixture._trainer(full_epoch=full_epoch)
            detailed, detailed_model = fixture._trainer(full_epoch=full_epoch)
            plain.grad_clip = detailed.grad_clip = 0.5
            detailed.diag_config = {'gradient_report_interval': 2}
            detailed.begin_gradient_report(2)
            plain.train_epoch_adam()
            detailed.train_epoch_adam()
            self.assertTrue(torch.equal(plain_model.value, detailed_model.value))
            before, after, _, _ = detailed.gradient_report
            self.assertGreater(before[0]['norm'], after[0]['norm'])
            self.assertLessEqual(after[0]['norm'], 0.5)
            out = io.StringIO()
            with redirect_stdout(out):
                print_gradient_report(detailed.gradient_report, 2)
            self.assertIn('value', out.getvalue())
            self.assertIn('clipping applied=yes', out.getvalue())
            detailed.begin_gradient_report(3)
            self.assertFalse(detailed._capture_gradient_report)
            self.assertIsNone(detailed.gradient_report)
            detailed.diag_config['gradient_report_interval'] = 0
            detailed.begin_gradient_report(4)
            self.assertFalse(detailed._capture_gradient_report)

    def test_states_and_norms(self):
        active = torch.nn.Parameter(torch.zeros(2))
        active.grad = torch.tensor([3., 4.])
        zero = torch.nn.Parameter(torch.zeros(1))
        zero.grad = torch.zeros(1)
        missing = torch.nn.Parameter(torch.zeros(1))
        frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        rows = collect_gradients(zip(('weight', 'bias', 'unused', 'fixed'), (active, zero, missing, frozen)))
        self.assertEqual(rows[0]['norm'], 5.)
        self.assertEqual(rows[0]['mean'], 3.5)
        self.assertEqual([r['status'] for r in rows], ['active', 'zero', 'None (no gradient)', 'frozen'])
        self.assertTrue(torch.equal(active.grad, torch.tensor([3., 4.])))

    def test_menu_interval_and_invalid_input(self):
        cfg = SimpleNamespace(config={'diagnostics': {'gradient_report_interval': 50}})
        with patch('builtins.input', side_effect=['15', '-2', '', '15', '3', '0']), patch('airfoil_pinn.training.menus.results.clear_screen'), redirect_stdout(io.StringIO()):
            diagnostics_menu(cfg)
        self.assertEqual(cfg.config['diagnostics']['gradient_report_interval'], 3)
