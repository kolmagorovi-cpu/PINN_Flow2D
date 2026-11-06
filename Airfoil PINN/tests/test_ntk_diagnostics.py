"""Analytical residual kernels, precision, sampling and training isolation."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from airfoil_pinn.physics.fvm import FVMResiduals
from airfoil_pinn.physics.pde import PDEResiduals
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.ntk import (
    _fvm_jacobian, boundary_residuals,
    format_ntk_diagnostics, kernel_spectrum, residual_jacobian,
)
from airfoil_pinn.training.telemetry import RunTelemetry
from parametric_physics_smoke import _grad


def _fvm_trainer(mesh, model, cases=(0, 1)):
    trainer = Trainer.__new__(Trainer)
    trainer.device, trainer.dtype = torch.device('cpu'), torch.float64
    trainer.model = model.double()
    trainer.loss_computer = SimpleNamespace(
        fvm=FVMResiduals(mesh).to('cpu', torch.float64), fvm_weight=1., bc_type='soft',
        build_input_tensor=lambda x, y, sdf, case_ids: torch.cat((x, y, case_ids.to(x).reshape(-1, 1)), dim=1),
        get_freestream_targets=lambda x, case_ids: (torch.ones_like(x), torch.zeros_like(x)),
    )
    xy = torch.tensor(np.concatenate((mesh.int_quad_coords, mesh.bnd_quad_coords)), dtype=torch.float64)
    trainer.data_handler = SimpleNamespace(
        flow_cases=dict.fromkeys(cases), fvm_quad_x=xy[:, :1], fvm_quad_y=xy[:, 1:],
        fvm_quad_sdf=torch.zeros_like(xy[:, :1]), fvm_quad_bt=None,
        fvm_int_quad_count=len(mesh.int_quad_coords),
    )
    trainer._fvm_quad_tensor_cache = None
    return trainer


class SpectrumTests(unittest.TestCase):
    def test_relative_cutoffs_are_scale_invariant_and_resolve_small_modes(self):
        jacobian = torch.diag(torch.tensor([1., 1e-9, 1e-13, 0.], dtype=torch.float64).sqrt())
        first = kernel_spectrum(jacobian)
        second = kernel_spectrum(jacobian * 1e-9)
        self.assertEqual(first['numerical_rank'], 3)
        self.assertIsNone(first['full_condition'])
        self.assertEqual([r['rank'] for r in first['cutoffs']], [2, 2, 1, 1])
        for a, b in zip(first['cutoffs'], second['cutoffs']):
            self.assertEqual(a['rank'], b['rank'])
            self.assertAlmostEqual(a['condition'] / b['condition'], 1.)
        self.assertAlmostEqual(first['effective_rank'], second['effective_rank'])

    def test_effective_rank_and_empty_zero_deficient_kernels(self):
        identity = kernel_spectrum(torch.eye(3, dtype=torch.float64))
        self.assertAlmostEqual(identity['effective_rank'], 3.)
        self.assertEqual(identity['full_condition'], 1.)
        duplicate = kernel_spectrum(torch.ones((2, 1), dtype=torch.float64))
        self.assertAlmostEqual(duplicate['effective_rank'], 1.)
        self.assertEqual(duplicate['numerical_rank'], 1)
        self.assertIsNone(duplicate['full_condition'])
        for shape in ((0, 3), (3, 0), (3, 2)):
            report = kernel_spectrum(torch.zeros(shape, dtype=torch.float64))
            self.assertEqual(report['effective_rank'], 0.)
            self.assertEqual(report['numerical_rank'], 0)
            json.dumps(report, allow_nan=False)

    def test_float32_and_nonfinite_are_rejected(self):
        with self.assertRaises(TypeError):
            kernel_spectrum(torch.eye(2))
        with self.assertRaises(ValueError):
            kernel_spectrum(torch.tensor([[float('nan')]], dtype=torch.float64))

    def test_scalar_rows_do_not_cancel_or_write_grad_buffers(self):
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float64), requires_grad=False)
        p.grad = torch.tensor([7.], dtype=torch.float64)
        residuals = torch.cat((p, -p, p.new_zeros(1)))
        jacobian = residual_jacobian(residuals, (p, unused, frozen))
        torch.testing.assert_close(jacobian, torch.tensor([[1., 0.], [-1., 0.], [0., 0.]], dtype=torch.float64))
        self.assertEqual(p.grad.item(), 7.)
        self.assertIsNone(unused.grad)
        self.assertEqual(kernel_spectrum(jacobian)['numerical_rank'], 1)


class ResidualOperatorTests(unittest.TestCase):
    def test_signed_pde_matches_training_squares_for_all_systems(self):
        x = torch.tensor([[0.3], [0.7]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.2], [0.8]], dtype=torch.float64, requires_grad=True)
        u, v, p = x.square() + y, x * y, x - y.square()
        for system in ('2d', 'euler_2d', 'axisymmetric', 'euler_axisymmetric'):
            with self.subTest(system=system):
                method = getattr(PDEResiduals, 'compute_residuals_' + system)
                args = [u, v, p, x, y]
                if not system.startswith('euler'):
                    args.append(torch.full_like(x, 0.1))
                args.append(_grad)
                signed, squared = method(*args, squared=False), method(*args)
                for name in signed:
                    torch.testing.assert_close(signed[name].square(), squared[name])

    def test_zero_pde_residual_keeps_nonzero_operator_jacobian(self):
        parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        x = torch.ones((1, 1), dtype=torch.float64, requires_grad=True)
        y = torch.ones_like(x, requires_grad=True)
        zero = 0. * x + 0. * y
        residuals = PDEResiduals.compute_residuals_euler_2d(
            parameter * x, zero, zero, x, y, _grad, squared=False,
        )
        self.assertEqual(residuals['cont'].item(), 0.)
        self.assertEqual(residual_jacobian(residuals['cont'], [parameter]).item(), 1.)

    def test_bc_signed_components_match_training_operators(self):
        from boundary_normal_bc_smoke import _loss_computer

        x = torch.tensor([[0.3], [0.7]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.2], [0.8]], dtype=torch.float64, requires_grad=True)
        u, v, p = x.square() + y, x * y, x - y.square()
        bt = torch.ones_like(x)
        nx, ny = torch.full_like(x, 0.6), torch.full_like(x, 0.8)
        for name in ('inlet', 'farfield', 'velocity_dirichlet', 'outlet',
                     'wall', 'slip_wall', 'symmetry', 'axis_symmetry'):
            with self.subTest(boundary=name):
                lc = _loss_computer(name)
                for weight in ('outlet_grad', 'symmetry_grad', 'axis_symmetry_grad'):
                    lc.loss_weights[weight] = 1.
                signed = boundary_residuals(lc, u, v, p, bt, x, y, None, nx, ny)
                losses, extra = lc.compute_bc_loss(u, v, p, bt, x, y, normal_x=nx, normal_y=ny)
                torch.testing.assert_close(signed.square().sum() / len(x), sum(losses.values()) + extra)

    def test_fvm_sampled_jacobian_equals_full_mesh_with_multiple_cases(self):
        from test_fvm_objective import _structured_cell_mesh

        mesh, _ = _structured_cell_mesh(3)
        trainer = _fvm_trainer(
            mesh, torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)),
        )
        fvm = trainer.loss_computer.fvm
        x, y, sdf, bt, q = trainer._get_fvm_quadrature_tensors()
        rows = []
        for _, out, stream in trainer._forward_fvm_cases(x, y, sdf, bt):
            rows.append(fvm.compute_residual_quadrature(out[:q, :2], out[q:, :2], stream)[[0, 8]]
                        * fvm.V[[0, 8]].sqrt())
        expected = residual_jacobian(torch.cat(rows), trainer.model.parameters())
        trainer.loss_computer.fvm = copy.copy(fvm)
        actual = _fvm_jacobian(trainer, 2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape[0], 4)
        self.assertEqual(fvm.Q, q)

    def test_fvm_unequal_cell_areas_weight_jacobian_and_spectrum(self):
        # Two adjacent rectangles: [0, 1] x [0, 2] and [1, 4] x [0, 2].
        mesh = SimpleNamespace(
            cv_volumes=np.array([2., 6.]),
            int_face_node_i=np.array([0]), int_face_node_j=np.array([1]),
            int_face_area=np.array([[2., 0.]]),
            int_quad_node_i=np.array([0]), int_quad_node_j=np.array([1]),
            int_quad_area=np.array([[2., 0.]]), int_quad_coords=np.array([[1., 1.]]),
            bnd_face_node=np.array([0, 0, 0, 1, 1, 1]),
            bnd_face_area=np.array([[-2., 0.], [0., -1.], [0., 1.],
                                    [2., 0.], [0., -3.], [0., 3.]]),
            bnd_quad_coords=np.array([[0., 1.], [0.5, 0.], [0.5, 2.],
                                      [4., 1.], [2.5, 0.], [2.5, 2.]]),
            bnd_face_type=np.full(6, 'outlet'),
        )

        class QuadraticVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coefficients = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

            def forward(self, inputs):
                x = inputs[:, :1]
                u = self.coefficients[0] * x + self.coefficients[1] * x.square()
                return torch.cat((u, torch.zeros_like(u), torch.zeros_like(u)), dim=1)

        # R = [a + b, a + 5b]; zero residuals still have nonzero Jacobians.
        expected_jacobian = torch.tensor([[1., 1.], [1., 5.]], dtype=torch.float64)
        expected_jacobian *= torch.tensor([[2.], [6.]], dtype=torch.float64).sqrt()
        expected_kernel = torch.tensor([[4., 6. * np.sqrt(12.)],
                                        [6. * np.sqrt(12.), 156.]], dtype=torch.float64)
        for limit in (1, 2):
            with self.subTest(limit=limit):
                trainer = _fvm_trainer(mesh, QuadraticVelocity(), cases=(0,))
                jacobian = _fvm_jacobian(trainer, limit)
                torch.testing.assert_close(jacobian, expected_jacobian[:limit])
                torch.testing.assert_close(jacobian @ jacobian.T, expected_kernel[:limit, :limit])
                spectrum = kernel_spectrum(jacobian)
                torch.testing.assert_close(
                    torch.tensor(spectrum['eigenvalues'], dtype=torch.float64),
                    torch.linalg.eigvalsh(expected_kernel[:limit, :limit]),
                )
                self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))


class IntegrationTests(unittest.TestCase):
    def test_real_pinn_float64_isolation_failure_and_json_history(self):
        from end_to_end_pipeline_smoke import _small_solver_config
        from airfoil_pinn.training.solver import PINNSolver

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            solver = PINNSolver(_small_solver_config(), experiment_dir=directory)
            trainer = solver.trainer
            batch = next(iter(solver.data_handler.get_dataloader()))
            for parameter in trainer.model.parameters():
                parameter.grad = torch.ones_like(parameter)
            before = copy.deepcopy(trainer.model.state_dict())
            gradients = [p.grad.clone() for p in trainer.model.parameters()]
            rng = torch.random.get_rng_state().clone()
            mode = trainer.model.training
            dtype = next(trainer.model.parameters()).dtype
            original_jacobian = residual_jacobian

            def checked(residuals, parameters):
                self.assertEqual(residuals.dtype, torch.float64)
                return original_jacobian(residuals, parameters)

            with patch('airfoil_pinn.training.ntk.residual_jacobian', side_effect=checked):
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    report = trainer.compute_ntk_diagnostics(batch, max_samples=3)
            self.assertEqual(set(report['blocks']), {'pde', 'bc', 'fvm'})
            self.assertGreater(report['blocks']['pde']['rows'], 0)
            self.assertGreater(report['blocks']['bc']['rows'], 0)
            self.assertEqual(report['blocks']['fvm']['status'], 'unavailable: no mesh')
            for name, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
            for parameter, gradient in zip(trainer.model.parameters(), gradients):
                self.assertTrue(torch.equal(parameter.grad, gradient))
                self.assertEqual(parameter.dtype, dtype)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertEqual(trainer.model.training, mode)

            with patch('airfoil_pinn.training.ntk._point_jacobians', side_effect=RuntimeError('test')):
                with self.assertRaisesRegex(RuntimeError, 'test'):
                    trainer.compute_ntk_diagnostics(batch, 2)
            self.assertEqual(trainer.model.training, mode)
            self.assertEqual(next(trainer.model.parameters()).dtype, dtype)

            report['epoch'] = 7
            # Empty FVM conditions must serialize as JSON null.
            RunTelemetry(directory).append_epoch({'diagnostics': {'ntk_diagnostics': report}})
            text = format_ntk_diagnostics(report, 7)
            self.assertIn('effective_rank=', text)
            self.assertIn('rtol=1.0e-12', text)
            solver.metrics.history.pop('ntk_diagnostics', None)
            solver.metrics._ensure_history_schema()
            solver.metrics.update_history(0., {}, {}, 0., 0., {}, {}, 0.,
                                          extra_diag={'ntk_diagnostics': report})
            self.assertEqual(solver.metrics.history['ntk_diagnostics'][0]['epoch'], 7)

            # Exercise the complete diagnostic copy with float32 FVM geometry.
            from test_fvm_objective import _structured_cell_mesh

            mesh, _ = _structured_cell_mesh(3)
            fvm = FVMResiduals(mesh).to('cpu', dtype)
            trainer.loss_computer.fvm = fvm
            trainer.loss_computer.fvm_weight = 1.
            xy = torch.tensor(np.concatenate((mesh.int_quad_coords, mesh.bnd_quad_coords)), dtype=dtype)
            dh = trainer.data_handler
            dh.fvm_quad_x, dh.fvm_quad_y = xy[:, :1], xy[:, 1:]
            dh.fvm_quad_sdf = trainer.loss_computer.evaluate_sdf_tensor(dh.fvm_quad_x, dh.fvm_quad_y)
            dh.fvm_quad_bt = None
            dh.fvm_int_quad_count = len(mesh.int_quad_coords)
            volumes, areas = fvm.V.clone(), fvm.quad_S.clone()
            with patch('airfoil_pinn.training.ntk.residual_jacobian', side_effect=checked):
                fvm_report = trainer.compute_ntk_diagnostics(batch, max_samples=2)
            block = fvm_report['blocks']['fvm']
            self.assertEqual(block['rows'], 2)
            self.assertGreater(block['lambda_max'], 0.)
            self.assertEqual(block['residual_weighting'], 'sqrt_control_volume')
            self.assertEqual(block['volume_normalization'], 'none')
            self.assertIn('sqrt(V_i) * divergence_i', format_ntk_diagnostics(fvm_report, 7))
            json.dumps(fvm_report, allow_nan=False)
            self.assertEqual(fvm.V.dtype, dtype)
            self.assertTrue(torch.equal(fvm.V, volumes))
            self.assertTrue(torch.equal(fvm.quad_S, areas))
            self.assertEqual(fvm.Q, dh.fvm_int_quad_count)

            trainer.loss_computer.fvm_weight = 0.
            disabled = trainer.compute_ntk_diagnostics(batch, max_samples=1)
            self.assertEqual(disabled['blocks']['fvm']['status'], 'disabled')

    def test_invalid_sample_sizes_fail_before_copying(self):
        trainer = Trainer.__new__(Trainer)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                trainer.compute_ntk_diagnostics((), value)

    def test_sample_size_configuration_validation(self):
        from airfoil_pinn.config.manager import ConfigManager
        from airfoil_pinn.config.validation import ConfigurationError, validate_static_config

        config = ConfigManager().config
        self.assertEqual(config['diagnostics']['ntk_sample_size'], 64)
        for value in (0, -1, True, 1.5):
            config['diagnostics']['ntk_sample_size'] = value
            with self.assertRaisesRegex(ConfigurationError, 'ntk_sample_size'):
                validate_static_config(config)


if __name__ == '__main__':
    unittest.main()
