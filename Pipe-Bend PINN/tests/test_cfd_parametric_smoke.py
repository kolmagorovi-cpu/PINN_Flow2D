#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сквозной smoke-тест: CFD-данные + параметрическая сеть + data-loss.

Создаёт синтетический CFD-экспорт на ПРАВИЛЬНОЙ геометрии PB_R100
(pipe_bend_domain.npy), подключает его через манифест, инициализирует
``PINNSolver`` в параметрическом режиме и выполняет одну эпоху Adam.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yaml

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver
from pipe_bend_pinn_data import (
    load_domain_polyline,
    points_in_polygon,
)

DOMAIN_NPY = os.path.join(
    PROJECT_ROOT, 'workspace', 'geometry', 'domains', 'pipe_bend_domain.npy'
)
SCALES = {
    'U_ref': 1.0,
    'pipe_width': 0.5,
    'density': 998.0,
    'dynamic_viscosity': 0.998,
    'static_pressure': 100000.0,
}
CSV_HEADER = (
    '[Name]\nPlane 1\n\n[Data]\n'
    'X [ m ], Y [ m ], Z [ m ], Pressure [ Pa ], '
    'Velocity [ m s^-1 ], Velocity u [ m s^-1 ], Velocity v [ m s^-1 ]\n'
)


def _synthetic_grid(step: float = 0.08) -> np.ndarray:
    polyline = load_domain_polyline(DOMAIN_NPY)
    xs = np.arange(-1.95, 1.21, step)
    ys = np.arange(-1.20, 1.96, step)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    return grid[points_in_polygon(grid, polyline)]


def _write_case(directory: str) -> str:
    points = _synthetic_grid()
    csv_path = os.path.join(directory, 'case_U1.0.csv')
    with open(csv_path, 'w', encoding='utf-8') as handle:
        handle.write(CSV_HEADER)
        for x, y in points:
            u = 0.5 + 0.05 * np.sin(3.0 * x) * np.cos(3.0 * y)
            v = 0.1 + 0.05 * np.cos(2.0 * x) * np.sin(2.0 * y)
            p = 100000.0 + 20.0 * np.sin(x + y)
            handle.write(
                f'{x:.8e}, {y:.8e}, 0.00000000e+00, {p:.8e}, '
                f'{np.hypot(u, v):.8e}, {u:.8e}, {v:.8e}\n'
            )
    return csv_path


def _write_manifest(directory: str, csv_path: str) -> str:
    manifest = {
        'ref_scales': SCALES,
        'geometry': {'domain_file': DOMAIN_NPY, 'closed': True},
        'cases': [{'file': csv_path, 'U_in': 1.0, 'name': 'synthetic_Re500'}],
    }
    path = os.path.join(directory, 'cfd_manifest.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True)
    return path


def _config(manifest_path: str, workdir: str) -> dict:
    config = copy.deepcopy(ConfigManager().config)
    config['project_name'] = 'cfd_parametric_smoke'
    config['architecture'].update(
        hidden_layers=[16, 16],
        activation='tanh',
        residual=False,
        parametric=True,
        spatial_dim=3,
        param_dim=1,
        fourier_features={'enabled': True, 'n_frequencies': 4, 'sigma': 3.0, 'gamma': 1.0},
        coordinate_scaling={'enabled': False, 'scale_factor': 1.0},
    )
    lw = config['loss_weights']
    lw['use_sa_weights'] = False
    lw['reference_field'] = {'enabled': False, 'weight': 0.0}
    lw['pressure_poisson'] = {'enabled': False, 'weight': 0.0}
    lw['fvm']['enabled'] = False
    lw['data_loss'] = {
        'enabled': True,
        'weight': 2.0,
        'weights': {'u': 1.0, 'v': 1.0, 'p': 0.5},
    }
    config['data']['cfd'] = {
        'enabled': True,
        'manifest': manifest_path,
        'batch_size': 128,
        'strict': True,
        're_range': [0.5, 2.0],
        're_default': 1.0,
    }
    config['process']['device'] = 'cpu'
    config['process']['torch_compile'] = False
    config['output']['console_only'] = True
    config['enable_key_listener'] = False
    config['training']['epochs'] = 1
    config['training']['rar_enable'] = False
    config['training']['batch_size'] = 256
    config['checkpoint_dir'] = os.path.join(workdir, 'experiments')
    return config


class CFDParametricSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workdir = tempfile.mkdtemp(prefix='cfd_smoke_')
        csv_path = _write_case(cls.workdir)
        manifest_path = _write_manifest(cls.workdir, csv_path)
        cls.solver = PINNSolver(_config(manifest_path, cls.workdir))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_cfd_handler_loaded(self) -> None:
        self.assertIsNotNone(self.solver.cfd_handler)
        self.assertGreater(len(self.solver.cfd_handler), 0)
        self.assertAlmostEqual(self.solver.re_ref, 500.0, places=6)

    def test_parametric_model(self) -> None:
        self.assertTrue(self.solver.loss_computer.parametric)
        self.assertEqual(self.solver.model.spatial_dim, 3)
        self.assertEqual(self.solver.model.param_dim, 1)

    def test_cfd_batch(self) -> None:
        batch = next(iter(self.solver.cfd_dataloader))
        for key in ('x', 'y', 'sdf', 're_norm', 'u', 'v', 'p', 'case_id'):
            self.assertIn(key, batch)
        self.assertEqual(tuple(batch['x'].shape)[1], 1)

    def test_train_one_epoch_with_data_loss(self) -> None:
        self.solver.loss_computer.data_enabled = True
        result = self.solver.trainer.train_epoch_adam(1.0)
        loss = result[0]
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(self.solver.trainer.last_data_loss, 0.0)

    def test_data_loss_toggle_off(self) -> None:
        self.solver.loss_computer.data_enabled = False
        result = self.solver.trainer.train_epoch_adam(1.0)
        self.assertTrue(np.isfinite(result[0]))
        self.assertEqual(self.solver.trainer.last_data_loss, 0.0)
        # Возвращаем флаг, чтобы не влиять на другие тесты.
        self.solver.loss_computer.data_enabled = True


if __name__ == '__main__':
    unittest.main(verbosity=2)

