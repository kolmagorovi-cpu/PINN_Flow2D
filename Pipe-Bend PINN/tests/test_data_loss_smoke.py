#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-тест интеграции data-loss и параметрического входа PINN.

Проверяет:
- параметрическая сеть принимает 4 входа (x, y, sdf, re_norm);
- data-loss включается/отключается флагом ``loss_weights.data_loss.enabled``;
- per-point вязкость не ломает PDE-невязки;
- при выключенном data-loss результат совместим с базовым режимом.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.data.handler import DataHandler
from pipe_bend_pinn.training.components.loss_computer import LossComputer
from pipe_bend_pinn.training.models import PINNIncompressible


def _config(data_loss_enabled: bool, parametric: bool = True) -> dict:
    config = copy.deepcopy(ConfigManager().config)
    config['project_name'] = 'data_loss_smoke'
    config['architecture'].update(
        hidden_layers=[16, 16],
        activation='tanh',
        residual=False,
        parametric=parametric,
        spatial_dim=3,
        param_dim=1,
        fourier_features={'enabled': True, 'n_frequencies': 4, 'sigma': 3.0, 'gamma': 1.0},
        coordinate_scaling={'enabled': False, 'scale_factor': 1.0},
    )
    config['boundary_conditions']['type'] = 'soft'
    lw = config['loss_weights']
    lw['use_sa_weights'] = False
    lw['reference_field'] = {'enabled': False, 'weight': 0.0}
    lw['pressure_poisson'] = {'enabled': False, 'weight': 0.0}
    lw['data_loss'] = {
        'enabled': data_loss_enabled,
        'weight': 1.0,
        'weights': {'u': 1.0, 'v': 1.0, 'p': 1.0},
    }
    config['data']['cfd'] = {
        'enabled': False,
        're_range': [0.5, 2.0],
        're_default': 1.0,
    }
    return config


class DataLossIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        cls.dtype = torch.float32
        config = _config(data_loss_enabled=True)
        points_file = os.path.join(
            PROJECT_ROOT, 'workspace', 'initial', 'fields', 'initial.npy'
        )
        cls.data_handler = DataHandler(
            points_file=points_file, batch_size=64, shuffle=False, device=cls.device
        )
        cls.config = config
        cls.re_ref = float(cls.data_handler.Re)
        cls.loss_computer = LossComputer(
            config=config,
            data_handler=cls.data_handler,
            rho_norm=torch.tensor(1.0),
            mu_norm=torch.tensor(1.0 / cls.re_ref),
            x_min=cls.data_handler.x_min,
            x_max=cls.data_handler.x_max,
            y_min=cls.data_handler.y_min,
            y_max=cls.data_handler.y_max,
            device=cls.device,
            dtype=cls.dtype,
            re_ref=cls.re_ref,
        )
        cls.model = PINNIncompressible(
            input_dim=4,
            layers=[16, 16],
            activation='tanh',
            output_dim=3,
            output_bias=(0.0, 0.0, 0.0),
            coordinate_scaling={'enabled': False, 'scale_factor': 1.0},
            fourier_features={'enabled': True, 'n_frequencies': 4, 'sigma': 3.0, 'gamma': 1.0},
            residual=False,
            parametric=True,
            spatial_dim=3,
            param_dim=1,
        )
        cls.batch = next(iter(cls.data_handler.get_dataloader()))

    def _data_batch(self, n: int = 32) -> dict:
        ds = self.data_handler.dataset
        return {
            'x': ds.x[:n].clone(),
            'y': ds.y[:n].clone(),
            'sdf': ds.sdf[:n].clone(),
            're_norm': torch.full((n, 1), 1.25),
            'u': 0.5 * torch.ones(n, 1),
            'v': 0.1 * torch.ones(n, 1),
            'p': 0.05 * torch.ones(n, 1),
        }

    def test_parametric_input_dim(self) -> None:
        x = torch.zeros(5, 4)
        inp = self.loss_computer.build_input_tensor(
            x[:, 0:1], x[:, 1:2], x[:, 2:3], None, x[:, 3:4]
        )
        self.assertEqual(tuple(inp.shape), (5, 4))

    def test_parametric_forward_shape(self) -> None:
        out = self.model(torch.zeros(7, 4))
        self.assertEqual(tuple(out.shape), (7, 3))

    def test_data_loss_enabled_changes_loss(self) -> None:
        self.loss_computer.data_enabled = True
        with_data = self.loss_computer.compute_loss(
            self.batch, self.model, pde_mult=1.0, data_batch=self._data_batch()
        )
        without_data = self.loss_computer.compute_loss(
            self.batch, self.model, pde_mult=1.0, data_batch=None
        )
        self.assertTrue(bool(torch.isfinite(with_data['loss']).all()))
        self.assertGreater(with_data['data_loss'], 0.0)
        self.assertEqual(without_data['data_loss'], 0.0)
        self.assertGreater(
            float(with_data['loss'].detach()),
            float(without_data['loss'].detach()),
        )

    def test_data_loss_disabled_by_flag(self) -> None:
        self.loss_computer.data_enabled = False
        result = self.loss_computer.compute_loss(
            self.batch, self.model, pde_mult=1.0, data_batch=self._data_batch()
        )
        self.assertEqual(result['data_loss'], 0.0)
        self.assertTrue(bool(torch.isfinite(result['loss']).all()))

    def test_re_sampling_covers_range(self) -> None:
        x = torch.zeros(128, 1)
        samples = [float(self.loss_computer.make_re_norm(x)[0, 0]) for _ in range(50)]
        self.assertGreaterEqual(min(samples), 0.5 - 1e-6)
        self.assertLessEqual(max(samples), 2.0 + 1e-6)
        self.assertGreater(len({round(s, 4) for s in samples}), 1)

    def test_non_parametric_backward_compatible(self) -> None:
        config = _config(data_loss_enabled=False, parametric=False)
        loss_computer = LossComputer(
            config=config,
            data_handler=self.data_handler,
            rho_norm=torch.tensor(1.0),
            mu_norm=torch.tensor(1.0 / self.re_ref),
            x_min=self.data_handler.x_min,
            x_max=self.data_handler.x_max,
            y_min=self.data_handler.y_min,
            y_max=self.data_handler.y_max,
            device=self.device,
            dtype=self.dtype,
            re_ref=self.re_ref,
        )
        model3 = PINNIncompressible(
            input_dim=3,
            layers=[16, 16],
            activation='tanh',
            output_dim=3,
            output_bias=(0.0, 0.0, 0.0),
            coordinate_scaling={'enabled': False, 'scale_factor': 1.0},
            fourier_features={'enabled': True, 'n_frequencies': 4, 'sigma': 3.0, 'gamma': 1.0},
            residual=False,
            parametric=False,
        )
        x = torch.zeros(4, 1)
        inp = loss_computer.build_input_tensor(x, x, x)
        self.assertEqual(tuple(inp.shape), (4, 3))
        result = loss_computer.compute_loss(self.batch, model3)
        self.assertTrue(bool(torch.isfinite(result['loss']).all()))
        self.assertEqual(result['data_loss'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

