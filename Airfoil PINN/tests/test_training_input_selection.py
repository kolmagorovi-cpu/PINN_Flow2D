"""Regression coverage for selected data files and coordinate warnings."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from airfoil_pinn.config.manager import ConfigManager, warn_activation
from airfoil_pinn.data.coordinate_bounds import read_coordinate_bounds
from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.training.menus.architecture import _configure_activation


class TrainingInputSelectionTests(unittest.TestCase):
    def test_warning_reads_selected_points_not_scaling_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected.npy'
            points = np.array([(-0.5, -0.25), (1.5, 0.75)],
                              dtype=[('x_norm', 'f8'), ('y_norm', 'f8')])
            np.save(path, points)
            cfg = ConfigManager()
            cfg.config['data']['points_file'] = str(path)
            output = io.StringIO()
            with patch('builtins.input', side_effect=['4', '', '', '', '']), contextlib.redirect_stdout(output):
                _configure_activation(cfg)
            self.assertIn('x/c: [-0.5, 1.5]', output.getvalue())
            self.assertIn('y/c: [-0.25, 0.75]', output.getvalue())
            self.assertNotIn('[-6', output.getvalue())
            self.assertEqual(cfg.config['architecture']['coordinate_scaling']['mode'], 'dataset_minmax')

    def test_unknown_bounds_do_not_display_defaults(self):
        output = io.StringIO()
        with patch('builtins.input', return_value=''), contextlib.redirect_stdout(output):
            warn_activation('siren')
        self.assertIn('неизвестен', output.getvalue())
        self.assertNotIn('-6', output.getvalue())

    def test_nonfinite_points_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npy'
            np.save(path, np.array([(np.nan, 0)], dtype=[('x_norm', 'f8'), ('y_norm', 'f8')]))
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                read_coordinate_bounds(str(path))

    def test_loading_config_does_not_inherit_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected.yaml'
            path.write_text('data:\n  points_file: selected.npy\n', encoding='utf-8')
            cfg = ConfigManager()
            cfg.config['data']['mesh_file'] = 'previous_experiment.npz'
            cfg.config['architecture']['siren_first_omega_0'] = 99
            cfg.load_from_file(str(path))
            self.assertEqual(cfg.config['data']['points_file'], 'selected.npy')
            self.assertEqual(cfg.config['data']['mesh_file'], cfg._default_config()['data']['mesh_file'])
            self.assertEqual(cfg.config['architecture']['siren_first_omega_0'], 30)

    def test_mesh_check_uses_chord_and_rejects_stale_domain(self):
        handler = DataHandler.__new__(DataHandler)
        handler.chord = 2.0
        handler.points_file = 'selected.npy'
        from types import SimpleNamespace
        handler.dataset = SimpleNamespace(x=torch.tensor([-0.5, 1.5]), y=torch.tensor([-0.5, 0.5]))
        handler._validate_mesh_coordinate_bounds(np.array([[-1, -1], [3, 1]]), 'valid.npz')
        with self.assertRaisesRegex(ValueError, 'stale.npz.*selected.npy'):
            handler._validate_mesh_coordinate_bounds(np.array([[-1, -1], [5, 1]]), 'stale.npz')


if __name__ == '__main__':
    unittest.main()
