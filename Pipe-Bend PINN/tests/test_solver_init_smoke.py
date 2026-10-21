#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-тест инициализации PINNSolver с обновлёнными LossComputer/Trainer.

Запускает полный ``__init__`` решателя на CPU с отключённым FVM и делает один
шаг ``compute_loss`` — проверяет, что базовый (непараметрический) режим не
сломан.
"""

from __future__ import annotations

import copy
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver


def main() -> int:
    config = copy.deepcopy(ConfigManager().config)
    config['process']['device'] = 'cpu'
    config['process']['torch_compile'] = False
    config['loss_weights']['fvm']['enabled'] = False
    config['output']['console_only'] = True
    config['enable_key_listener'] = False
    config['training']['epochs'] = 1
    config['training']['rar_enable'] = False
    config['architecture']['parametric'] = False
    config['loss_weights']['data_loss'] = {'enabled': False, 'weight': 0.0}
    config['data']['cfd'] = {'enabled': False}

    solver = PINNSolver(config)
    model = solver.model
    batch = next(iter(solver.data_handler.get_dataloader()))
    result = solver.loss_computer.compute_loss(batch, model)

    print(
        'solver OK:',
        type(model).__name__,
        'parametric=', getattr(model, 'parametric', None),
        're_ref=', solver.re_ref,
    )
    print(
        'loss=', float(result['loss'].detach()),
        'data_loss=', result['data_loss'],
        'pde_max=', result['pde_max'],
    )
    assert result['data_loss'] == 0.0
    assert result['loss'].detach().isfinite()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
