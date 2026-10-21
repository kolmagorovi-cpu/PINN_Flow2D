#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный запуск обучения Pipe-Bend PINN (ламинарный режим Re=500).

Конфигурация: workspace/configs/config.yaml
  - 20000 эпох Adam (cosine LR), двухэтапное обучение
  - SA-PINN (per-point веса)
  - RAR каждые 1000 эпох
  - L-BFGS полировка с эпохи 15000

Запуск:
    python run_training.py
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Обучение Pipe-Bend PINN")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "workspace", "configs", "config.yaml"),
        help="путь к YAML-конфигу (по умолчанию workspace/configs/config.yaml)",
    )
    args = parser.parse_args()
    config_path = args.config
    print(f"=== Запуск обучения по конфигу: {config_path}")
    cm = ConfigManager(config_path=config_path)
    cfg = cm.config
    print(f"   Проект: {cfg.get('project_name')}")
    print(f"   Точки: {cfg['data']['points_file']}")
    print(f"   Архитектура: {cfg['architecture']['hidden_layers']} "
          f"{cfg['architecture']['activation']} FF={cfg['architecture']['fourier_features']['enabled']}")

    solver = PINNSolver(cfg)
    solver.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
