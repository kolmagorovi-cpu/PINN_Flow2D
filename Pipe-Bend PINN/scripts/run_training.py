#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Универсальный запуск обучения PINN по пути к конфигу (для любых конфигов и геометрий).

Пример:
    python scripts/run_training.py workspace/configs/config_refine_v6.yaml
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver


def main(argv) -> int:
    if len(argv) < 2:
        print("Использование: python scripts/run_training.py <config.yaml>")
        return 1
    cfg_path = os.path.join(PROJECT_DIR, argv[1])
    if not os.path.exists(cfg_path):
        # допустим абсолютный путь
        cfg_path = argv[1]
    print(f"=== Обучение: {cfg_path}")
    cm = ConfigManager(config_path=cfg_path)
    cfg = cm.config
    print("   project_name:", cfg.get("project_name"))
    print("   epochs:", cfg["training"]["epochs"],
          "| load_model_path:", cfg["process"].get("load_model_path"))
    solver = PINNSolver(cfg)
    solver.train()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
