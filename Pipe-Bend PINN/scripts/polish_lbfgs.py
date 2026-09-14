#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L-BFGS-полировка результата: resume с pinn_final.pt + короткая L-BFGS фаза."""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver


def main() -> int:
    cfg_path = os.path.join(PROJECT_DIR, "workspace", "configs",
                            "config_lbfgs_polish.yaml")
    print(f"=== L-BFGS полировка: {cfg_path}")
    cm = ConfigManager(config_path=cfg_path)
    cfg = cm.config
    print("   load_model_path:", cfg["process"]["load_model_path"])
    print("   epochs:", cfg["training"]["epochs"],
          "| switch_to_lbfgs_epoch:", cfg["training"]["switch_to_lbfgs_epoch"],
          "| lbfgs_max_iter:", cfg["training"]["lbfgs_max_iter"])
    solver = PINNSolver(cfg)
    solver.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
