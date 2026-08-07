"""Тест импортов для проверки совместимости измененных модулей."""
import sys
import os

# Добавляем корень проекта в sys.path (родительский каталог tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch
    from pipe_bend_pinn.data.handler import PINNDataset, DataHandler
    from pipe_bend_pinn.training.components.loss_computer import LossComputer
    from pipe_bend_pinn.training.components.trainer import Trainer
    from pipe_bend_pinn.training.components.metrics import MetricsCollector
    from pipe_bend_pinn.physics.pde import PDEResiduals
    from pipe_bend_pinn.training.models import PINNIncompressible, SelfAdaptiveWeights
    from pipe_bend_pinn.training.checkpoint import CheckpointManager
    from pipe_bend_pinn.training.solver import PINNSolver
    print("OK: all imports")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)