"""Тест импортов для проверки совместимости измененных модулей."""
import sys
import os

# Добавляем текущую папку в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import torch
    from data_handler import PINNDataset, DataHandler
    from solver_components.loss_computer import LossComputer
    from solver_components.trainer import Trainer
    from solver_components.metrics import MetricsCollector
    from pde_residuals import PDEResiduals
    from models import PINNIncompressible, SelfAdaptiveWeights
    from checkpoint import CheckpointManager
    from solver import PINNSolver
    print("OK: all imports")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)