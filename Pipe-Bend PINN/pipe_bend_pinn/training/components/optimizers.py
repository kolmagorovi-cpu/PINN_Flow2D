from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Оптимизаторы для Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import numpy as np
from typing import Optional


class LocalMinimaDetector:
    """Детектор плато (локального минимума) в функции потерь."""

    def __init__(self, window: int = 20, threshold: float = 1e-4, patience: int = 50):
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self.loss_history = []
        self.plateau_count = 0
        self._min_loss = float('inf')

    def update(self, loss: float) -> bool:
        """
        Обновляет историю потерь и проверяет, достигнуто ли плато.

        Returns:
            True если плато обнаружено
        """
        self.loss_history.append(loss)
        self._min_loss = min(self._min_loss, loss)

        if len(self.loss_history) < self.window:
            return False

        recent = self.loss_history[-self.window:]
        max_val = max(recent)
        min_val = min(recent)
        relative_change = (max_val - min_val) / (abs(self._min_loss) + 1e-12)

        if relative_change < self.threshold:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        return self.plateau_count >= self.patience


class CyclicLR:
    """Циклический learning rate с косинусной модуляцией."""

    def __init__(self, optimizer, base_lr: float, max_lr: float, period: int = 100):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.period = period
        self.step_num = 0

    def step(self):
        """Обновляет learning rate."""
        self.step_num += 1
        theta = 2 * np.pi * self.step_num / self.period
        lr = self.base_lr + (self.max_lr - self.base_lr) * (0.5 * (1 - np.cos(theta)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr