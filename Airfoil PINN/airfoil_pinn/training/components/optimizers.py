#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль optimizers для проекта PINN Learning.
Содержит классы LocalMinimaDetector (детектор плато) и CyclicLR (циклический learning rate).
"""

import numpy as np
from collections import deque
from typing import Optional

__all__ = [
    "CyclicLR",
    "LocalMinimaDetector",
]


class LocalMinimaDetector:
    """
    Отслеживает скользящее среднее потерь и сигнализирует о застое (плато).
    Используется в стратегии обучения для обнаружения локальных минимумов.
    """

    def __init__(self, window: int = 20, threshold: float = 1e-4, patience: int = 10):
        """
        Args:
            window: размер окна для скользящего среднего
            threshold: относительный порог улучшения (если новое значение лучше best_loss
                       менее чем на threshold, считается отсутствием прогресса)
            patience: сколько эпох без улучшения, чтобы считать плато достигнутым
        """
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self.loss_history = deque(maxlen=window)
        self.steps_since_improvement = 0
        self.best_loss = float('inf')

    def update(self, loss: float) -> bool:
        """
        Добавляет новое значение потерь и возвращает True, если обнаружено плато.

        Args:
            loss: текущее значение функции потерь

        Returns:
            True, если количество шагов без улучшения превысило patience.
        """
        self.loss_history.append(loss)

        # Ждём, пока накопится достаточно данных
        if len(self.loss_history) < self.window:
            return False

        current_min = min(self.loss_history)

        # Проверяем улучшение относительно лучшего известного значения
        if current_min < self.best_loss * (1 - self.threshold):
            self.best_loss = current_min
            self.steps_since_improvement = 0
            return False
        else:
            self.steps_since_improvement += 1
            return self.steps_since_improvement >= self.patience


class CyclicLR:
    """
    Простой циклический планировщик learning rate, изменяющий LR по синусоидальному закону.
    """

    def __init__(self, optimizer, base_lr: float, max_lr: float, period: int):
        """
        Args:
            optimizer: оптимизатор PyTorch, для которого изменяется learning rate
            base_lr: минимальное значение learning rate
            max_lr: максимальное значение learning rate
            period: период колебаний в эпохах
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.period = period
        self.step_count = 0

    def step(self):
        """
        Обновляет learning rate в соответствии с текущим шагом.
        Должен вызываться после каждой эпохи (или батча, в зависимости от настроек).
        """
        self.step_count += 1

        # Синусоидальный закон: (sin(2π * t / period))^2
        cycle = np.sin(2 * np.pi * self.step_count / self.period) ** 2
        lr = self.base_lr + (self.max_lr - self.base_lr) * cycle

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Возвращает текущее значение learning rate (из первой группы параметров)."""
        return self.optimizer.param_groups[0]['lr']
