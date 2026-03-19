#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели нейронных сетей для проекта PINN Learning.
Содержит классы PINNIncompressible и SelfAdaptiveWeights.

Автор: Колмагоров И.Д.
Версия: 3.260226 (с поддержкой выбора функции активации)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Tuple, Optional, Dict
import numpy as np
import math


class Sine(nn.Module):
    """Синусоидальная активация для SIREN."""
    def forward(self, x):
        return torch.sin(x)


class PINNIncompressible(nn.Module):
    """
    Полносвязная нейросеть для предсказания (u, v, p) по координатам (x, y) и SDF.
    Входные параметры могут включать дополнительные физические величины.
    Параметр output_bias задаёт начальное смещение выходного слоя.
    Поддерживает различные функции активации: tanh, relu, sigmoid, leaky_relu,
    elu, swish (SiLU) и siren.
    Поддерживает масштабирование координат для улучшения обусловленности.
    """
    def __init__(self, input_dim: int, layers: List[int], activation: str = 'tanh',
                 output_dim: int = 3,
                 output_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 coordinate_scaling: Optional[Dict] = None):
        """
        Args:
            input_dim: размерность входного вектора
            layers: список чисел нейронов в скрытых слоях
            activation: тип функции активации ('tanh', 'relu', 'sigmoid',
                       'leaky_relu', 'elu', 'swish', 'siren')
            output_dim: размерность выхода (обычно 3: u, v, p)
            output_bias: начальные значения смещения выходного слоя (u0, v0, p0)
            coordinate_scaling: словарь с параметрами масштабирования:
                - enabled: bool (включено ли масштабирование)
                - scale_factor: float (коэффициент масштабирования)
                - input_min: float (минимальное значение входа)
                - input_max: float (максимальное значение входа)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        
        # Параметры масштабирования координат
        self.coordinate_scaling = coordinate_scaling or {'enabled': False, 'scale_factor': 1.0}
        self.scale_factor = self.coordinate_scaling.get('scale_factor', 1.0) if self.coordinate_scaling.get('enabled', False) else 1.0

        prev = input_dim
        for hidden in layers:
            self.hidden_layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.output_layer = nn.Linear(prev, output_dim)

        # Инициализация весов
        self._init_weights()

        # Установка bias выходного слоя
        with torch.no_grad():
            bias = torch.tensor(output_bias[:output_dim], dtype=torch.float32)
            self.output_layer.bias.data = bias

    def _get_activation(self, name: str) -> nn.Module:
        """Возвращает модуль активации по имени."""
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif name == 'elu':
            return nn.ELU()
        elif name == 'swish':
            return nn.SiLU()  # SiLU = Swish
        elif name == 'siren':
            return Sine()
        else:
            raise ValueError(f"Неизвестная функция активации: {name}")

    def _init_weights(self):
        """Инициализация весов в зависимости от типа активации."""
        for layer in self.hidden_layers:
            if self.activation_name == 'siren':
                # Инициализация для SIREN: равномерное распределение с масштабом sqrt(6 / fan_in)
                fan_in = layer.weight.size(1)
                bound = math.sqrt(6 / fan_in)
                init.uniform_(layer.weight, -bound, bound)
                init.zeros_(layer.bias)
            else:
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        # Выходной слой всегда инициализируем Xavier (можно также использовать другую инициализацию)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход с масштабированием координат (если включено).
        
        Args:
            x: входной тензор формы (batch_size, input_dim)
        Returns:
            тензор формы (batch_size, output_dim)
        """
        # Масштабирование координат для улучшения обусловленности
        if self.coordinate_scaling.get('enabled', False):
            h = x * self.scale_factor
        else:
            h = x
        
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
        out = self.output_layer(h)
        return out


class SelfAdaptiveWeights(nn.Module):
    """
    Обучаемые веса для каждой точки и каждого уравнения.
    Используются в SA-PINN для автоматической балансировки вклада точек.
    Веса всегда положительны благодаря Softplus.
    """
    def __init__(self, n_points: int, n_eqns: int, init_val: float = 1.0):
        """
        Args:
            n_points: общее количество точек коллокации
            n_eqns: количество уравнений (например, 3 для PDE)
            init_val: начальное значение весов
        """
        super().__init__()
        self.log_weights = nn.Parameter(torch.full((n_points, n_eqns), np.log(init_val)))
        self.softplus = nn.Softplus()

    def forward(self) -> torch.Tensor:
        """
        Возвращает тензор весов формы (n_points, n_eqns).
        """
        return self.softplus(self.log_weights)