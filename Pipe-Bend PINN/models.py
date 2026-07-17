"""
Модели нейронных сетей для Pipe-Bend PINN.

Содержит классы PINNIncompressible и SelfAdaptiveWeights.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Tuple, Optional, Dict
import numpy as np
import math


class Sine(nn.Module):
    """Синусоидальная активация для SIREN."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class PINNIncompressible(nn.Module):
    """
    Полносвязная нейросеть для предсказания (u, v, p) по координатам (x, y) и SDF.

    Поддерживает различные функции активации и масштабирование координат.
    """
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        activation: str = 'tanh',
        output_dim: int = 3,
        output_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        coordinate_scaling: Optional[Dict] = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        
        self.coordinate_scaling = coordinate_scaling or {'enabled': False, 'scale_factor': 1.0}
        self.scale_factor = self.coordinate_scaling.get('scale_factor', 1.0) if self.coordinate_scaling.get('enabled', False) else 1.0

        prev = input_dim
        for hidden in layers:
            self.hidden_layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.output_layer = nn.Linear(prev, output_dim)

        self._init_weights()

        with torch.no_grad():
            bias = torch.tensor(output_bias[:output_dim], dtype=torch.float32)
            self.output_layer.bias.data = bias

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'siren': Sine(),
        }
        if name not in activations:
            raise ValueError(f"Неизвестная функция активации: {name}")
        return activations[name]

    def _init_weights(self):
        for layer in self.hidden_layers:
            if self.activation_name == 'siren':
                fan_in = layer.weight.size(1)
                bound = math.sqrt(6 / fan_in)
                init.uniform_(layer.weight, -bound, bound)
                init.zeros_(layer.bias)
            else:
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    def __init__(self, n_points: int, n_eqns: int, init_val: float = 1.0):
        super().__init__()
        self.log_weights = nn.Parameter(torch.full((n_points, n_eqns), np.log(init_val)))
        self.softplus = nn.Softplus()

    def forward(self) -> torch.Tensor:
        return self.softplus(self.log_weights)