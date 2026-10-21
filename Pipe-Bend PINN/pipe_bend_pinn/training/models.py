from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

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


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping: x -> [x, gamma*cos(B@x), gamma*sin(B@x)].

    Улучшает сходимость PINN для гладких полей с высокочастотными
    составляющими (пограничные слои, отрыв в отводе).
    """
    def __init__(self, in_dim: int, n_frequencies: int = 32,
                 sigma: float = 3.0, gamma: float = 1.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.gamma = gamma
        B = torch.randn(in_dim, n_frequencies) * sigma
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B                          # (B, n_freq)
        feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.cat([x, self.gamma * feat], dim=-1)

    def output_dim(self, in_dim: int) -> int:
        return in_dim + 2 * self.n_frequencies


class ResidualBlock(nn.Module):
    """Residual MLP-блок: x -> x + fc2(act(fc1(x))). Стабилизирует обучение глубоких сетей."""
    def __init__(self, dim: int, act: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


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
        coordinate_scaling: Optional[Dict] = None,
        fourier_features: Optional[Dict] = None,
        residual: bool = False,
        parametric: bool = False,
        spatial_dim: int = 3,
        param_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        self.residual = bool(residual)

        # Параметрический режим: 4-й вход — безразмерный параметр Re (re_norm).
        # Fourier-признаки применяются ТОЛЬКО к пространственным входам, а
        # параметр конкатенируется после них: иначе каждая скорость получала бы
        # собственный случайный embedding и обобщение по Re ломалось бы.
        self.parametric = bool(parametric)
        self.spatial_dim = int(spatial_dim) if self.parametric else int(input_dim)
        self.param_dim = int(param_dim) if self.parametric else 0
        if self.parametric and self.spatial_dim + self.param_dim != int(input_dim):
            raise ValueError(
                "parametric: spatial_dim + param_dim = "
                f"{self.spatial_dim + self.param_dim} != input_dim = {input_dim}"
            )

        self.coordinate_scaling = coordinate_scaling or {'enabled': False, 'scale_factor': 1.0}
        self.scale_factor = self.coordinate_scaling.get('scale_factor', 1.0) if self.coordinate_scaling.get('enabled', False) else 1.0

        # Fourier features (только на координаты/входные признаки)
        self.fourier = None
        ff_cfg = fourier_features or {}
        if ff_cfg.get('enabled', False):
            self.fourier = FourierFeatures(
                in_dim=self.spatial_dim,
                n_frequencies=ff_cfg.get('n_frequencies', 32),
                sigma=ff_cfg.get('sigma', 3.0),
                gamma=ff_cfg.get('gamma', 1.0),
            )
            prev = self.fourier.output_dim(self.spatial_dim) + self.param_dim
        else:
            prev = self.spatial_dim + self.param_dim

        for hidden in layers:
            if prev == hidden and self.residual:
                # Residual-блок только для слоёв одинаковой ширины
                self.hidden_layers.append(ResidualBlock(hidden, self.activation))
            else:
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
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'siren': Sine(),
        }
        if name not in activations:
            raise ValueError(f"Неизвестная функция активации: {name}")
        return activations[name]

    def _init_weights(self):
        for layer in self.hidden_layers:
            if isinstance(layer, ResidualBlock):
                for sub in layer.children():
                    if isinstance(sub, nn.Linear):
                        init.xavier_uniform_(sub.weight)
                        init.zeros_(sub.bias)
            elif self.activation_name == 'siren':
                fan_in = layer.weight.size(1)
                bound = math.sqrt(6 / fan_in)
                init.uniform_(layer.weight, -bound, bound)
                init.zeros_(layer.bias)
            else:
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parametric:
            spatial = x[:, :self.spatial_dim]
            param = x[:, self.spatial_dim:]
        else:
            spatial = x
            param = None

        if self.coordinate_scaling.get('enabled', False):
            h = spatial * self.scale_factor
        else:
            h = spatial

        if self.fourier is not None:
            h = self.fourier(h)

        if param is not None:
            h = torch.cat([h, param], dim=1)

        for layer in self.hidden_layers:
            if isinstance(layer, ResidualBlock):
                h = layer(h)
            else:
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