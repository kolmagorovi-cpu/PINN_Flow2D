#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable neural architectures used by the incompressible-flow PINN.

The module contains a fully connected field model and pointwise SA-PINN
weights.  The SIREN option follows the frequency scaling and layer-specific
initialization proposed by Sitzmann et al.; this is important because the
physics loss differentiates network outputs twice with respect to coordinates.

Example:
    >>> model = PINNIncompressible(
    ...     input_dim=5,
    ...     layers=[50, 50, 50],
    ...     activation='tanh',
    ...     output_dim=3
    ... )

Автор: Колмагоров И.Д.
Версия: 3.260226 (с поддержкой выбора функции активации)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    "PINNIncompressible",
    "SelfAdaptiveWeights",
    "Sine",
]


class Sine(nn.Module):
    """Frequency-scaled sinusoidal activation used by SIREN.

    Example:
        >>> activation = Sine(omega_0=30.0)
        >>> x = torch.randn(10, 5)
        >>> y = activation(x)
    """

    def __init__(self, omega_0: float = 30.0) -> None:
        """Store the positive angular-frequency multiplier."""
        super().__init__()
        if not math.isfinite(omega_0) or omega_0 <= 0.0:
            raise ValueError("SIREN omega_0 must be finite and positive")
        self.omega_0 = float(omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sin(omega_0 * x)`` without detaching the graph."""
        return torch.sin(self.omega_0 * x)


class PINNIncompressible(nn.Module):
    """
    Полносвязная нейросеть для предсказания (u, v, p) по координатам (x, y) и SDF.

    Входные параметры могут включать дополнительные физические величины.
    Параметр output_bias задаёт начальное смещение выходного слоя.
    Поддерживает различные функции активации: tanh, relu, sigmoid, leaky_relu,
    elu, swish (SiLU) и siren.
    Поддерживает масштабирование координат для улучшения обусловленности.

    Attributes:
        input_dim: Размерность входного вектора
        hidden_layers: Список слоёв сети
        activation: Функция активации
        output_layer: Выходной слой
        coordinate_scaling: Параметры масштабирования координат
    """
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        activation: str = 'tanh',
        output_dim: int = 3,
        output_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        coordinate_scaling: Optional[Dict] = None,
        siren_first_omega_0: float = 30.0,
        siren_hidden_omega_0: float = 30.0,
    ) -> None:
        """
        Инициализирует нейронную сеть.

        Args:
            input_dim: Размерность входного вектора
            layers: Список чисел нейронов в скрытых слоях
            activation: Тип функции активации
            output_dim: Размерность выхода (обычно 3: u, v, p)
            output_bias: Начальные значения смещения выходного слоя
            coordinate_scaling: Параметры масштабирования координат
            siren_first_omega_0: SIREN frequency in the first hidden layer.
            siren_hidden_omega_0: SIREN frequency in later hidden layers.
        """
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Model input_dim and output_dim must be positive")
        if not layers or any(width <= 0 for width in layers):
            raise ValueError("Model must contain positive-width hidden layers")
        for name, value in (
            ("siren_first_omega_0", siren_first_omega_0),
            ("siren_hidden_omega_0", siren_hidden_omega_0),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.activation_name = str(activation).lower()
        self.siren_first_omega_0 = float(siren_first_omega_0)
        self.siren_hidden_omega_0 = float(siren_hidden_omega_0)
        self.activations = nn.ModuleList()

        # Параметры масштабирования координат
        self.coordinate_scaling = coordinate_scaling or {
            'enabled': False,
            'scale_factor': 1.0,
        }
        self.scale_factor = (
            self.coordinate_scaling.get('scale_factor', 1.0)
            if self.coordinate_scaling.get('enabled', False)
            else 1.0
        )
        self.coord_dim = int(self.coordinate_scaling.get('coord_dim', 2))

        prev = input_dim
        for index, hidden in enumerate(layers):
            self.hidden_layers.append(nn.Linear(prev, hidden))
            omega_0 = (
                self.siren_first_omega_0
                if index == 0
                else self.siren_hidden_omega_0
            )
            self.activations.append(
                self._get_activation(self.activation_name, omega_0=omega_0)
            )
            prev = hidden
        self.output_layer = nn.Linear(prev, output_dim)
        # Backward-compatible inspection attribute. Forward uses the
        # layer-specific modules above.
        self.activation = self.activations[0]

        # Инициализация весов
        self._init_weights()

        # Установка bias выходного слоя
        with torch.no_grad():
            bias = torch.as_tensor(
                output_bias,
                dtype=self.output_layer.bias.dtype,
                device=self.output_layer.bias.device,
            )
            if bias.numel() != output_dim:
                raise ValueError(
                    "output_bias length must equal model output_dim"
                )
            self.output_layer.bias.copy_(bias)

    def _get_activation(
        self,
        name: str,
        *,
        omega_0: float = 30.0,
    ) -> nn.Module:
        """
        Возвращает модуль активации по имени.

        Args:
            name: Название функции активации

        Returns:
            Модуль активации PyTorch

        Raises:
            ValueError: Если функция активации не поддерживается
        """
        if name == 'tanh':
            return nn.Tanh()
        if name == 'relu':
            return nn.ReLU()
        if name == 'sigmoid':
            return nn.Sigmoid()
        if name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        if name == 'elu':
            return nn.ELU()
        if name == 'swish':
            return nn.SiLU()  # SiLU = Swish
        if name == 'siren':
            return Sine(omega_0=omega_0)
        raise ValueError(f"Неизвестная функция активации: {name}")

    def _init_weights(self) -> None:
        """Initialize smooth MLP or frequency-scaled SIREN weights."""
        for index, layer in enumerate(self.hidden_layers):
            if self.activation_name == 'siren':
                fan_in = layer.weight.size(1)
                if index == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = (
                        math.sqrt(6.0 / fan_in)
                        / self.siren_hidden_omega_0
                    )
                init.uniform_(layer.weight, -bound, bound)
            else:
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)
        if self.activation_name == 'siren':
            output_bound = (
                math.sqrt(6.0 / self.output_layer.weight.size(1))
                / self.siren_hidden_omega_0
            )
            init.uniform_(
                self.output_layer.weight,
                -output_bound,
                output_bound,
            )
        else:
            init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход с масштабированием координат (если включено).

        Args:
            x: входной тензор формы (batch_size, input_dim)
        Returns:
            тензор формы (batch_size, output_dim)
        """
        # Масштабирование только координат x,y; остальные признаки сохраняют физический смысл.
        if self.coordinate_scaling.get('enabled', False) and self.coord_dim > 0:
            h = x.clone()
            coord_dim = min(self.coord_dim, h.shape[1])
            h[:, :coord_dim] = h[:, :coord_dim] * self.scale_factor
        else:
            h = x

        for layer, activation in zip(
            self.hidden_layers,
            self.activations,
        ):
            h = layer(h)
            h = activation(h)
        return self.output_layer(h)


class SelfAdaptiveWeights(nn.Module):
    """Positive pointwise SA-PINN weights trained by gradient ascent.

    The network parameters minimize the weighted loss, whereas the attention
    parameters maximize it, as required by the SA-PINN min--max objective.
    Gradient reversal lets both parameter groups use one Adam optimizer without
    accidentally driving all attention weights toward zero.
    """

    def __init__(
        self,
        n_points: int,
        n_eqns: int,
        init_val: float = 1.0,
        max_value: float = 100.0,
    ):
        """Initialize bounded softplus attention masks.

        Args:
            n_points: Number of collocation-point associations.
            n_eqns: Number of independently weighted residual components.
            init_val: Positive initial attention value.
            max_value: Positive safety ceiling applied after optimizer steps.
        """
        super().__init__()
        if n_points <= 0 or n_eqns <= 0:
            raise ValueError("SelfAdaptiveWeights dimensions must be positive")
        if init_val <= 0:
            raise ValueError("init_val для SelfAdaptiveWeights должен быть положительным")
        if max_value <= init_val:
            raise ValueError("max_value must be greater than init_val")
        self.max_value = float(max_value)
        self.initial_raw_value = float(np.log(np.expm1(init_val)))
        self.maximum_raw_value = float(np.log(np.expm1(max_value)))
        self.log_weights = nn.Parameter(
            torch.full((n_points, n_eqns), self.initial_raw_value)
        )
        self.log_weights.register_hook(self._reverse_gradient)
        self.softplus = nn.Softplus()

    @staticmethod
    def _reverse_gradient(gradient: torch.Tensor) -> torch.Tensor:
        """Convert optimizer descent into ascent for attention parameters."""
        return -gradient

    def forward(self) -> torch.Tensor:
        """Return positive weights with shape ``(n_points, n_eqns)``."""
        return self.softplus(self.log_weights)

    @torch.no_grad()
    def post_step_(self) -> None:
        """Clamp raw weights after an optimizer step for numerical robustness."""
        self.log_weights.clamp_(max=self.maximum_raw_value)

    def extend(self, n_new: int) -> None:
        """Append freshly initialized point weights while preserving ascent."""
        if n_new <= 0:
            return
        current = self.log_weights.detach()
        appended = torch.full(
            (int(n_new), current.shape[1]),
            self.initial_raw_value,
            device=current.device,
            dtype=current.dtype,
        )
        self.log_weights = nn.Parameter(torch.cat([current, appended], dim=0))
        self.log_weights.register_hook(self._reverse_gradient)
