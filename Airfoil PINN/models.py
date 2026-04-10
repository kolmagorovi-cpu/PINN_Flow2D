#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели нейронных сетей для проекта PINN Learning.

Содержит классы PINNIncompressible и SelfAdaptiveWeights.

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

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Tuple, Optional, Dict
import numpy as np
import math


class Sine(nn.Module):
    """
    Синусоидальная активация для SIREN.

    Example:
        >>> activation = Sine()
        >>> x = torch.randn(10, 5)
        >>> y = activation(x)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применяет синусоидальную активацию.

        Args:
            x: Входной тензор

        Returns:
            Тензор с применённой активацией sin(x)
        """
        return torch.sin(x)


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
        coordinate_scaling: Optional[Dict] = None
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


class PotentialPINN(nn.Module):
    """
    Полносвязная нейросеть для предсказания потенциала скорости φ(x, y).
    
    Выход: один скаляр (потенциал φ).
    Скорости вычисляются через градиенты:
        u = ∂φ/∂x
        v = ∂φ/∂y
    
    Входные параметры могут включать координаты, SDF, параметры задачи.
    Поддерживает различные функции активации: tanh, relu, sigmoid, leaky_relu,
    elu, swish (SiLU) и siren.
    Поддерживает масштабирование координат для улучшения обусловленности.

    Attributes:
        input_dim: Размерность входного вектора
        hidden_layers: Список слоёв сети
        activation: Функция активации
        output_layer: Выходной слой (один выход: φ)
        coordinate_scaling: Параметры масштабирования координат
    
    Example:
        >>> model = PotentialPINN(
        ...     input_dim=3,
        ...     layers=[50, 50, 50, 50],
        ...     activation='tanh'
        ... )
        >>> phi = model(input_tensor)
        >>> u, v = model.get_velocity(input_tensor, x, y)
    """
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        activation: str = 'tanh',
        output_bias: float = 0.0,
        coordinate_scaling: Optional[Dict] = None
    ) -> None:
        """
        Инициализирует нейронную сеть для потенциала скорости.

        Args:
            input_dim: Размерность входного вектора
            layers: Список чисел нейронов в скрытых слоях
            activation: Тип функции активации
            output_bias: Начальное значение смещения выходного слоя
            coordinate_scaling: Параметры масштабирования координат
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
        # Один выход: потенциал φ
        self.output_layer = nn.Linear(prev, 1)

        # Инициализация весов
        self._init_weights()

        # Установка bias выходного слоя
        with torch.no_grad():
            self.output_layer.bias.data.fill_(output_bias)

    def _get_activation(self, name: str) -> nn.Module:
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
        # Выходной слой всегда инициализируем Xavier
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход с масштабированием координат (если включено).

        Args:
            x: входной тензор формы (batch_size, input_dim)
        Returns:
            тензор формы (batch_size, 1) - потенциал φ
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

    def get_velocity(
        self,
        input_tensor: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет компоненты скорости (u, v) через градиенты потенциала.
        
        u = ∂φ/∂x
        v = ∂φ/∂y

        Args:
            input_tensor: входной тензор для модели (batch_size, input_dim)
            x: координата x (batch_size, 1) с requires_grad=True
            y: координата y (batch_size, 1) с requires_grad=True

        Returns:
            tuple (u, v): компоненты скорости, тензоры той же формы, что и x,y
        """
        # Прямой проход для получения потенциала
        phi = self.forward(input_tensor)
        
        # Вычисление градиентов потенциала по координатам
        # torch.autograd.grad возвращает кортеж градиентов
        grad_phi = torch.autograd.grad(
            phi,
            [x, y],
            grad_outputs=torch.ones_like(phi),
            create_graph=True,  # Важно для вычисления вторых производных
            retain_graph=True
        )
        
        u = grad_phi[0]  # ∂φ/∂x
        v = grad_phi[1]  # ∂φ/∂y
        
        return u, v

    def get_potential_and_velocity(
        self,
        input_tensor: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет потенциал φ и компоненты скорости (u, v).
        
        Args:
            input_tensor: входной тензор для модели (batch_size, input_dim)
            x: координата x (batch_size, 1) с requires_grad=True
            y: координата y (batch_size, 1) с requires_grad=True

        Returns:
            tuple (phi, u, v): потенциал и компоненты скорости
        """
        phi = self.forward(input_tensor)
        
        grad_phi = torch.autograd.grad(
            phi,
            [x, y],
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )
        
        u = grad_phi[0]
        v = grad_phi[1]
        
        return phi, u, v

    def get_laplacian(
        self,
        input_tensor: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет потенциал, скорости и лапласиан ∇²φ.
        
        Используется для вычисления невязки PDE.

        Args:
            input_tensor: входной тензор для модели (batch_size, input_dim)
            x: координата x (batch_size, 1) с requires_grad=True
            y: координата y (batch_size, 1) с requires_grad=True

        Returns:
            tuple (phi, u, v, laplacian): потенциал, скорости и лапласиан
        """
        phi, u, v = self.get_potential_and_velocity(input_tensor, x, y)
        
        # Вторые производные для лапласиана
        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=False,
            retain_graph=False
        )[0]
        
        grad_v = torch.autograd.grad(
            v,
            y,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=False
        )[0]
        
        laplacian = grad_u + grad_v  # ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
        
        return phi, u, v, laplacian


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