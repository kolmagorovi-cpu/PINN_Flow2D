# Исправленный файл: data_handler.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка данных для проекта PINN Learning.
Загружает точки коллокации и метаданные, предоставляет DataLoader.
Поддерживает hard граничные условия (маски фиксированных компонент).
Добавлены методы для диагностики и динамического расширения датасета (RAR).

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная, с поддержкой расширения датасета и фильтрации по y)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class PINNDataset(Dataset):
    """
    Dataset для точек коллокации.
    Возвращает кортеж: (idx, x, y, sdf, boundary_types,
                        mask_u_fixed, mask_v_fixed, mask_p_fixed,
                        u_bc, v_bc, p_bc)
    Все тензоры хранятся в float32.
    """
    def __init__(self, points_file: str):
        data = np.load(points_file, allow_pickle=True)
        if data.dtype.names is None:
            raise ValueError("Файл точек должен быть структурированным массивом")

        required = ['x_norm', 'y_norm', 'sdf_norm', 'p_norm', 'u_norm', 'v_norm']
        for fld in required:
            if fld not in data.dtype.names:
                raise ValueError(f"В файле отсутствует поле {fld}")

        self.x = torch.tensor(np.ascontiguousarray(data['x_norm']), dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(np.ascontiguousarray(data['y_norm']), dtype=torch.float32).view(-1, 1)
        self.sdf = torch.tensor(np.ascontiguousarray(data['sdf_norm']), dtype=torch.float32).view(-1, 1)

        # Целевые значения (для справки, не используются при hard)
        self.p_target = torch.tensor(np.ascontiguousarray(data['p_norm']), dtype=torch.float32).view(-1, 1)
        self.u_target = torch.tensor(np.ascontiguousarray(data['u_norm']), dtype=torch.float32).view(-1, 1)
        self.v_target = torch.tensor(np.ascontiguousarray(data['v_norm']), dtype=torch.float32).view(-1, 1)

        # Обработка типов границ (one-hot) – все имена в нижнем регистре
        self.boundary_types = None
        self.boundary_type_names = []
        self.type_to_idx = {}  # словарь для быстрого доступа к индексам

        # Маски фиксированных компонент (по умолчанию все False)
        self.mask_u_fixed = torch.zeros(len(self.x), dtype=torch.bool)
        self.mask_v_fixed = torch.zeros(len(self.x), dtype=torch.bool)
        self.mask_p_fixed = torch.zeros(len(self.x), dtype=torch.bool)

        # Значения на границах (по умолчанию 0)
        self.u_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)
        self.v_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)
        self.p_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)

        if 'boundary_type' in data.dtype.names:
            # Приводим все уникальные значения к нижнему регистру
            unique_raw = np.unique(data['boundary_type'])
            self.boundary_type_names = [str(t).lower() for t in unique_raw]
            self.type_to_idx = {t: i for i, t in enumerate(self.boundary_type_names)}

            # Преобразуем исходные значения в нижний регистр для поиска индекса
            indices = []
            for bt in data['boundary_type']:
                bt_lower = str(bt).lower()
                indices.append(self.type_to_idx[bt_lower])

            one_hot = np.zeros((len(indices), len(self.boundary_type_names)), dtype=np.float32)
            one_hot[np.arange(len(indices)), indices] = 1.0
            self.boundary_types = torch.tensor(np.ascontiguousarray(one_hot), dtype=torch.float32)

            # Заполняем маски и значения в зависимости от типа границы
            for i, bt in enumerate(data['boundary_type']):
                bt_str = str(bt).lower()

                if bt_str == 'inlet':
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # Значения будут позже загружены из метаданных

                elif bt_str == 'outlet':
                    self.mask_p_fixed[i] = True
                    # p = 0 (уже по умолчанию)

                elif bt_str == 'wall':
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # u = 0, v = 0 (уже по умолчанию)

                elif bt_str == 'symmetry':
                    self.mask_v_fixed[i] = True
                    # v = 0

                elif bt_str == 'axis_symmetry':
                    self.mask_u_fixed[i] = True
                    # u = 0

        self.n_points = len(self.x)
        logger.info(f"Типы границ в данных: {self.boundary_type_names}")

    def set_inlet_values(self, u_norm: float, v_norm: float):
        """Устанавливает значения на inlet после загрузки метаданных."""
        if self.boundary_types is None:
            logger.warning("Нет информации о границах, невозможно установить значения на inlet.")
            return

        inlet_idx = self.type_to_idx.get('inlet')
        if inlet_idx is None:
            logger.warning("Тип 'inlet' отсутствует в данных.")
            return

        if inlet_idx >= self.boundary_types.shape[1]:
            logger.error(f"Индекс inlet ({inlet_idx}) выходит за пределы размерности boundary_types ({self.boundary_types.shape[1]})")
            return

        mask = self.boundary_types[:, inlet_idx] > 0.5
        self.u_bc[mask] = u_norm
        self.v_bc[mask] = v_norm

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> Tuple:
        # Базовые тензоры
        result = (idx, self.x[idx], self.y[idx], self.sdf[idx])

        # One-hot границ (если есть)
        if self.boundary_types is not None:
            result += (self.boundary_types[idx],)
        else:
            result += (torch.zeros(0),)

        # Маски и значения
        result += (self.mask_u_fixed[idx].view(1),
                   self.mask_v_fixed[idx].view(1),
                   self.mask_p_fixed[idx].view(1),
                   self.u_bc[idx].view(1),
                   self.v_bc[idx].view(1),
                   self.p_bc[idx].view(1))

        return result

    def extend(self, indices: torch.Tensor) -> 'PINNDataset':
        """
        Создаёт новый датасет, дополненный точками по указанным индексам.
        Индексы должны быть целочисленным тензором на CPU.
        """
        new_dataset = PINNDataset.__new__(PINNDataset)
        # Копируем все атрибуты, конкатенируя тензоры
        for attr in ['x', 'y', 'sdf', 'p_target', 'u_target', 'v_target',
                     'mask_u_fixed', 'mask_v_fixed', 'mask_p_fixed',
                     'u_bc', 'v_bc', 'p_bc', 'boundary_types']:
            old = getattr(self, attr)
            if old is not None:
                new = torch.cat([old, old[indices]], dim=0)
                setattr(new_dataset, attr, new)
            else:
                setattr(new_dataset, attr, None)

        new_dataset.boundary_type_names = self.boundary_type_names
        new_dataset.type_to_idx = self.type_to_idx
        new_dataset.n_points = len(new_dataset.x)
        return new_dataset


class DataHandler:
    """Загружает данные и метаданные, предоставляет DataLoader и методы для диагностики."""
    def __init__(self, points_file: str, batch_size: int, shuffle: bool = True):
        self.points_file = points_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = PINNDataset(points_file)
        self.load_metadata()
        # После загрузки метаданных устанавливаем значения на inlet
        self.dataset.set_inlet_values(self.inlet_u_norm, self.inlet_v_norm)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def load_metadata(self) -> None:
        """Загружает метаданные из файла _metadata.npy."""
        meta_path = self.points_file.replace('.npy', '_metadata.npy')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Файл метаданных {meta_path} не найден")
        meta = np.load(meta_path, allow_pickle=True).item()
        self.chord = meta.get('chord', 1.0)
        self.U_inf = meta.get('U_inf', 1.0)
        self.p_ref = meta.get('p_ref', 0.0)
        self.rho = meta.get('rho', 1.0)
        self.mu = meta.get('mu', 1.0)
        self.Re = meta.get('Re', 1.0)
        self.inlet_u_norm = meta.get('inlet_u_norm', 1.0)
        self.inlet_v_norm = meta.get('inlet_v_norm', 0.0)
        # Границы области (нормированные)
        self.x_min = self.dataset.x.min().item()
        self.x_max = self.dataset.x.max().item()
        self.y_min = self.dataset.y.min().item()
        self.y_max = self.dataset.y.max().item()
        logger.info(f"Метаданные: chord={self.chord:.4f} м, U_inf={self.U_inf:.4f} м/с, Re={self.Re:.2e}")
        logger.info(f"Вход: u_norm={self.inlet_u_norm:.3f}, v_norm={self.inlet_v_norm:.3f}")

    def get_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для полного датасета (все точки)."""
        return self.dataloader

    @property
    def n_points(self) -> int:
        return self.dataset.n_points

    @property
    def boundary_type_names(self) -> List[str]:
        return self.dataset.boundary_type_names

    def add_points(self, indices: torch.Tensor) -> None:
        """
        Добавляет точки по индексам в датасет и обновляет DataLoader.
        Индексы должны быть целочисленным тензором на CPU.
        """
        self.dataset = self.dataset.extend(indices)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        logger.info(f"Датасет расширен: добавлено {len(indices)} точек, всего {self.n_points}")

    # ------------------------------------------------------------------
    # Методы для диагностики (с фильтрацией по y)
    # ------------------------------------------------------------------
    def get_points_by_type(self, type_name: str) -> Dict[str, torch.Tensor]:
        """
        Возвращает координаты и значения для точек заданного типа границы.
        Возвращает словарь с ключами: 'x', 'y', 'u_target', 'v_target', 'p_target',
        'sdf', 'mask_u', 'mask_v', 'mask_p', 'u_bc', 'v_bc', 'p_bc'.
        """
        if type_name not in self.boundary_type_names:
            raise ValueError(f"Тип границы '{type_name}' не найден в данных")
        idx = self.boundary_type_names.index(type_name)
        mask = self.dataset.boundary_types[:, idx] > 0.5
        return {
            'x': self.dataset.x[mask],
            'y': self.dataset.y[mask],
            'u_target': self.dataset.u_target[mask],
            'v_target': self.dataset.v_target[mask],
            'p_target': self.dataset.p_target[mask],
            'sdf': self.dataset.sdf[mask],
            'mask_u': self.dataset.mask_u_fixed[mask],
            'mask_v': self.dataset.mask_v_fixed[mask],
            'mask_p': self.dataset.mask_p_fixed[mask],
            'u_bc': self.dataset.u_bc[mask],
            'v_bc': self.dataset.v_bc[mask],
            'p_bc': self.dataset.p_bc[mask],
        }

    def get_points_by_type_and_coordinate(self, type_name: str, coord: str, value: float, tol: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Возвращает точки заданного типа, у которых координата coord ('x' или 'y')
        близка к значению value с допуском tol (в нормированных единицах).
        """
        pts = self.get_points_by_type(type_name)
        if coord == 'x':
            mask = torch.abs(pts['x'].flatten() - value) < tol
        elif coord == 'y':
            mask = torch.abs(pts['y'].flatten() - value) < tol
        else:
            raise ValueError("coord должен быть 'x' или 'y'")
        return {k: v[mask] for k, v in pts.items()}

    def interpolate_profile(self, x_target: float, u_vals: Optional[torch.Tensor] = None,
                            y_grid: Optional[torch.Tensor] = None,
                            num_points: int = 100, y_max_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интерполирует профиль u(y) в сечении x = x_target (нормированное).
        Если u_vals не передан, используются начальные значения (self.dataset.u_target).
        Игнорируются точки с y > y_max_ratio * self.y_max (для исключения приграничных эффектов).
        Если y_grid не задан, создаётся равномерная сетка от y_min до y_max_ratio * y_max.
        Возвращает (y_coords, u_interp) – массивы numpy.
        """
        tol = 0.05 * (self.x_max - self.x_min)
        # Выбираем точки, близкие к x_target
        mask_x = torch.abs(self.dataset.x.flatten() - x_target) < tol
        # Ограничение по y
        y_limit = y_max_ratio * self.y_max
        mask_y = self.dataset.y.flatten() < y_limit
        mask = mask_x & mask_y

        if mask.sum() < 3:
            logger.warning(f"Мало точек в сечении x={x_target} после ограничения y<{y_limit:.3f}, интерполяция может быть неточной.")

        y_vals = self.dataset.y[mask].cpu().numpy().flatten()

        if u_vals is not None:
            # Используем переданные значения (предсказания)
            u_vals_np = u_vals[mask].cpu().numpy().flatten()
        else:
            # Используем начальные значения (u_target) – для обратной совместимости
            u_vals_np = self.dataset.u_target[mask].cpu().numpy().flatten()

        # Сортируем по y
        sort_idx = np.argsort(y_vals)
        y_sorted = y_vals[sort_idx]
        u_sorted = u_vals_np[sort_idx]

        # Если нужна равномерная сетка
        if y_grid is None:
            y_grid = np.linspace(self.y_min, y_limit, num_points)
        u_interp = np.interp(y_grid, y_sorted, u_sorted, left=np.nan, right=np.nan)
        # Заполняем NaN ближайшими значениями (экстраполяция)
        if np.any(np.isnan(u_interp)):
            u_interp = np.interp(y_grid, y_sorted, u_sorted, left=u_sorted[0], right=u_sorted[-1])
        return y_grid, u_interp