# Исправленный файл: data_handler.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка данных для проекта PINN Learning.

Загружает точки коллокации и метаданные, предоставляет DataLoader.
Поддерживает hard граничные условия (маски фиксированных компонент).
Добавлены методы для диагностики и динамического расширения датасета (RAR).

Example:
    >>> handler = DataHandler('collocation_points.npy', batch_size=10)
    >>> dataset = handler.dataset
    >>> print(f"Загружено {handler.n_points} точек")

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная, с поддержкой расширения датасета и фильтрации по y)
"""

import os
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset
import logging
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree

from type_hints import PointDtypeDict

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """
    FVM-сетка, загруженная из mesh2d.npz.

    Все массивы хранятся в numpy; метод to_torch() конвертирует
    нужные поля в тензоры для вычислений в loss_computer.

    Attributes:
        coords:           (N, 2)  физические координаты узлов [м]
        triangles:        (T, 3)  индексы вершин треугольников
        cv_volumes:       (N,)    объёмы КО [м²]
        int_face_node_i:  (F,)    узел i внутренней грани
        int_face_node_j:  (F,)    узел j внутренней грани
        int_face_area:    (F, 2)  векторы площадей S_ij (outward от i)
        bnd_face_node:    (Fb,)   узлы граничных граней
        bnd_face_neighbor:(Fb,)   смежные граничные узлы
        bnd_face_type:    (Fb,)   строки типов ГУ ('wall', 'inlet', ...)
        bnd_face_area:    (Fb, 2) векторы площадей граничных граней (outward)
        node_type:        (N,)    'interior' или 'boundary'
        node_bnd_type:    (N,)    строки типов ГУ узлов
    """
    coords: np.ndarray
    triangles: np.ndarray
    cv_volumes: np.ndarray
    int_face_node_i: np.ndarray
    int_face_node_j: np.ndarray
    int_face_area: np.ndarray
    bnd_face_node: np.ndarray
    bnd_face_neighbor: np.ndarray
    bnd_face_type: np.ndarray
    bnd_face_area: np.ndarray
    node_type: np.ndarray
    node_bnd_type: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    @property
    def n_int_faces(self) -> int:
        return len(self.int_face_node_i)

    @property
    def n_bnd_faces(self) -> int:
        return len(self.bnd_face_node)

    def to_torch(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """
        Конвертирует числовые поля сетки в torch-тензоры.

        Возвращает словарь ключей, готовых для использования в loss_computer:
            coords, cv_volumes, int_face_node_i, int_face_node_j,
            int_face_area, bnd_face_node, bnd_face_neighbor, bnd_face_area
        """
        return {
            'coords': torch.tensor(self.coords, dtype=dtype, device=device),
            'cv_volumes': torch.tensor(self.cv_volumes, dtype=dtype, device=device),
            'int_face_node_i': torch.tensor(self.int_face_node_i, dtype=torch.long, device=device),
            'int_face_node_j': torch.tensor(self.int_face_node_j, dtype=torch.long, device=device),
            'int_face_area': torch.tensor(self.int_face_area, dtype=dtype, device=device),
            'bnd_face_node': torch.tensor(self.bnd_face_node, dtype=torch.long, device=device),
            'bnd_face_neighbor': torch.tensor(self.bnd_face_neighbor, dtype=torch.long, device=device),
            'bnd_face_area': torch.tensor(self.bnd_face_area, dtype=dtype, device=device),
        }


class PINNDataset(Dataset):
    """
    Dataset для точек коллокации.

    Возвращает кортеж: (idx, x, y, sdf, boundary_types,
                        mask_u_fixed, mask_v_fixed, mask_p_fixed,
                        u_bc, v_bc, p_bc)
    Все тензоры хранятся в float32.

    Attributes:
        x: X-координаты (нормированные)
        y: Y-координаты (нормированные)
        sdf: Расстояние до стенки (нормированное)
        p_target: Целевое давление
        u_target: Целевая скорость U
        v_target: Целевая скорость V
        boundary_types: One-hot кодирование типов границ
        mask_u_fixed: Маска фиксированных компонент U
        mask_v_fixed: Маска фиксированных компонент V
        mask_p_fixed: Маска фиксированных компонент P
        u_bc: Значения BC для U
        v_bc: Значения BC для V
        p_bc: Значения BC для P
    """
    def __init__(self, points_file: str) -> None:
        """
        Инициализирует датасет из файла.

        Args:
            points_file: Путь к файлу с точками коллокации

        Raises:
            ValueError: Если файл имеет неверный формат
        """
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
                    # u_bc/v_bc для inlet остаются 0 до вызова set_inlet_values() из DataHandler

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
        self.source_indices = torch.arange(self.n_points, dtype=torch.long)
        logger.info(f"Типы границ в данных: {self.boundary_type_names}")
        if 'inlet' in self.boundary_type_names:
            logger.warning(
                "inlet-точки загружены: маски mask_u_fixed/mask_v_fixed выставлены, "
                "но u_bc/v_bc = 0 до вызова DataHandler.set_inlet_values(). "
                "Убедитесь, что DataHandler вызывает set_inlet_values() после загрузки метаданных."
            )

    def set_inlet_values(self, u_norm: float, v_norm: float) -> None:
        """
        Устанавливает значения на inlet после загрузки метаданных.

        Args:
            u_norm: Нормированная скорость U
            v_norm: Нормированная скорость V
        """
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
        """
        Возвращает количество точек в датасете.

        Returns:
            Количество точек
        """
        return self.n_points

    def __getitem__(self, idx: int) -> Tuple:
        """
        Возвращает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Кортеж с данными точки
        """
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

        Args:
            indices: Индексы точек для добавления

        Returns:
            Новый расширенный датасет
        """
        indices = indices.detach().cpu().long()
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
        if hasattr(self, 'source_indices'):
            new_dataset.source_indices = torch.cat(
                [self.source_indices, self.source_indices[indices]],
                dim=0
            )
        else:
            base = torch.arange(len(self.x), dtype=torch.long)
            new_dataset.source_indices = torch.cat([base, base[indices]], dim=0)
        new_dataset.n_points = len(new_dataset.x)
        return new_dataset


class DataHandler:
    """
    Загружает данные и метаданные, предоставляет DataLoader и методы для диагностики.

    Attributes:
        points_file: Путь к файлу с точками
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        dataset: Объект датасета
        dataloader: DataLoader для итерации по батчам
        mesh_data: FVM-сетка (MeshData) или None, если не загружена
    """
    def __init__(
        self,
        points_file: str,
        batch_size: int,
        shuffle: bool = True,
        mesh_file: Optional[str] = None,
    ) -> None:
        """
        Инициализирует обработчик данных.

        Args:
            points_file: Путь к файлу с точками коллокации
            batch_size: Размер батча для обучения
            shuffle: Перемешивать ли данные
            mesh_file: Путь к файлу FVM-сетки (.npz от mesh2d).
                       Если None — сетка не загружается.
        """
        self.points_file = points_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mesh_data: Optional[MeshData] = None

        self.dataset = PINNDataset(points_file)
        self.load_metadata()
        # После загрузки метаданных устанавливаем значения на inlet
        self.dataset.set_inlet_values(self.inlet_u_norm, self.inlet_v_norm)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        if mesh_file is not None:
            self.load_mesh(mesh_file)

    def load_metadata(self) -> None:
        """
        Загружает метаданные из файла _metadata.npy.

        Raises:
            FileNotFoundError: Если файл метаданных не найден
        """
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

    def load_mesh(self, mesh_file: str) -> None:
        """
        Загружает FVM-сетку из файла .npz (созданного mesh2d.Mesh2D.save()).

        Результат доступен через self.mesh_data (объект MeshData).
        Повторный вызов перезаписывает ранее загруженную сетку.

        Args:
            mesh_file: Путь к .npz файлу сетки

        Raises:
            FileNotFoundError: Если файл не найден
        """
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Файл сетки не найден: {mesh_file}")
        with np.load(mesh_file, allow_pickle=True) as data:
            self.mesh_data = MeshData(
                coords=np.array(data['coords']),
                triangles=np.array(data['triangles']),
                cv_volumes=np.array(data['cv_volumes']),
                int_face_node_i=np.array(data['int_face_node_i']),
                int_face_node_j=np.array(data['int_face_node_j']),
                int_face_area=np.array(data['int_face_area']),
                bnd_face_node=np.array(data['bnd_face_node']),
                bnd_face_neighbor=np.array(data['bnd_face_neighbor']),
                bnd_face_type=np.array(data['bnd_face_type']).astype(str),
                bnd_face_area=np.array(data['bnd_face_area']),
                node_type=np.array(data['node_type']).astype(str),
                node_bnd_type=np.array(data['node_bnd_type']).astype(str),
            )
        logger.info(
            "FVM-сетка загружена: %d узлов, %d треуг., %d вн. граней, %d гран. граней",
            self.mesh_data.n_nodes, self.mesh_data.n_triangles,
            self.mesh_data.n_int_faces, self.mesh_data.n_bnd_faces,
        )
        self._prepare_fvm_tensors()

    def _prepare_fvm_tensors(self) -> None:
        """
        Подготавливает нормированные тензоры (x, y, sdf, boundary_types) для FVM-шага.

        Координаты берутся из узлов сетки (mesh_data.coords / chord).
        SDF интерполируется из ближайших точек обучающего датасета.
        Типы границ строятся из mesh_data.node_bnd_type с тем же one-hot, что и датасет.

        Результат: self.fvm_x, self.fvm_y, self.fvm_sdf, self.fvm_bt — готовы для
        loss_computer.build_input_tensor() без участия обучающего датасета.
        """
        coords = self.mesh_data.coords          # (N, 2) физические
        N = len(coords)

        x_norm_np = (coords[:, 0] / self.chord).astype(np.float32)
        y_norm_np = (coords[:, 1] / self.chord).astype(np.float32)

        # SDF по ближайшему соседу из обучающего датасета
        x_data = self.dataset.x.numpy().flatten().astype(np.float32)
        y_data = self.dataset.y.numpy().flatten().astype(np.float32)
        sdf_data = self.dataset.sdf.numpy().flatten().astype(np.float32)

        tree = cKDTree(np.column_stack([x_data, y_data]))
        _, nn_idx = tree.query(np.column_stack([x_norm_np, y_norm_np]), workers=-1)
        sdf_norm_np = sdf_data[nn_idx]

        # One-hot типов границ (по именам из датасета)
        bt_names = self.dataset.boundary_type_names
        type_to_idx = self.dataset.type_to_idx
        bt_one_hot = np.zeros((N, len(bt_names)), dtype=np.float32)
        for i, t in enumerate(self.mesh_data.node_bnd_type):
            t_lower = str(t).lower()
            if t_lower in type_to_idx:
                bt_one_hot[i, type_to_idx[t_lower]] = 1.0

        self.fvm_x   = torch.tensor(x_norm_np,   dtype=torch.float32).view(-1, 1)
        self.fvm_y   = torch.tensor(y_norm_np,   dtype=torch.float32).view(-1, 1)
        self.fvm_sdf = torch.tensor(sdf_norm_np, dtype=torch.float32).view(-1, 1)
        self.fvm_bt  = torch.tensor(bt_one_hot,  dtype=torch.float32) if bt_names else None

        if N != self.n_points:
            logger.warning(
                "FVM-сетка (%d узлов) ≠ датасет (%d точек) — "
                "Sampler пропустил %d узлов. FVM использует координаты сетки напрямую.",
                N, self.n_points, abs(N - self.n_points),
            )
        logger.info("FVM-тензоры подготовлены: %d узлов", N)

    def get_dataloader(self) -> DataLoader:
        """
        Возвращает DataLoader для полного датасета.

        Returns:
            DataLoader для итерации по батчам
        """
        return self.dataloader

    def get_eval_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Возвращает DataLoader без перемешивания для диагностики и сохранения полей.

        Порядок элементов совпадает с порядком тензоров датасета, поэтому результаты
        можно безопасно индексировать масками из self.dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False
        )

    def get_boundary_eval_dataloader(
        self,
        type_names: Tuple[str, ...] = ('inlet', 'outlet'),
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """
        Возвращает DataLoader только для заданных типов границ.

        Используется для дешёвого мониторинга inlet/outlet вместо полного
        прохода по внутренним точкам датасета.
        """
        if self.dataset.boundary_types is None or not self.boundary_type_names:
            return DataLoader(Subset(self.dataset, []), batch_size=batch_size or self.batch_size)

        mask = torch.zeros(self.n_points, dtype=torch.bool)
        for name in type_names:
            idx = self.dataset.type_to_idx.get(name)
            if idx is not None and idx < self.dataset.boundary_types.shape[1]:
                mask |= self.dataset.boundary_types[:, idx] > 0.5

        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        return DataLoader(
            Subset(self.dataset, indices),
            batch_size=batch_size or self.batch_size,
            shuffle=False
        )

    @property
    def n_points(self) -> int:
        """
        Возвращает количество точек в датасете.

        Returns:
            Количество точек
        """
        return self.dataset.n_points

    @property
    def boundary_type_names(self) -> List[str]:
        """
        Возвращает список имён типов границ.

        Returns:
            Список названий типов границ
        """
        return self.dataset.boundary_type_names

    def add_points(self, indices: torch.Tensor) -> None:
        """
        Добавляет точки по индексам в датасет и обновляет DataLoader.

        Args:
            indices: Индексы точек для добавления
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

        Args:
            type_name: Название типа границы

        Returns:
            Словарь с ключами: 'x', 'y', 'u_target', 'v_target', 'p_target',
            'sdf', 'mask_u', 'mask_v', 'mask_p', 'u_bc', 'v_bc', 'p_bc'

        Raises:
            ValueError: Если тип границы не найден
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

    def get_points_by_type_and_coordinate(
        self,
        type_name: str,
        coord: str,
        value: float,
        tol: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Возвращает точки заданного типа, у которых координата coord близка к value.

        Args:
            type_name: Название типа границы
            coord: Координата ('x' или 'y')
            value: Целевое значение координаты
            tol: Допуск для сравнения

        Returns:
            Словарь с отфильтрованными данными

        Raises:
            ValueError: Если coord не 'x' или 'y'
        """
        pts = self.get_points_by_type(type_name)
        if coord == 'x':
            mask = torch.abs(pts['x'].flatten() - value) < tol
        elif coord == 'y':
            mask = torch.abs(pts['y'].flatten() - value) < tol
        else:
            raise ValueError("coord должен быть 'x' или 'y'")
        return {k: v[mask] for k, v in pts.items()}

    def interpolate_profile(
        self,
        x_target: float,
        u_vals: Optional[torch.Tensor] = None,
        y_grid: Optional[torch.Tensor] = None,
        num_points: int = 100,
        y_max_ratio: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интерполирует профиль u(y) в сечении x = x_target (нормированное).

        Args:
            x_target: Целевая X-координата (нормированная)
            u_vals: Значения скорости U (если None, используются начальные значения)
            y_grid: Сетка по Y для интерполяции
            num_points: Количество точек для равномерной сетки
            y_max_ratio: Доля от y_max для ограничения по Y

        Returns:
            Кортеж (y_coords, u_interp) – массивы numpy
        """
        tol = 0.05 * (self.x_max - self.x_min)
        # Выбираем точки, близкие к x_target
        mask_x = torch.abs(self.dataset.x.flatten() - x_target) < tol
        # Ограничение по y
        y_limit = y_max_ratio * self.y_max
        mask_y = self.dataset.y.flatten() < y_limit
        mask = mask_x & mask_y

        if mask.sum() == 0:
            logger.warning(f"Нет точек в сечении x={x_target} (tol={tol:.4f}), пропускаем интерполяцию.")
            if y_grid is None:
                y_grid = np.linspace(self.y_min, y_limit, num_points)
            return y_grid, np.full_like(y_grid, np.nan)

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
