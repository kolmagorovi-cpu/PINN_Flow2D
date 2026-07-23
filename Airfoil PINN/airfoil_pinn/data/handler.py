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
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial import cKDTree

from airfoil_pinn.app.paths import (
    GEOM_CONFIG_PATH,
    GEOMETRY_DOMAINS_DIR,
    GEOMETRY_METADATA_DIR,
    PATHS,
    SAMPLING_COLLOCATION_DIR,
    resolve_project_path,
)
from airfoil_pinn.core.type_hints import PointDtypeDict
from airfoil_pinn.data.normalization import (
    ReferenceScales,
    extract_geometry_reference,
    load_companion_metadata,
)
from airfoil_pinn.data.sdf_diagnostics import (
    estimate_boundary_layer_delta,
    resolve_boundary_layer_model,
)

logger = logging.getLogger(__name__)

# Обработчик данных ищет артефакты через workspace. Старые пути ниже
# используются только как fallback при открытии пользовательских архивов.
PROJECT_ROOT = str(PATHS.project_root)

__all__ = [
    "DataHandler",
    "MeshData",
    "MeshPointAdapter",
    "PINNDataset",
]


@dataclass
class MeshPointAdapter:
    """Минимальная точка для Mesh2D, восстановленная из обучающего датасета."""
    id: int
    x: float
    y: float
    boundary_type: str
    on_airfoil: bool = False
    on_boundary: bool = False
    segment_id: int = -1


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
        points_file = str(resolve_project_path(points_file))
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
        self.on_boundary = torch.zeros(len(self.x), dtype=torch.bool)
        self.on_airfoil = torch.zeros(len(self.x), dtype=torch.bool)
        self.segment_id = torch.full((len(self.x),), -1, dtype=torch.long)

        if 'on_boundary' in data.dtype.names:
            self.on_boundary = torch.tensor(
                np.ascontiguousarray(data['on_boundary']),
                dtype=torch.bool
            )
        if 'on_airfoil' in data.dtype.names:
            self.on_airfoil = torch.tensor(
                np.ascontiguousarray(data['on_airfoil']),
                dtype=torch.bool
            )
        if 'segment_id' in data.dtype.names:
            self.segment_id = torch.tensor(
                np.ascontiguousarray(data['segment_id']),
                dtype=torch.long
            )

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

                if bt_str in ('inlet', 'velocity_dirichlet'):
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # u_bc/v_bc остаются 0 до вызова set_inlet_values() из DataHandler

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

            if 'on_boundary' not in data.dtype.names:
                interior_idx = self.type_to_idx.get('interior')
                if interior_idx is not None:
                    self.on_boundary = self.boundary_types[:, interior_idx] <= 0.5
                else:
                    self.on_boundary = torch.ones(len(self.x), dtype=torch.bool)

        self.n_points = len(self.x)
        self.source_indices = torch.arange(self.n_points, dtype=torch.long)
        logger.info(f"Типы границ в данных: {self.boundary_type_names}")
        if 'inlet' in self.boundary_type_names:
            logger.debug(
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

        target_types = ('inlet', 'farfield', 'velocity_dirichlet')
        applied = False
        for type_name in target_types:
            type_idx = self.type_to_idx.get(type_name)
            if type_idx is None:
                continue

            if type_idx >= self.boundary_types.shape[1]:
                logger.error(
                    f"Индекс {type_name} ({type_idx}) выходит за пределы "
                    f"размерности boundary_types ({self.boundary_types.shape[1]})"
                )
                continue

            mask = self.boundary_types[:, type_idx] > 0.5
            if bool(mask.any()):
                self.u_bc[mask] = u_norm
                self.v_bc[mask] = v_norm
                applied = True

        if not applied:
            logger.warning("Типы 'inlet/farfield/velocity_dirichlet' отсутствуют в данных.")

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
                     'u_bc', 'v_bc', 'p_bc', 'boundary_types',
                     'on_boundary', 'on_airfoil', 'segment_id',
                     'mesh_x_phys', 'mesh_y_phys']:
            old = getattr(self, attr, None)
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
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        """
        Инициализирует обработчик данных.

        Args:
            points_file: Путь к файлу с точками коллокации
            batch_size: Размер батча для обучения
            shuffle: Перемешивать ли данные
            mesh_file: Путь к файлу FVM-сетки (.npz от mesh2d).
                       Если None — сетка не загружается.
            num_workers: число worker-процессов DataLoader.
            pin_memory: закреплять CPU-память для ускорения копирования на CUDA.
            persistent_workers: держать worker'ы между эпохами.
            prefetch_factor: число заранее подготовленных batch на worker.
        """
        self.points_file = str(resolve_project_path(points_file))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = (
            bool(persistent_workers)
            if persistent_workers is not None
            else self.num_workers > 0
        )
        self.prefetch_factor = (
            max(1, int(prefetch_factor))
            if prefetch_factor is not None
            else 2
        )
        self.mesh_data: Optional[MeshData] = None
        self.mesh_file = str(resolve_project_path(mesh_file)) if mesh_file is not None else None
        self.mesh_domain_coords: Optional[np.ndarray] = None
        self.mesh_airfoil_coords: Optional[np.ndarray] = None
        self.sampler_metadata: Optional[Dict[str, Any]] = None
        self.geometry_reference: Optional[Dict[str, Any]] = None

        self.dataset = PINNDataset(self.points_file)
        self.load_metadata()
        self._load_sampler_point_metadata()
        self._reconcile_geometry_normalization()
        self.load_mesh_geometry_from_geom_config()
        # После загрузки метаданных устанавливаем значения на inlet
        self.dataset.set_inlet_values(self.inlet_u_norm, self.inlet_v_norm)
        self.dataloader = self._make_dataloader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        if self.mesh_file is not None:
            self.load_mesh(self.mesh_file)

    def _make_dataloader(
        self,
        dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Создаёт DataLoader с едиными настройками multiprocessing/pinned memory."""
        kwargs = {
            'batch_size': batch_size or self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs['persistent_workers'] = self.persistent_workers
            kwargs['prefetch_factor'] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def load_metadata(self) -> None:
        """
        Load and validate the versioned CFD normalization contract.

        Raises:
            FileNotFoundError: Если файл метаданных не найден
            ValueError: If physical reference scales are missing or invalid.
        """
        meta_path = self.points_file.replace('.npy', '_metadata.npy')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Файл метаданных {meta_path} не найден")
        meta = np.load(meta_path, allow_pickle=True).item()
        self.initial_metadata = dict(meta)
        self.fluid_name = meta.get('fluid_name')
        self.conditions_name = meta.get('conditions_name')
        self.reference_scales = ReferenceScales.from_metadata(meta)
        self.chord = self.reference_scales.chord
        self.U_inf = self.reference_scales.velocity
        self.p_ref = self.reference_scales.static_pressure
        self.rho = self.reference_scales.density
        self.mu = self.reference_scales.dynamic_viscosity
        self.Re = self.reference_scales.reynolds

        stored_re = meta.get('Re')
        if stored_re is not None:
            try:
                stored_re_value = float(stored_re)
            except (TypeError, ValueError):
                stored_re_value = float('nan')
            if (
                np.isfinite(stored_re_value)
                and stored_re_value > 0.0
                and not np.isclose(stored_re_value, self.Re, rtol=1e-8, atol=0.0)
            ):
                logger.warning(
                    "Stored Re_c=%.9g disagrees with reference scales "
                    "(Re_c=%.9g); using the value recomputed from rho*U*c/mu.",
                    stored_re_value,
                    self.Re,
                )
        self.sdf_norm_factor = float(meta.get('sdf_norm_factor', self.chord))
        self.delta_bl_99_est = meta.get('delta_bl_99_est')
        self.delta_bl_99_est_norm = meta.get('delta_bl_99_est_norm')
        self.boundary_layer_model = str(meta.get('delta_bl_model', 'laminar'))
        self.geometry_reference = meta.get('geometry_reference')
        self.inlet_u_norm = meta.get('inlet_u_norm', 1.0)
        self.inlet_v_norm = meta.get('inlet_v_norm', 0.0)
        # Границы области (нормированные)
        self.x_min = self.dataset.x.min().item()
        self.x_max = self.dataset.x.max().item()
        self.y_min = self.dataset.y.min().item()
        self.y_max = self.dataset.y.max().item()
        logger.info(
            "Метаданные: chord=%.4f м, U_inf=%.4f м/с, Re=%.2e, "
            "normalization=%s:%d, fingerprint=%s",
            self.chord,
            self.U_inf,
            self.Re,
            self.reference_scales.to_metadata()['schema'],
            self.reference_scales.to_metadata()['version'],
            self.reference_scales.fingerprint()[:12],
        )
        logger.info(f"Вход: u_norm={self.inlet_u_norm:.3f}, v_norm={self.inlet_v_norm:.3f}")
        self.dataset.mesh_x_phys = (self.dataset.x.double().flatten() * float(self.chord)).clone()
        self.dataset.mesh_y_phys = (self.dataset.y.double().flatten() * float(self.chord)).clone()

    def _load_sampler_point_metadata(self) -> None:
        """
        Дополняет initial.npy геометрическими флагами из collocation_points.npy.

        workspace/initial/fields/initial.npy хранит нормированные поля, но обычно не хранит
        on_airfoil/segment_id. Эти флаги нужны только для корректной перестройки
        boundary-fitted Mesh2D после adaptive RAR.
        """
        project_dir = PROJECT_ROOT
        candidates = [
            os.path.join(SAMPLING_COLLOCATION_DIR, "collocation_points.npy"),
            os.path.join(project_dir, "Sampler", "Data", "collocation_points.npy"),
        ]

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                arr = np.load(path, allow_pickle=True)
            except Exception as exc:
                logger.debug("Не удалось прочитать %s: %s", path, exc)
                continue

            names = arr.dtype.names or ()
            required = {'x', 'y', 'on_boundary', 'on_airfoil', 'segment_id'}
            if not required.issubset(names) or len(arr) != self.dataset.n_points:
                continue

            x_norm = np.asarray(arr['x'], dtype=np.float64) / float(self.chord)
            y_norm = np.asarray(arr['y'], dtype=np.float64) / float(self.chord)
            ds_x = self.dataset.x.numpy().reshape(-1)
            ds_y = self.dataset.y.numpy().reshape(-1)
            if not (np.allclose(x_norm, ds_x, rtol=1e-6, atol=1e-8)
                    and np.allclose(y_norm, ds_y, rtol=1e-6, atol=1e-8)):
                logger.debug("Sampler metadata skipped: coordinates do not match %s", path)
                continue

            self.dataset.on_boundary = torch.tensor(
                np.ascontiguousarray(arr['on_boundary']),
                dtype=torch.bool
            )
            self.dataset.on_airfoil = torch.tensor(
                np.ascontiguousarray(arr['on_airfoil']),
                dtype=torch.bool
            )
            self.dataset.segment_id = torch.tensor(
                np.ascontiguousarray(arr['segment_id']),
                dtype=torch.long
            )
            self.dataset.mesh_x_phys = torch.tensor(
                np.ascontiguousarray(arr['x']),
                dtype=torch.float64
            )
            self.dataset.mesh_y_phys = torch.tensor(
                np.ascontiguousarray(arr['y']),
                dtype=torch.float64
            )
            self.sampler_metadata = load_companion_metadata(path)

            airfoil_mask = np.asarray(arr['on_airfoil'], dtype=bool)
            if np.any(airfoil_mask):
                foil = np.column_stack([arr['x'][airfoil_mask], arr['y'][airfoil_mask]])
                foil_closed = bool(
                    isinstance(self.sampler_metadata, dict)
                    and self.sampler_metadata.get('foil_closed', False)
                )
                if foil_closed and not np.allclose(foil[0], foil[-1], atol=1e-12):
                    foil = np.vstack([foil, foil[0]])
                self.mesh_airfoil_coords = np.asarray(foil, dtype=np.float64)

            logger.info("Геометрические флаги точек загружены из %s", path)
            return

        if self.dataset.boundary_types is not None and 'wall' in self.dataset.type_to_idx:
            wall_idx = self.dataset.type_to_idx['wall']
            self.dataset.on_airfoil = self.dataset.boundary_types[:, wall_idx] > 0.5
            logger.info(
                "on_airfoil восстановлен эвристически по boundary_type='wall' "
                "(collocation_points.npy не найден или не совпал)."
            )

    def _reconcile_geometry_normalization(self) -> None:
        """Align legacy normalized fields with the sampled geometry reference."""
        reference = extract_geometry_reference(self.sampler_metadata)
        if reference is None:
            return

        new_chord = float(reference['chord'])
        old_chord = float(self.chord)
        old_sdf_factor = float(self.sdf_norm_factor)
        self.geometry_reference = dict(reference)

        if not np.isclose(old_chord, new_chord, rtol=1e-10, atol=1e-12):
            physical_x = getattr(self.dataset, 'mesh_x_phys', None)
            physical_y = getattr(self.dataset, 'mesh_y_phys', None)
            if physical_x is None or physical_y is None:
                logger.warning(
                    "Хорда initial metadata (%.6g м) отличается от sampler metadata (%.6g м), "
                    "но физические координаты недоступны; runtime-нормировка не изменена.",
                    old_chord,
                    new_chord,
                )
                return

            self.dataset.x = (
                physical_x.reshape(-1, 1) / new_chord
            ).to(dtype=self.dataset.x.dtype)
            self.dataset.y = (
                physical_y.reshape(-1, 1) / new_chord
            ).to(dtype=self.dataset.y.dtype)
            logger.warning(
                "Исправлена устаревшая нормировка координат: chord %.6g -> %.6g м (%s).",
                old_chord,
                new_chord,
                reference.get('source', 'sampler metadata'),
            )

        sdf_scale = old_sdf_factor / new_chord
        if not np.isclose(sdf_scale, 1.0, rtol=1e-10, atol=1e-12):
            self.dataset.sdf = self.dataset.sdf * float(sdf_scale)
            logger.warning(
                "Исправлена нормировка SDF: factor %.6g -> %.6g м.",
                old_sdf_factor,
                new_chord,
            )

        self.chord = new_chord
        self.sdf_norm_factor = new_chord
        self.reference_scales = self.reference_scales.with_chord(new_chord)
        self.Re = self.reference_scales.reynolds
        normal_layer = (
            self.sampler_metadata.get('normal_layer')
            if isinstance(self.sampler_metadata, dict)
            else None
        )
        sampled_model = normal_layer.get('cf_model') if isinstance(normal_layer, dict) else None
        self.boundary_layer_model = resolve_boundary_layer_model(
            self.Re,
            str(sampled_model or self.boundary_layer_model),
        )
        self.delta_bl_99_est, self.delta_bl_99_est_norm = estimate_boundary_layer_delta(
            self.chord,
            self.Re,
            model=self.boundary_layer_model,
        )
        self.x_min = self.dataset.x.min().item()
        self.x_max = self.dataset.x.max().item()
        self.y_min = self.dataset.y.min().item()
        self.y_max = self.dataset.y.max().item()

        sampled_re = normal_layer.get('re_chord') if isinstance(normal_layer, dict) else None
        if sampled_re is not None and float(sampled_re) > 0.0:
            relative_error = abs(float(sampled_re) - self.Re) / self.Re
            if relative_error > 0.01:
                logger.warning(
                    "Normal-layer точки построены для Re_c=%.6g, текущие физические условия "
                    "дают Re_c=%.6g (расхождение %.1f%%).",
                    float(sampled_re),
                    self.Re,
                    100.0 * relative_error,
                )

        sampled_delta = normal_layer.get('delta') if isinstance(normal_layer, dict) else None
        if (
            sampled_delta is not None
            and float(sampled_delta) > 0.0
            and self.delta_bl_99_est is not None
        ):
            delta_error = abs(float(sampled_delta) - self.delta_bl_99_est) / self.delta_bl_99_est
            if delta_error > 0.01:
                logger.warning(
                    "Normal-layer сетка покрывает delta/c=%.6g, текущий reference delta/c=%.6g "
                    "(расхождение %.1f%%). Для полного согласования сетку нужно пересоздать.",
                    float(sampled_delta) / self.chord,
                    self.delta_bl_99_est_norm,
                    100.0 * delta_error,
                )

        logger.info(
            "Единые reference scales: chord=%.6g м, Re_c=%.6g, delta_99/c=%.6g (%s)",
            self.chord,
            self.Re,
            self.delta_bl_99_est_norm or float('nan'),
            self.boundary_layer_model,
        )

    def load_mesh_geometry_from_geom_config(self, config_path: str = GEOM_CONFIG_PATH) -> bool:
        """
        Загружает контуры домена/профиля для последующих rebuild Mesh2D.

        Новые mesh2d.npz сохраняют эти контуры прямо в файле. Для старых сеток
        остаётся fallback через geom_config.yaml и сохранённые контуры геометрии.
        """
        if self.mesh_domain_coords is not None:
            return True

        project_dir = PROJECT_ROOT
        cfg_path = config_path
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(project_dir, cfg_path)
        if not os.path.exists(cfg_path):
            return False

        try:
            import yaml
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.debug("Не удалось прочитать geom_config.yaml: %s", exc)
            return False

        geom_name = cfg.get('geom_name')
        if not geom_name:
            return False

        def _load_coords(path_base: str) -> Optional[np.ndarray]:
            for ext in (".npy", ".txt", ".csv"):
                path = path_base + ext
                if not os.path.exists(path):
                    continue
                if ext == ".npy":
                    return np.asarray(np.load(path), dtype=np.float64)
                return np.asarray(np.loadtxt(path, delimiter=',' if ext == ".csv" else None), dtype=np.float64)
            return None

        search_pairs = [
            (GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR),
            (
                os.path.join(project_dir, "Geometry", "Data"),
                os.path.join(project_dir, "Geometry", "Metadata"),
            ),
        ]
        domain = None
        foil = None
        for data_dir, meta_dir in search_pairs:
            domain = _load_coords(os.path.join(data_dir, f"{geom_name}_domain"))
            foil = _load_coords(os.path.join(meta_dir, f"{geom_name}_foil"))
            if foil is None:
                foil = _load_coords(os.path.join(data_dir, f"{geom_name}_foil"))
            if domain is not None:
                break

        if domain is None or len(domain) < 2:
            return False
        if len(domain) > 2 and not np.allclose(domain[0], domain[-1], atol=1e-9):
            domain = np.vstack([domain, domain[0]])

        self.mesh_domain_coords = domain
        config_foil = foil if foil is not None and len(foil) >= 2 else None
        if self.mesh_airfoil_coords is None:
            self.mesh_airfoil_coords = config_foil
        elif config_foil is not None:
            sampler_chord = float(np.ptp(self.mesh_airfoil_coords[:, 0]))
            config_chord = float(np.ptp(config_foil[:, 0]))
            if not np.isclose(sampler_chord, config_chord, rtol=0.02, atol=1e-12):
                logger.warning(
                    "Профиль из текущего geom_config имеет хорду %.6g м, а collocation-точки "
                    "%.6g м; для диагностики сохранён контур collocation-точек.",
                    config_chord,
                    sampler_chord,
                )
        logger.info(
            "Геометрия для FVM rebuild загружена: domain=%d, foil=%s",
            len(domain),
            len(self.mesh_airfoil_coords) if self.mesh_airfoil_coords is not None else "нет",
        )
        return True

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
        mesh_file = str(resolve_project_path(mesh_file))
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Файл сетки не найден: {mesh_file}")
        with np.load(mesh_file, allow_pickle=True) as data:
            if 'domain_coords' in data.files and len(data['domain_coords']) >= 2:
                self.mesh_domain_coords = np.asarray(data['domain_coords'], dtype=np.float64)
            if 'airfoil_coords' in data.files and len(data['airfoil_coords']) >= 2:
                self.mesh_airfoil_coords = np.asarray(data['airfoil_coords'], dtype=np.float64)
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

    def set_mesh_from_mesh2d(self, mesh) -> None:
        """Обновляет MeshData и FVM-тензоры из уже построенного Mesh2D."""
        ifaces = mesh.internal_faces
        bfaces = mesh.boundary_faces
        self.mesh_data = MeshData(
            coords=np.asarray(mesh.coords, dtype=np.float64),
            triangles=np.asarray(mesh.triangles, dtype=np.int32),
            cv_volumes=np.asarray(mesh.cv_volumes, dtype=np.float64),
            int_face_node_i=np.array([f.node_i for f in ifaces], dtype=np.int32),
            int_face_node_j=np.array([f.node_j for f in ifaces], dtype=np.int32),
            int_face_area=np.array([f.area_vector for f in ifaces], dtype=np.float64)
            if ifaces else np.empty((0, 2), dtype=np.float64),
            bnd_face_node=np.array([f.node for f in bfaces], dtype=np.int32),
            bnd_face_neighbor=np.array([f.neighbor_node for f in bfaces], dtype=np.int32),
            bnd_face_type=np.array([f.boundary_type for f in bfaces], dtype=object),
            bnd_face_area=np.array([f.area_vector for f in bfaces], dtype=np.float64)
            if bfaces else np.empty((0, 2), dtype=np.float64),
            node_type=np.array([nd.node_type for nd in mesh.nodes], dtype=object),
            node_bnd_type=np.array([nd.boundary_type for nd in mesh.nodes], dtype=object),
        )
        self._prepare_fvm_tensors()

    def _dataset_boundary_type_strings(self) -> List[str]:
        ds = self.dataset
        if ds.boundary_types is None or not ds.boundary_type_names:
            return ["interior"] * ds.n_points
        idx = torch.argmax(ds.boundary_types, dim=1).cpu().numpy()
        names = ds.boundary_type_names
        return [names[int(i)] if int(i) < len(names) else "interior" for i in idx]

    def mesh_points_from_dataset(self) -> List[MeshPointAdapter]:
        """Восстанавливает список точек Mesh2D из текущего обучающего датасета."""
        ds = self.dataset
        if hasattr(ds, 'mesh_x_phys') and hasattr(ds, 'mesh_y_phys'):
            x_phys = ds.mesh_x_phys.cpu().numpy().reshape(-1)
            y_phys = ds.mesh_y_phys.cpu().numpy().reshape(-1)
        else:
            x_phys = ds.x.cpu().numpy().reshape(-1).astype(np.float64) * float(self.chord)
            y_phys = ds.y.cpu().numpy().reshape(-1).astype(np.float64) * float(self.chord)
        bt = self._dataset_boundary_type_strings()
        on_boundary = getattr(ds, 'on_boundary', torch.zeros(ds.n_points, dtype=torch.bool)).cpu().numpy()
        on_airfoil = getattr(ds, 'on_airfoil', torch.zeros(ds.n_points, dtype=torch.bool)).cpu().numpy()
        segment_id = getattr(ds, 'segment_id', torch.full((ds.n_points,), -1, dtype=torch.long)).cpu().numpy()

        return [
            MeshPointAdapter(
                id=i,
                x=float(x_phys[i]),
                y=float(y_phys[i]),
                boundary_type=str(bt[i]),
                on_airfoil=bool(on_airfoil[i]),
                on_boundary=bool(on_boundary[i]),
                segment_id=int(segment_id[i]),
            )
            for i in range(ds.n_points)
        ]

    def build_mesh_from_current_points(self, verbose: bool = False):
        """Строит новую Mesh2D по текущим точкам датасета без графиков и записи на диск."""
        if self.mesh_domain_coords is None:
            self.load_mesh_geometry_from_geom_config()
        if self.mesh_domain_coords is None:
            raise RuntimeError(
                "Нет domain_coords для перестройки FVM-сетки. "
                "Пересоздайте mesh2d.npz через S2DR или проверьте geom_config.yaml."
            )

        from airfoil_pinn.sampling.mesh import Mesh2D

        mesh = Mesh2D(
            self.mesh_points_from_dataset(),
            domain_coords=self.mesh_domain_coords,
            airfoil_coords=self.mesh_airfoil_coords,
        )
        mesh.build(verbose=verbose)
        return mesh

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
        return self._make_dataloader(
            self.dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
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
            return self._make_dataloader(
                Subset(self.dataset, []),
                batch_size=batch_size or self.batch_size,
                shuffle=False,
            )

        mask = torch.zeros(self.n_points, dtype=torch.bool)
        for name in type_names:
            idx = self.dataset.type_to_idx.get(name)
            if idx is not None and idx < self.dataset.boundary_types.shape[1]:
                mask |= self.dataset.boundary_types[:, idx] > 0.5

        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        return self._make_dataloader(
            Subset(self.dataset, indices),
            batch_size=batch_size or self.batch_size,
            shuffle=False,
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
        self.dataloader = self._make_dataloader(self.dataset, shuffle=self.shuffle)
        logger.info(f"Датасет расширен: добавлено {len(indices)} точек, всего {self.n_points}")

    def add_interior_points(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        sdf_norm: np.ndarray,
    ) -> int:
        """
        Добавляет новые interior-точки с уникальными координатами.

        Целевые p/u/v для справочных полей берутся у ближайшей старой точки,
        а маски hard-BC остаются выключенными.
        """
        x_norm = np.asarray(x_norm, dtype=np.float32).reshape(-1)
        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
        sdf_norm = np.asarray(sdf_norm, dtype=np.float32).reshape(-1)
        n_new = int(len(x_norm))
        if n_new == 0:
            return 0
        if not (len(y_norm) == n_new and len(sdf_norm) == n_new):
            raise ValueError("x_norm, y_norm и sdf_norm должны иметь одинаковую длину")

        ds = self.dataset
        if ds.boundary_types is None or 'interior' not in ds.type_to_idx:
            raise RuntimeError(
                "Adaptive RAR требует существующий boundary_type='interior', "
                "иначе изменится размер входа модели."
            )

        old_n = ds.n_points
        old_xy = np.column_stack([
            ds.x.cpu().numpy().reshape(-1),
            ds.y.cpu().numpy().reshape(-1),
        ])
        _, nn_idx = cKDTree(old_xy).query(np.column_stack([x_norm, y_norm]), workers=-1)
        nn_idx_t = torch.tensor(nn_idx, dtype=torch.long)

        ds.x = torch.cat([ds.x, torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.y = torch.cat([ds.y, torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.sdf = torch.cat([ds.sdf, torch.tensor(sdf_norm, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.mesh_x_phys = torch.cat(
            [ds.mesh_x_phys, torch.tensor(x_norm * float(self.chord), dtype=torch.float64)],
            dim=0,
        )
        ds.mesh_y_phys = torch.cat(
            [ds.mesh_y_phys, torch.tensor(y_norm * float(self.chord), dtype=torch.float64)],
            dim=0,
        )

        ds.p_target = torch.cat([ds.p_target, ds.p_target[nn_idx_t].clone()], dim=0)
        ds.u_target = torch.cat([ds.u_target, ds.u_target[nn_idx_t].clone()], dim=0)
        ds.v_target = torch.cat([ds.v_target, ds.v_target[nn_idx_t].clone()], dim=0)

        ds.mask_u_fixed = torch.cat([ds.mask_u_fixed, torch.zeros(n_new, dtype=torch.bool)], dim=0)
        ds.mask_v_fixed = torch.cat([ds.mask_v_fixed, torch.zeros(n_new, dtype=torch.bool)], dim=0)
        ds.mask_p_fixed = torch.cat([ds.mask_p_fixed, torch.zeros(n_new, dtype=torch.bool)], dim=0)
        ds.u_bc = torch.cat([ds.u_bc, torch.zeros(n_new, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.v_bc = torch.cat([ds.v_bc, torch.zeros(n_new, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.p_bc = torch.cat([ds.p_bc, torch.zeros(n_new, dtype=torch.float32).view(-1, 1)], dim=0)

        bt_new = torch.zeros(n_new, ds.boundary_types.shape[1], dtype=torch.float32)
        bt_new[:, ds.type_to_idx['interior']] = 1.0
        ds.boundary_types = torch.cat([ds.boundary_types, bt_new], dim=0)

        ds.on_boundary = torch.cat([ds.on_boundary, torch.zeros(n_new, dtype=torch.bool)], dim=0)
        ds.on_airfoil = torch.cat([ds.on_airfoil, torch.zeros(n_new, dtype=torch.bool)], dim=0)
        ds.segment_id = torch.cat([ds.segment_id, torch.full((n_new,), -1, dtype=torch.long)], dim=0)
        ds.source_indices = torch.cat(
            [ds.source_indices, torch.arange(old_n, old_n + n_new, dtype=torch.long)],
            dim=0,
        )
        ds.n_points = old_n + n_new

        self.dataloader = self._make_dataloader(self.dataset, shuffle=self.shuffle)
        logger.info("Adaptive RAR: добавлено %d новых interior-точек, всего %d", n_new, self.n_points)
        return n_new

    def get_state(self) -> Dict[str, Any]:
        """Возвращает состояние датасета/сетки для корректного resume."""
        ds = self.dataset
        tensor_attrs = [
            'x', 'y', 'sdf', 'p_target', 'u_target', 'v_target',
            'boundary_types', 'mask_u_fixed', 'mask_v_fixed', 'mask_p_fixed',
            'u_bc', 'v_bc', 'p_bc', 'on_boundary', 'on_airfoil',
            'segment_id', 'source_indices', 'mesh_x_phys', 'mesh_y_phys',
        ]
        dataset_state = {
            attr: getattr(ds, attr, None).detach().cpu().clone()
            if torch.is_tensor(getattr(ds, attr, None)) else None
            for attr in tensor_attrs
        }
        dataset_state.update({
            'boundary_type_names': list(ds.boundary_type_names),
            'type_to_idx': dict(ds.type_to_idx),
            'n_points': int(ds.n_points),
        })

        mesh_state = None
        if self.mesh_data is not None:
            mesh_state = {
                'coords': self.mesh_data.coords,
                'triangles': self.mesh_data.triangles,
                'cv_volumes': self.mesh_data.cv_volumes,
                'int_face_node_i': self.mesh_data.int_face_node_i,
                'int_face_node_j': self.mesh_data.int_face_node_j,
                'int_face_area': self.mesh_data.int_face_area,
                'bnd_face_node': self.mesh_data.bnd_face_node,
                'bnd_face_neighbor': self.mesh_data.bnd_face_neighbor,
                'bnd_face_type': self.mesh_data.bnd_face_type,
                'bnd_face_area': self.mesh_data.bnd_face_area,
                'node_type': self.mesh_data.node_type,
                'node_bnd_type': self.mesh_data.node_bnd_type,
            }

        return {
            'dataset': dataset_state,
            'mesh_data': mesh_state,
            'mesh_domain_coords': self.mesh_domain_coords,
            'mesh_airfoil_coords': self.mesh_airfoil_coords,
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Восстанавливает датасет/состояние сетки из чекпоинта."""
        if not state:
            return

        ds_state = state.get('dataset') or {}
        ds = self.dataset
        for attr, value in ds_state.items():
            if attr in ('boundary_type_names', 'type_to_idx', 'n_points'):
                continue
            if torch.is_tensor(value):
                setattr(ds, attr, value.detach().cpu().clone())
            elif value is not None:
                setattr(ds, attr, value)

        ds.boundary_type_names = list(ds_state.get('boundary_type_names', ds.boundary_type_names))
        ds.type_to_idx = dict(ds_state.get('type_to_idx', ds.type_to_idx))
        ds.n_points = int(ds_state.get('n_points', len(ds.x)))

        if getattr(ds, 'source_indices', None) is None:
            ds.source_indices = torch.arange(ds.n_points, dtype=torch.long)
        if getattr(ds, 'mesh_x_phys', None) is None:
            ds.mesh_x_phys = (ds.x.double().flatten() * float(self.chord)).clone()
        if getattr(ds, 'mesh_y_phys', None) is None:
            ds.mesh_y_phys = (ds.y.double().flatten() * float(self.chord)).clone()

        self.mesh_domain_coords = state.get('mesh_domain_coords', self.mesh_domain_coords)
        self.mesh_airfoil_coords = state.get('mesh_airfoil_coords', self.mesh_airfoil_coords)

        mesh_state = state.get('mesh_data')
        if mesh_state is not None:
            self.mesh_data = MeshData(**mesh_state)
            self._prepare_fvm_tensors()
        else:
            self.mesh_data = None
            for attr in ('fvm_x', 'fvm_y', 'fvm_sdf', 'fvm_bt'):
                if hasattr(self, attr):
                    delattr(self, attr)

        self.dataloader = self._make_dataloader(self.dataset, shuffle=self.shuffle)
        logger.info("DataHandler восстановлен из чекпоинта: %d точек", self.n_points)

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
