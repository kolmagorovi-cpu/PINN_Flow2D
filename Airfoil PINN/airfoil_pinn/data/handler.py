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
import random
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset
import logging
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial import cKDTree

from airfoil_pinn.app.paths import (
    PATHS,
    SAMPLING_COLLOCATION_DIR,
    resolve_project_path,
)
from airfoil_pinn.core.type_hints import PointDtypeDict
from airfoil_pinn.data.batch import PINNBatch, collate_pinn_batches
from airfoil_pinn.data.flow import (
    FlowCase,
    FlowCaseBatch,
    flow_cases_from_metadata,
)
from airfoil_pinn.data.normalization import (
    ReferenceScales,
    extract_geometry_reference,
    load_companion_metadata,
)
from airfoil_pinn.geometry.sdf import PolylineDistanceField
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


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by PyTorch.

    The function intentionally lives at module scope so it remains picklable
    with the Windows ``spawn`` multiprocessing method.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        int_quad_coords:  (Q, 2)  midpoint прямых сегментов дуальных граней
        int_quad_area:    (Q, 2)  area vector каждого сегмента
        int_quad_node_i/j:(Q,)    соседние контрольные объёмы сегмента
        bnd_quad_coords:  (Fb, 2) midpoint граничных полуграней
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
    cell_nodes: Optional[np.ndarray] = None
    cell_offsets: Optional[np.ndarray] = None
    quads: Optional[np.ndarray] = None
    int_quad_coords: Optional[np.ndarray] = None
    int_quad_area: Optional[np.ndarray] = None
    int_quad_node_i: Optional[np.ndarray] = None
    int_quad_node_j: Optional[np.ndarray] = None
    int_quad_face: Optional[np.ndarray] = None
    bnd_quad_coords: Optional[np.ndarray] = None
    mesh_schema_version: int = 1

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

    @property
    def n_int_quad_points(self) -> int:
        return (
            0
            if self.int_quad_coords is None
            else len(self.int_quad_coords)
        )

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
        tensors = {
            'coords': torch.tensor(self.coords, dtype=dtype, device=device),
            'cv_volumes': torch.tensor(self.cv_volumes, dtype=dtype, device=device),
            'int_face_node_i': torch.tensor(self.int_face_node_i, dtype=torch.long, device=device),
            'int_face_node_j': torch.tensor(self.int_face_node_j, dtype=torch.long, device=device),
            'int_face_area': torch.tensor(self.int_face_area, dtype=dtype, device=device),
            'bnd_face_node': torch.tensor(self.bnd_face_node, dtype=torch.long, device=device),
            'bnd_face_neighbor': torch.tensor(self.bnd_face_neighbor, dtype=torch.long, device=device),
            'bnd_face_area': torch.tensor(self.bnd_face_area, dtype=dtype, device=device),
        }
        if self.int_quad_coords is not None:
            tensors.update({
                'int_quad_coords': torch.tensor(
                    self.int_quad_coords,
                    dtype=dtype,
                    device=device,
                ),
                'int_quad_area': torch.tensor(
                    self.int_quad_area,
                    dtype=dtype,
                    device=device,
                ),
                'int_quad_node_i': torch.tensor(
                    self.int_quad_node_i,
                    dtype=torch.long,
                    device=device,
                ),
                'int_quad_node_j': torch.tensor(
                    self.int_quad_node_j,
                    dtype=torch.long,
                    device=device,
                ),
                'bnd_quad_coords': torch.tensor(
                    self.bnd_quad_coords,
                    dtype=dtype,
                    device=device,
                ),
            })
        return tensors


class PINNDataset(Dataset):
    """
    Dataset для точек коллокации.

    Returns a :class:`PINNBatch` point sample.  Its iterator exposes the
    historical tuple ``(idx, x, y, sdf, boundary_types, mask_u_fixed,
    mask_v_fixed, mask_p_fixed, u_bc, v_bc, p_bc)`` while ``case_id`` remains
    available as a named field.
    Все тензоры хранятся в float32.

    Attributes:
        x: X-координаты (нормированные)
        y: Y-координаты (нормированные)
        sdf: Расстояние до стенки (нормированное)
        p_initial_guess: Начальное поле давления, не supervised target
        u_initial_guess: Начальное поле скорости U, не supervised target
        v_initial_guess: Начальное поле скорости V, не supervised target
        boundary_types: One-hot кодирование типов границ
        mask_u_fixed: Маска фиксированных компонент U
        mask_v_fixed: Маска фиксированных компонент V
        mask_p_fixed: Маска фиксированных компонент P
        u_bc: Значения BC для U
        v_bc: Значения BC для V
        p_bc: Значения BC для P
        normal_x: X-компонента единичной нормали границы
        normal_y: Y-компонента единичной нормали границы
    """
    def __init__(
        self,
        points_file: str,
        pde_system: str = 'incompressible_navier_stokes',
    ) -> None:
        """
        Инициализирует датасет из файла.

        Args:
            points_file: Путь к файлу с точками коллокации
            pde_system: Система PDE, определяющая компонентные hard BC.

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

        def _field_copy(name: str) -> np.ndarray:
            # A field view of a structured dtype can have a record-sized
            # stride that torch.as_tensor rejects even when NumPy labels it
            # C-contiguous (notably for a one-row file).
            return np.array(data[name], copy=True, order='C')

        self.x = torch.tensor(_field_copy('x_norm'), dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(_field_copy('y_norm'), dtype=torch.float32).view(-1, 1)
        self.sdf = torch.tensor(_field_copy('sdf_norm'), dtype=torch.float32).view(-1, 1)

        # These are initializer fields, not ground-truth solution labels.
        # Explicit names prevent accidental inclusion in a supervised loss.
        self.p_initial_guess = torch.tensor(
            _field_copy('p_norm'),
            dtype=torch.float32,
        ).view(-1, 1)
        self.u_initial_guess = torch.tensor(
            _field_copy('u_norm'),
            dtype=torch.float32,
        ).view(-1, 1)
        self.v_initial_guess = torch.tensor(
            _field_copy('v_norm'),
            dtype=torch.float32,
        ).view(-1, 1)

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
        self.normal_x = torch.full(
            (len(self.x), 1),
            float("nan"),
            dtype=torch.float32,
        )
        self.normal_y = torch.full(
            (len(self.x), 1),
            float("nan"),
            dtype=torch.float32,
        )
        self.case_id = torch.zeros(len(self.x), dtype=torch.long)

        if 'on_boundary' in data.dtype.names:
            self.on_boundary = torch.tensor(
                _field_copy('on_boundary'),
                dtype=torch.bool
            )
        if 'on_airfoil' in data.dtype.names:
            self.on_airfoil = torch.tensor(
                _field_copy('on_airfoil'),
                dtype=torch.bool
            )
        if 'segment_id' in data.dtype.names:
            self.segment_id = torch.tensor(
                _field_copy('segment_id'),
                dtype=torch.long
            )
        if 'normal_x' in data.dtype.names:
            self.normal_x = torch.tensor(
                _field_copy('normal_x'),
                dtype=torch.float32,
            ).view(-1, 1)
        if 'normal_y' in data.dtype.names:
            self.normal_y = torch.tensor(
                _field_copy('normal_y'),
                dtype=torch.float32,
            ).view(-1, 1)
        if 'case_id' in data.dtype.names:
            self.case_id = torch.tensor(
                _field_copy('case_id'),
                dtype=torch.long,
            )
            if bool((self.case_id < 0).any()):
                raise ValueError("case_id values must be non-negative")

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

                elif (
                    bt_str == 'wall'
                    and pde_system == 'incompressible_navier_stokes'
                ):
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # u = 0, v = 0 (уже по умолчанию)

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

    @property
    def p_target(self) -> torch.Tensor:
        """Deprecated alias for the non-supervised pressure initial guess."""
        return self.p_initial_guess

    @p_target.setter
    def p_target(self, value: torch.Tensor) -> None:
        self.p_initial_guess = value

    @property
    def u_target(self) -> torch.Tensor:
        """Deprecated alias for the non-supervised x-velocity initial guess."""
        return self.u_initial_guess

    @u_target.setter
    def u_target(self, value: torch.Tensor) -> None:
        self.u_initial_guess = value

    @property
    def v_target(self) -> torch.Tensor:
        """Deprecated alias for the non-supervised y-velocity initial guess."""
        return self.v_initial_guess

    @v_target.setter
    def v_target(self, value: torch.Tensor) -> None:
        self.v_initial_guess = value

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

    def __getitem__(self, idx: int) -> PINNBatch:
        """
        Возвращает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Named point sample with legacy tuple iteration.
        """
        boundary_types = (
            self.boundary_types[idx]
            if self.boundary_types is not None
            else torch.zeros(0)
        )
        return PINNBatch(
            index=torch.tensor(idx, dtype=torch.long),
            x=self.x[idx],
            y=self.y[idx],
            sdf=self.sdf[idx],
            boundary_types=boundary_types,
            mask_u_fixed=self.mask_u_fixed[idx].view(1),
            mask_v_fixed=self.mask_v_fixed[idx].view(1),
            mask_p_fixed=self.mask_p_fixed[idx].view(1),
            u_bc=self.u_bc[idx].view(1),
            v_bc=self.v_bc[idx].view(1),
            p_bc=self.p_bc[idx].view(1),
            normal_x=self.normal_x[idx],
            normal_y=self.normal_y[idx],
            case_id=self.case_id[idx],
        )

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
                     'normal_x', 'normal_y',
                     'case_id', 'mesh_x_phys', 'mesh_y_phys']:
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
        seed: int = 42,
        pde_system: str = 'incompressible_navier_stokes',
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
            seed: базовый seed перемешивания и worker-процессов.
            pde_system: Система PDE для построения hard-BC масок.
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
        self.seed = int(seed)
        self.pde_system = str(pde_system)
        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(self.seed)
        self.mesh_data: Optional[MeshData] = None
        self.mesh_file = str(resolve_project_path(mesh_file)) if mesh_file is not None else None
        self.mesh_domain_coords: Optional[np.ndarray] = None
        self.mesh_airfoil_coords: Optional[np.ndarray] = None
        self.sampler_metadata: Optional[Dict[str, Any]] = None
        self.geometry_reference: Optional[Dict[str, Any]] = None
        self.fvm_sdf_geometry_fingerprint: Optional[str] = None
        self.cached_sdf_geometry_fingerprint: Optional[str] = None
        self.cached_sdf_method: Optional[str] = None
        self.cached_sdf_max_abs_error: Optional[float] = None

        self.dataset = PINNDataset(
            self.points_file,
            pde_system=self.pde_system,
        )
        self.load_metadata()
        self._load_sampler_point_metadata()
        self._reconcile_geometry_normalization()
        self._log_flow_cases()
        # После загрузки метаданных устанавливаем значения на inlet
        self.dataset.set_inlet_values(self.inlet_u_norm, self.inlet_v_norm)
        self.dataloader = self._make_dataloader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        logger.info(
            "DataLoader: batch=%d, workers=%d, pin_memory=%s, "
            "persistent=%s, prefetch=%s, seed=%d",
            self.batch_size,
            self.num_workers,
            self.pin_memory,
            self.persistent_workers if self.num_workers > 0 else False,
            self.prefetch_factor if self.num_workers > 0 else None,
            self.seed,
        )
        if os.name == "nt" and self.num_workers > 0:
            logger.info(
                "Windows multiprocessing uses spawn; launch training below "
                "an `if __name__ == '__main__'` guard."
            )

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
            'collate_fn': collate_pinn_batches,
            'generator': self._loader_generator,
            'worker_init_fn': _seed_dataloader_worker,
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
        self.field_role = str(
            meta.get("field_role", "legacy_initial_guess")
        ).lower()
        if self.field_role not in {
            "initial_guess_only",
            "legacy_initial_guess",
        }:
            raise ValueError(
                "Collocation artifact field_role must identify initializer "
                f"fields, received {self.field_role!r}. Supervised reference "
                "solutions require a separate validated data contract."
            )
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
        cached_fingerprint = meta.get('sdf_geometry_fingerprint')
        self.cached_sdf_geometry_fingerprint = (
            str(cached_fingerprint)
            if cached_fingerprint
            else None
        )
        cached_method = meta.get('sdf_method')
        self.cached_sdf_method = (
            str(cached_method)
            if cached_method
            else None
        )
        self.delta_bl_99_est = meta.get('delta_bl_99_est')
        self.delta_bl_99_est_norm = meta.get('delta_bl_99_est_norm')
        self.boundary_layer_model = str(meta.get('delta_bl_model', 'laminar'))
        self.geometry_reference = meta.get('geometry_reference')
        self.inlet_u_norm = meta.get('inlet_u_norm', 1.0)
        self.inlet_v_norm = meta.get('inlet_v_norm', 0.0)
        self.has_explicit_flow_cases = meta.get('flow_cases') is not None
        self.flow_cases = flow_cases_from_metadata(
            meta,
            fallback_reynolds=self.Re,
            fallback_u_normalized=self.inlet_u_norm,
            fallback_v_normalized=self.inlet_v_norm,
        )
        unknown_case_ids = (
            set(int(value) for value in torch.unique(self.dataset.case_id))
            - set(self.flow_cases)
        )
        if unknown_case_ids:
            raise ValueError(
                "Dataset references undefined flow case IDs: "
                f"{sorted(unknown_case_ids)}"
            )
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

    def _log_flow_cases(self) -> None:
        """Log the final case table after legacy geometry reconciliation."""
        logger.info(
            "Flow cases: %s",
            ", ".join(
                f"id={case.case_id}:Re={case.reynolds:.6g},"
                f"alpha={case.angle_degrees:.6g}deg"
                for case in self.flow_cases.values()
            ),
        )

    def get_flow_case_batch(
        self,
        case_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FlowCaseBatch:
        """Materialize dimensionless case parameters for point associations."""
        ids = case_ids.to(device=device, dtype=torch.long).reshape(-1, 1)
        batch_size = ids.shape[0]
        reynolds = torch.empty((batch_size, 1), device=device, dtype=dtype)
        cos_alpha = torch.empty_like(reynolds)
        sin_alpha = torch.empty_like(reynolds)

        pressure_defined = all(
            case.pressure_drop_coefficient is not None
            for case in self.flow_cases.values()
        )
        pressure_drop = (
            torch.empty_like(reynolds)
            if pressure_defined
            else None
        )
        geometry_widths = {
            len(case.geometry_parameters) for case in self.flow_cases.values()
        }
        if len(geometry_widths) != 1:
            raise ValueError(
                "All flow cases must use the same geometry parameter width"
            )
        geometry_width = geometry_widths.pop()
        geometry = (
            torch.empty(
                (batch_size, geometry_width),
                device=device,
                dtype=dtype,
            )
            if geometry_width > 0
            else None
        )

        assigned = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        for case_id, case in self.flow_cases.items():
            mask = ids == int(case_id)
            if not bool(mask.any()):
                continue
            reynolds[mask] = case.reynolds
            cos_alpha[mask] = case.cos_alpha
            sin_alpha[mask] = case.sin_alpha
            assigned |= mask
            if pressure_drop is not None:
                pressure_drop[mask] = float(
                    case.pressure_drop_coefficient
                )
            if geometry is not None:
                row_mask = mask.reshape(-1)
                geometry[row_mask] = torch.as_tensor(
                    case.geometry_parameters,
                    device=device,
                    dtype=dtype,
                )

        if not bool(assigned.all()):
            missing = torch.unique(ids[~assigned]).detach().cpu().tolist()
            raise ValueError(f"Undefined flow case IDs in batch: {missing}")
        return FlowCaseBatch(
            case_id=ids,
            reynolds=reynolds,
            cos_alpha=cos_alpha,
            sin_alpha=sin_alpha,
            pressure_drop_coefficient=pressure_drop,
            geometry_parameters=geometry,
        )

    def _load_sampler_point_metadata(self) -> None:
        """
        Дополняет initial.npy геометрическими флагами и нормалями из
        collocation_points.npy.

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
            if {'normal_x', 'normal_y'}.issubset(names):
                self.dataset.normal_x = torch.tensor(
                    np.array(arr['normal_x'], copy=True),
                    dtype=torch.float32,
                ).view(-1, 1)
                self.dataset.normal_y = torch.tensor(
                    np.array(arr['normal_y'], copy=True),
                    dtype=torch.float32,
                ).view(-1, 1)
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
        if not self.has_explicit_flow_cases:
            legacy_case = self.flow_cases[0]
            self.flow_cases[0] = FlowCase(
                case_id=0,
                reynolds=self.Re,
                angle_degrees=legacy_case.angle_degrees,
                pressure_drop_coefficient=(
                    legacy_case.pressure_drop_coefficient
                ),
                geometry_parameters=legacy_case.geometry_parameters,
                name=legacy_case.name,
            )
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

    @staticmethod
    def _mesh_cells(mesh_data: MeshData) -> List[np.ndarray]:
        """Return canonical polygon cells stored in a mesh artifact."""
        if (
            mesh_data.cell_nodes is not None
            and mesh_data.cell_offsets is not None
            and len(mesh_data.cell_offsets) >= 2
        ):
            nodes = np.asarray(mesh_data.cell_nodes, dtype=np.int64)
            offsets = np.asarray(mesh_data.cell_offsets, dtype=np.int64)
            return [
                nodes[int(start):int(stop)]
                for start, stop in zip(offsets[:-1], offsets[1:])
            ]
        return [
            np.asarray(triangle, dtype=np.int64)
            for triangle in np.asarray(mesh_data.triangles, dtype=np.int64)
        ]

    def _ensure_fvm_quadrature(self) -> None:
        """Populate direct-PINN face quadrature, including legacy meshes."""
        mesh = self.mesh_data
        if mesh is None:
            return
        required = (
            mesh.int_quad_coords,
            mesh.int_quad_area,
            mesh.int_quad_node_i,
            mesh.int_quad_node_j,
            mesh.bnd_quad_coords,
        )
        if all(value is not None for value in required):
            return

        from airfoil_pinn.sampling.mesh.core import (
            build_dual_face_quadrature,
        )

        quadrature = build_dual_face_quadrature(
            mesh.coords,
            self._mesh_cells(mesh),
            mesh.int_face_node_i,
            mesh.int_face_node_j,
            mesh.int_face_area,
            mesh.bnd_face_node,
            mesh.bnd_face_neighbor,
        )
        for name, values in quadrature.items():
            setattr(mesh, name, values)
        mesh.mesh_schema_version = 2
        logger.info(
            "FVM face quadrature reconstructed: internal segments=%d, "
            "boundary segments=%d",
            mesh.n_int_quad_points,
            len(mesh.bnd_quad_coords),
        )

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
            mesh_coords = np.asarray(data['coords'], dtype=np.float64)
            self._validate_mesh_coordinate_bounds(mesh_coords, mesh_file)
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
                cell_nodes=(
                    np.array(data['cell_nodes']) if 'cell_nodes' in data.files else None
                ),
                cell_offsets=(
                    np.array(data['cell_offsets']) if 'cell_offsets' in data.files else None
                ),
                quads=np.array(data['quads']) if 'quads' in data.files else None,
                int_quad_coords=(
                    np.array(data['int_quad_coords'])
                    if 'int_quad_coords' in data.files else None
                ),
                int_quad_area=(
                    np.array(data['int_quad_area'])
                    if 'int_quad_area' in data.files else None
                ),
                int_quad_node_i=(
                    np.array(data['int_quad_node_i'])
                    if 'int_quad_node_i' in data.files else None
                ),
                int_quad_node_j=(
                    np.array(data['int_quad_node_j'])
                    if 'int_quad_node_j' in data.files else None
                ),
                int_quad_face=(
                    np.array(data['int_quad_face'])
                    if 'int_quad_face' in data.files else None
                ),
                bnd_quad_coords=(
                    np.array(data['bnd_quad_coords'])
                    if 'bnd_quad_coords' in data.files else None
                ),
                mesh_schema_version=(
                    int(np.asarray(data['mesh_schema_version']).item())
                    if 'mesh_schema_version' in data.files else 1
                ),
            )
        self._ensure_fvm_quadrature()
        logger.info(
            "FVM-сетка загружена: %d узлов, %d треуг., %d вн. граней, %d гран. граней",
            self.mesh_data.n_nodes, self.mesh_data.n_triangles,
            self.mesh_data.n_int_faces, self.mesh_data.n_bnd_faces,
        )
        self._prepare_fvm_tensors()

    def _validate_mesh_coordinate_bounds(self, coords, mesh_file: str) -> None:
        """Reject a stale mesh extending beyond the selected training domain."""
        normalized = np.asarray(coords, dtype=np.float64) / float(self.chord)
        lower = np.array([self.dataset.x.min().item(), self.dataset.y.min().item()])
        upper = np.array([self.dataset.x.max().item(), self.dataset.y.max().item()])
        mesh_lower, mesh_upper = normalized.min(axis=0), normalized.max(axis=0)
        tolerance = 1e-6 * max(1.0, float(np.max(np.abs([lower, upper]))))
        if np.any(mesh_lower < lower - tolerance) or np.any(mesh_upper > upper + tolerance):
            raise ValueError(
                f"FVM-сетка {mesh_file} выходит за область датасета {self.points_file}. "
                f"Датасет: x/c=[{lower[0]:.6g}, {upper[0]:.6g}], "
                f"y/c=[{lower[1]:.6g}, {upper[1]:.6g}]; "
                f"сетка: x/c=[{mesh_lower[0]:.6g}, {mesh_upper[0]:.6g}], "
                f"y/c=[{mesh_lower[1]:.6g}, {mesh_upper[1]:.6g}]. "
                "Выберите согласованные data.points_file и data.mesh_file "
                "или пересоздайте сетку и начальные поля для одной области."
            )

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
            cell_nodes=(
                np.concatenate(mesh.cells).astype(np.int32)
                if mesh.cells else np.empty(0, dtype=np.int32)
            ),
            cell_offsets=np.concatenate((
                np.zeros(1, dtype=np.int64),
                np.cumsum([len(cell) for cell in mesh.cells], dtype=np.int64),
            )),
            quads=np.asarray(mesh.quads, dtype=np.int32),
        )
        self._ensure_fvm_quadrature()
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
            raise RuntimeError(
                "В загруженном mesh2d.npz отсутствует domain_coords. "
                "Пересоздайте сетку через S2DR: актуальный mesh-артефакт должен "
                "хранить контур расчётной области внутри самого файла."
            )

        from airfoil_pinn.sampling.mesh import Mesh2D

        quad_cells_xy = None
        if self.mesh_data.quads is not None and len(self.mesh_data.quads):
            quad_cells_xy = self.mesh_data.coords[self.mesh_data.quads]
        mesh = Mesh2D(
            self.mesh_points_from_dataset(),
            domain_coords=self.mesh_domain_coords,
            airfoil_coords=self.mesh_airfoil_coords,
            quad_cells_xy=quad_cells_xy,
        )
        mesh.build(verbose=verbose)
        return mesh

    def _prepare_fvm_tensors(self) -> None:
        """
        Подготавливает нормированные тензоры (x, y, sdf, boundary_types) для FVM-шага.

        Подготавливаются как узловые координаты для интегральной диагностики,
        так и midpoint-координаты прямых сегментов дуальных FVM-граней для
        production FVM loss. SDF вычисляется каноническим
        PolylineDistanceField по тем же нормированным сегментам, что
        используются training и inference.

        ``fvm_quad_*`` используются для прямого PINN-предсказания гранных
        скоростей без интерполяции узловых значений.
        """
        self._ensure_fvm_quadrature()
        coords = self.mesh_data.coords          # (N, 2) физические
        N = len(coords)

        x_norm_float64 = np.asarray(
            coords[:, 0] / self.chord,
            dtype=np.float64,
        )
        y_norm_float64 = np.asarray(
            coords[:, 1] / self.chord,
            dtype=np.float64,
        )
        query_norm = np.column_stack(
            [x_norm_float64, y_norm_float64]
        )
        if (
            self.mesh_airfoil_coords is None
            or len(self.mesh_airfoil_coords) < 2
        ):
            raise RuntimeError(
                "FVM SDF requires an ordered airfoil polyline; "
                "nearest-neighbour/vertex SDF fallback is forbidden."
            )

        airfoil_norm = (
            np.asarray(self.mesh_airfoil_coords, dtype=np.float64)
            / self.chord
        )
        closed = (
            bool(self.sampler_metadata.get('foil_closed', True))
            if isinstance(self.sampler_metadata, dict)
            else len(airfoil_norm) >= 3
        )
        distance_field = PolylineDistanceField(
            airfoil_norm,
            closed=closed,
            segment_chunk_size=256,
        )
        sdf_norm_float64 = distance_field.evaluate_numpy(
            query_norm,
            point_chunk_size=2048,
        )
        self.fvm_sdf_float64 = torch.from_numpy(
            np.ascontiguousarray(sdf_norm_float64)
        ).view(-1, 1)
        self.fvm_sdf_geometry_fingerprint = (
            distance_field.geometry_fingerprint
        )
        logger.info(
            "FVM SDF calculated from exact normalized segments: "
            "segments=%d, geometry=%s",
            distance_field.segment_count,
            self.fvm_sdf_geometry_fingerprint[:12],
        )

        # One-hot типов границ (по именам из датасета)
        bt_names = self.dataset.boundary_type_names
        type_to_idx = self.dataset.type_to_idx
        bt_one_hot = np.zeros((N, len(bt_names)), dtype=np.float32)
        for i, t in enumerate(self.mesh_data.node_bnd_type):
            t_lower = str(t).lower()
            if t_lower in type_to_idx:
                bt_one_hot[i, type_to_idx[t_lower]] = 1.0

        # Keep the canonical query coordinates and their SDF in float64.
        # Consumers cast these tensors to the model dtype at the device
        # boundary, so preserving them here avoids an irreversible coordinate
        # round-trip before inference or FVM evaluation.
        self.fvm_x = torch.from_numpy(
            np.ascontiguousarray(x_norm_float64)
        ).view(-1, 1)
        self.fvm_y = torch.from_numpy(
            np.ascontiguousarray(y_norm_float64)
        ).view(-1, 1)
        self.fvm_sdf = self.fvm_sdf_float64
        self.fvm_bt  = torch.tensor(bt_one_hot,  dtype=torch.float32) if bt_names else None

        # Direct-PINN FVM quadrature.  Internal points are midpoints of the
        # straight median-dual face segments; boundary points are midpoints of
        # physical boundary half-edges.  They are evaluated by the network
        # directly and are never reconstructed from nodal velocities.
        int_quad = np.asarray(
            self.mesh_data.int_quad_coords,
            dtype=np.float64,
        ).reshape(-1, 2)
        bnd_quad = np.asarray(
            self.mesh_data.bnd_quad_coords,
            dtype=np.float64,
        ).reshape(-1, 2)
        quad_phys = np.vstack((int_quad, bnd_quad))
        quad_norm = quad_phys / float(self.chord)
        quad_sdf = distance_field.evaluate_numpy(
            quad_norm,
            point_chunk_size=2048,
        )
        self.fvm_int_quad_count = len(int_quad)
        self.fvm_bnd_quad_count = len(bnd_quad)
        self.fvm_quad_x = torch.from_numpy(
            np.ascontiguousarray(quad_norm[:, 0])
        ).view(-1, 1)
        self.fvm_quad_y = torch.from_numpy(
            np.ascontiguousarray(quad_norm[:, 1])
        ).view(-1, 1)
        self.fvm_quad_sdf = torch.from_numpy(
            np.ascontiguousarray(quad_sdf)
        ).view(-1, 1)

        quad_bt = np.zeros((len(quad_norm), len(bt_names)), dtype=np.float32)
        if bt_names:
            offset = self.fvm_int_quad_count
            for local_index, boundary_type in enumerate(
                self.mesh_data.bnd_face_type
            ):
                type_index = type_to_idx.get(str(boundary_type).lower())
                if type_index is not None:
                    quad_bt[offset + local_index, type_index] = 1.0
        self.fvm_quad_bt = (
            torch.from_numpy(quad_bt) if bt_names else None
        )

        if N != self.n_points:
            logger.warning(
                "FVM-сетка (%d узлов) ≠ датасет (%d точек) — "
                "Sampler пропустил %d узлов. FVM использует координаты сетки напрямую.",
                N, self.n_points, abs(N - self.n_points),
            )
        logger.info(
            "FVM tensors prepared: nodes=%d, internal quadrature=%d, "
            "boundary quadrature=%d",
            N,
            self.fvm_int_quad_count,
            self.fvm_bnd_quad_count,
        )

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
        case_id: Any = 0,
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
        case_ids = np.asarray(case_id, dtype=np.int64)
        if case_ids.ndim == 0:
            case_ids = np.full(n_new, int(case_ids), dtype=np.int64)
        else:
            case_ids = case_ids.reshape(-1)
        if len(case_ids) != n_new:
            raise ValueError(
                "case_id must be scalar or contain one value per RAR point"
            )
        undefined = sorted(set(case_ids.tolist()) - set(self.flow_cases))
        if undefined:
            raise ValueError(
                f"Undefined flow case IDs for RAR: {undefined}"
            )

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
        new_xy = np.column_stack([x_norm, y_norm])
        nn_idx = np.empty(n_new, dtype=np.int64)
        old_case_ids = ds.case_id.cpu().numpy().reshape(-1)
        for current_case in np.unique(case_ids):
            new_mask = case_ids == current_case
            old_indices = np.flatnonzero(old_case_ids == current_case)
            if len(old_indices) == 0:
                raise RuntimeError(
                    "RAR cannot initialize reference fields because flow case "
                    f"{int(current_case)} has no existing dataset points"
                )
            _, local_idx = cKDTree(old_xy[old_indices]).query(
                new_xy[new_mask],
                workers=-1,
            )
            nn_idx[new_mask] = old_indices[np.asarray(local_idx).reshape(-1)]
        nn_idx_t = torch.tensor(nn_idx, dtype=torch.long)

        ds.x = torch.cat([ds.x, torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.y = torch.cat([ds.y, torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)], dim=0)
        ds.sdf = torch.cat(
            [
                ds.sdf,
                torch.tensor(
                    sdf_norm,
                    dtype=torch.float64,
                ).view(-1, 1),
            ],
            dim=0,
        )
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
        missing_normals = torch.full(
            (n_new, 1),
            float("nan"),
            dtype=torch.float32,
        )
        ds.normal_x = torch.cat([ds.normal_x, missing_normals.clone()], dim=0)
        ds.normal_y = torch.cat([ds.normal_y, missing_normals], dim=0)
        ds.case_id = torch.cat(
            [
                ds.case_id,
                torch.as_tensor(case_ids, dtype=torch.long),
            ],
            dim=0,
        )
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
            'segment_id', 'normal_x', 'normal_y', 'case_id',
            'source_indices', 'mesh_x_phys',
            'mesh_y_phys',
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
                'cell_nodes': self.mesh_data.cell_nodes,
                'cell_offsets': self.mesh_data.cell_offsets,
                'quads': self.mesh_data.quads,
                'int_quad_coords': self.mesh_data.int_quad_coords,
                'int_quad_area': self.mesh_data.int_quad_area,
                'int_quad_node_i': self.mesh_data.int_quad_node_i,
                'int_quad_node_j': self.mesh_data.int_quad_node_j,
                'int_quad_face': self.mesh_data.int_quad_face,
                'bnd_quad_coords': self.mesh_data.bnd_quad_coords,
                'mesh_schema_version': self.mesh_data.mesh_schema_version,
            }

        return {
            'dataset': dataset_state,
            'mesh_data': mesh_state,
            'mesh_domain_coords': self.mesh_domain_coords,
            'mesh_airfoil_coords': self.mesh_airfoil_coords,
            'cached_sdf_geometry_fingerprint': (
                self.cached_sdf_geometry_fingerprint
            ),
            'cached_sdf_method': self.cached_sdf_method,
            'loader_generator_state': (
                self._loader_generator.get_state().cpu()
            ),
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
        if (
            getattr(ds, 'case_id', None) is None
            or len(ds.case_id) != ds.n_points
        ):
            ds.case_id = torch.zeros(ds.n_points, dtype=torch.long)
        if getattr(ds, 'mesh_x_phys', None) is None:
            ds.mesh_x_phys = (ds.x.double().flatten() * float(self.chord)).clone()
        if getattr(ds, 'mesh_y_phys', None) is None:
            ds.mesh_y_phys = (ds.y.double().flatten() * float(self.chord)).clone()

        self.mesh_domain_coords = state.get('mesh_domain_coords', self.mesh_domain_coords)
        self.mesh_airfoil_coords = state.get('mesh_airfoil_coords', self.mesh_airfoil_coords)
        self.cached_sdf_geometry_fingerprint = state.get(
            'cached_sdf_geometry_fingerprint',
            self.cached_sdf_geometry_fingerprint,
        )
        self.cached_sdf_method = state.get(
            'cached_sdf_method',
            self.cached_sdf_method,
        )
        loader_generator_state = state.get('loader_generator_state')
        if loader_generator_state is not None:
            self._loader_generator.set_state(
                loader_generator_state.cpu()
            )

        mesh_state = state.get('mesh_data')
        if mesh_state is not None:
            self.mesh_data = MeshData(**mesh_state)
            self._prepare_fvm_tensors()
        else:
            self.mesh_data = None
            for attr in (
                'fvm_x',
                'fvm_y',
                'fvm_sdf',
                'fvm_sdf_float64',
                'fvm_bt',
                'fvm_quad_x',
                'fvm_quad_y',
                'fvm_quad_sdf',
                'fvm_quad_bt',
                'fvm_int_quad_count',
                'fvm_bnd_quad_count',
                'fvm_sdf_geometry_fingerprint',
            ):
                if hasattr(self, attr):
                    delattr(self, attr)

        self.dataloader = self._make_dataloader(self.dataset, shuffle=self.shuffle)
        logger.info("DataHandler восстановлен из чекпоинта: %d точек", self.n_points)

    # ------------------------------------------------------------------
    # Методы для диагностики профилей и границ
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
        y_max_ratio: float = 0.95,
        case_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интерполирует профиль ``u(y)`` строго на линии ``x=x_target``.

        Если загружена FVM-сетка, значения сначала однозначно сопоставляются
        её узлам, после чего используется кусочно-линейная интерполяция по
        исходным треугольникам. Поэтому профиль не смешивает точки из широкой
        полосы по X, а область внутри профиля остаётся замаскированной.

        Args:
            x_target: Целевая X-координата (нормированная)
            u_vals: Значения скорости U (если None, используются начальные значения)
            y_grid: Сетка по Y для интерполяции
            num_points: Количество точек для равномерной сетки
            y_max_ratio: Доля полной высоты области, симметрично относительно
                её середины
            case_id: Индекс режима для параметрического датасета

        Returns:
            Кортеж (y_coords, u_interp) – массивы numpy
        """
        source = self.dataset.u_target if u_vals is None else u_vals
        if self.mesh_data is not None:
            node_values = self._dataset_values_on_mesh_nodes(
                source,
                case_id=case_id,
            )
            return self.interpolate_fvm_profile(
                x_target,
                node_values,
                y_grid=y_grid,
                num_points=num_points,
                y_max_ratio=y_max_ratio,
            )

        logger.warning(
            "Точная профильная интерполяция x=%.6g недоступна без "
            "FVM-треугольников; используется узкий scattered fallback.",
            x_target,
        )
        return self._interpolate_profile_without_mesh(
            x_target,
            source,
            y_grid=y_grid,
            num_points=num_points,
            y_max_ratio=y_max_ratio,
            case_id=case_id,
        )

    def interpolate_fvm_profile(
        self,
        x_target: float,
        node_values: Any,
        y_grid: Optional[Any] = None,
        num_points: int = 100,
        y_max_ratio: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a nodal FVM field on the exact vertical line ``x=x_target``."""
        if self.mesh_data is None:
            raise RuntimeError(
                "interpolate_fvm_profile requires a loaded FVM mesh"
            )
        if not 0.0 < float(y_max_ratio) <= 1.0:
            raise ValueError("y_max_ratio must be in the interval (0, 1]")
        if int(num_points) < 2:
            raise ValueError("num_points must be at least 2")

        values = (
            node_values.detach().cpu().numpy()
            if torch.is_tensor(node_values)
            else np.asarray(node_values)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self.mesh_data.n_nodes:
            raise ValueError(
                "node_values length does not match the FVM mesh: "
                f"{values.size} != {self.mesh_data.n_nodes}"
            )

        triangulation = self._get_profile_triangulation()
        coords_norm = (
            np.asarray(self.mesh_data.coords, dtype=np.float64)
            / float(self.chord)
        )
        y_min = float(np.min(coords_norm[:, 1]))
        y_max = float(np.max(coords_norm[:, 1]))
        if y_grid is None:
            centre = 0.5 * (y_min + y_max)
            half_height = (
                0.5 * float(y_max_ratio) * (y_max - y_min)
            )
            y_query = np.linspace(
                centre - half_height,
                centre + half_height,
                int(num_points),
                dtype=np.float64,
            )
        else:
            y_query = (
                y_grid.detach().cpu().numpy()
                if torch.is_tensor(y_grid)
                else np.asarray(y_grid)
            )
            y_query = np.asarray(y_query, dtype=np.float64).reshape(-1)

        x_query = np.full_like(y_query, float(x_target))
        nodes, weights, valid = self._get_profile_stencil(
            triangulation,
            x_query,
            y_query,
        )
        sampled = np.full(y_query.shape, np.nan, dtype=np.float64)
        if bool(np.any(valid)):
            sampled[valid] = np.sum(
                values[nodes[valid]] * weights[valid],
                axis=1,
            )
        return y_query, sampled

    def integrate_fvm_vertical_flux(
        self,
        x_target: float,
        node_values: Any,
    ) -> float:
        """Exactly integrate a piecewise-linear nodal field on ``x=x_target``.

        The vertical line is split at every intersected triangle edge.  Each
        fluid interval therefore lies in one triangle and the trapezoidal
        integral is exact for the mesh's linear interpolant.  Intervals inside
        an airfoil hole or outside the domain are omitted.
        """
        if self.mesh_data is None:
            raise RuntimeError(
                "integrate_fvm_vertical_flux requires a loaded FVM mesh"
            )
        values = (
            node_values.detach().cpu().numpy()
            if torch.is_tensor(node_values)
            else np.asarray(node_values)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self.mesh_data.n_nodes:
            raise ValueError(
                "node_values length does not match the FVM mesh: "
                f"{values.size} != {self.mesh_data.n_nodes}"
            )

        triangulation = self._get_profile_triangulation()
        (
            nodes,
            lower_weights,
            upper_weights,
            interval_heights,
        ) = self._get_vertical_integration_stencil(
            triangulation,
            float(x_target),
        )
        if interval_heights.size == 0:
            return float("nan")
        lower_values = np.sum(
            values[nodes] * lower_weights,
            axis=1,
        )
        upper_values = np.sum(
            values[nodes] * upper_weights,
            axis=1,
        )
        return float(
            np.sum(
                0.5
                * (lower_values + upper_values)
                * interval_heights
            )
        )

    def _get_profile_triangulation(self):
        """Return a cached Matplotlib triangulation in chord coordinates."""
        if self.mesh_data is None:
            raise RuntimeError("FVM mesh is not loaded")
        cache_key = (
            id(self.mesh_data.coords),
            id(self.mesh_data.triangles),
            float(self.chord),
        )
        cached = getattr(self, "_profile_triangulation_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        import matplotlib.tri as mtri

        coords = (
            np.asarray(self.mesh_data.coords, dtype=np.float64)
            / float(self.chord)
        )
        triangulation = mtri.Triangulation(
            coords[:, 0],
            coords[:, 1],
            np.asarray(self.mesh_data.triangles, dtype=np.int32),
        )
        # A rebuilt adaptive mesh invalidates every geometry-only profile
        # stencil.  Resetting here also prevents stale cache growth across RAR
        # remeshing events.
        self._profile_stencil_cache = {}
        self._vertical_integration_stencil_cache = {}
        self._profile_mesh_index_cache = None
        self._profile_triangulation_cache = (
            cache_key,
            triangulation,
        )
        return triangulation

    def _get_profile_stencil(
        self,
        triangulation,
        x_query: np.ndarray,
        y_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build/cache triangle nodes and barycentric weights for a query line."""
        x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
        y_query = np.asarray(y_query, dtype=np.float64).reshape(-1)
        if x_query.shape != y_query.shape:
            raise ValueError("x_query and y_query must have equal shapes")

        stencil_key = (
            id(triangulation),
            x_query.tobytes(),
            y_query.tobytes(),
        )
        cache = getattr(self, "_profile_stencil_cache", None)
        if cache is None:
            cache = {}
            self._profile_stencil_cache = cache
        cached = cache.get(stencil_key)
        if cached is not None:
            return cached

        triangle_ids = np.asarray(
            triangulation.get_trifinder()(x_query, y_query),
            dtype=np.int64,
        )
        valid = triangle_ids >= 0
        nodes = np.zeros((x_query.size, 3), dtype=np.int64)
        weights = np.zeros((x_query.size, 3), dtype=np.float64)
        if bool(np.any(valid)):
            valid_nodes = np.asarray(
                triangulation.triangles[triangle_ids[valid]],
                dtype=np.int64,
            )
            nodes[valid] = valid_nodes
            x_vertices = np.asarray(
                triangulation.x[valid_nodes],
                dtype=np.float64,
            )
            y_vertices = np.asarray(
                triangulation.y[valid_nodes],
                dtype=np.float64,
            )
            x_points = x_query[valid]
            y_points = y_query[valid]
            denominator = (
                (y_vertices[:, 1] - y_vertices[:, 2])
                * (x_vertices[:, 0] - x_vertices[:, 2])
                + (x_vertices[:, 2] - x_vertices[:, 1])
                * (y_vertices[:, 0] - y_vertices[:, 2])
            )
            nondegenerate = np.abs(denominator) > np.finfo(
                np.float64
            ).eps
            if not bool(np.all(nondegenerate)):
                valid_positions = np.flatnonzero(valid)
                valid[valid_positions[~nondegenerate]] = False
                valid_nodes = valid_nodes[nondegenerate]
                x_vertices = x_vertices[nondegenerate]
                y_vertices = y_vertices[nondegenerate]
                x_points = x_points[nondegenerate]
                y_points = y_points[nondegenerate]
                denominator = denominator[nondegenerate]

            w0 = (
                (y_vertices[:, 1] - y_vertices[:, 2])
                * (x_points - x_vertices[:, 2])
                + (x_vertices[:, 2] - x_vertices[:, 1])
                * (y_points - y_vertices[:, 2])
            ) / denominator
            w1 = (
                (y_vertices[:, 2] - y_vertices[:, 0])
                * (x_points - x_vertices[:, 2])
                + (x_vertices[:, 0] - x_vertices[:, 2])
                * (y_points - y_vertices[:, 2])
            ) / denominator
            valid_weights = np.column_stack(
                (w0, w1, 1.0 - w0 - w1)
            )
            weights[valid] = valid_weights

        result = (nodes, weights, valid)
        cache[stencil_key] = result
        return result

    def _get_vertical_integration_stencil(
        self,
        triangulation,
        x_target: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cache exact linear-integration intervals for one vertical cut."""
        cache_key = (id(triangulation), float(x_target))
        cache = getattr(
            self,
            "_vertical_integration_stencil_cache",
            None,
        )
        if cache is None:
            cache = {}
            self._vertical_integration_stencil_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        triangle_nodes = np.asarray(
            triangulation.triangles,
            dtype=np.int64,
        )
        edge_start = triangle_nodes[:, [0, 1, 2]].reshape(-1)
        edge_end = triangle_nodes[:, [1, 2, 0]].reshape(-1)
        x0 = np.asarray(
            triangulation.x[edge_start],
            dtype=np.float64,
        )
        y0 = np.asarray(
            triangulation.y[edge_start],
            dtype=np.float64,
        )
        x1 = np.asarray(
            triangulation.x[edge_end],
            dtype=np.float64,
        )
        y1 = np.asarray(
            triangulation.y[edge_end],
            dtype=np.float64,
        )
        coordinate_scale = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        np.concatenate(
                            (
                                np.asarray(
                                    triangulation.x,
                                    dtype=np.float64,
                                ),
                                np.asarray(
                                    triangulation.y,
                                    dtype=np.float64,
                                ),
                            )
                        )
                    )
                )
            ),
        )
        tolerance = 1e-12 * coordinate_scale
        dx = x1 - x0
        nonvertical = np.abs(dx) > tolerance
        parameter = np.zeros_like(dx)
        parameter[nonvertical] = (
            (float(x_target) - x0[nonvertical])
            / dx[nonvertical]
        )
        crosses = (
            nonvertical
            & (parameter >= -tolerance)
            & (parameter <= 1.0 + tolerance)
        )
        intersections = [
            y0[crosses]
            + parameter[crosses] * (y1[crosses] - y0[crosses])
        ]
        coincident = (
            ~nonvertical
            & (np.abs(x0 - float(x_target)) <= tolerance)
        )
        if bool(np.any(coincident)):
            intersections.extend(
                (y0[coincident], y1[coincident])
            )
        y_intersections = np.concatenate(intersections)
        y_intersections = y_intersections[
            np.isfinite(y_intersections)
        ]
        y_intersections.sort()
        if y_intersections.size < 2:
            empty_nodes = np.empty((0, 3), dtype=np.int64)
            empty_weights = np.empty((0, 3), dtype=np.float64)
            result = (
                empty_nodes,
                empty_weights,
                empty_weights.copy(),
                np.empty(0, dtype=np.float64),
            )
            cache[cache_key] = result
            return result

        keep = np.concatenate(
            (
                np.array([True]),
                np.diff(y_intersections) > tolerance,
            )
        )
        breakpoints = y_intersections[keep]
        lower = breakpoints[:-1]
        upper = breakpoints[1:]
        heights = upper - lower
        nonempty = heights > tolerance
        lower = lower[nonempty]
        upper = upper[nonempty]
        heights = heights[nonempty]
        midpoint = 0.5 * (lower + upper)
        triangle_ids = np.asarray(
            triangulation.get_trifinder()(
                np.full_like(midpoint, float(x_target)),
                midpoint,
            ),
            dtype=np.int64,
        )
        fluid = triangle_ids >= 0
        lower = lower[fluid]
        upper = upper[fluid]
        heights = heights[fluid]
        nodes = np.asarray(
            triangulation.triangles[triangle_ids[fluid]],
            dtype=np.int64,
        )
        x_values = np.full(lower.shape, float(x_target))
        lower_weights = self._barycentric_weights(
            triangulation,
            nodes,
            x_values,
            lower,
        )
        upper_weights = self._barycentric_weights(
            triangulation,
            nodes,
            x_values,
            upper,
        )
        result = (
            nodes,
            lower_weights,
            upper_weights,
            heights,
        )
        cache[cache_key] = result
        return result

    @staticmethod
    def _barycentric_weights(
        triangulation,
        nodes: np.ndarray,
        x_query: np.ndarray,
        y_query: np.ndarray,
    ) -> np.ndarray:
        """Evaluate barycentric weights in explicitly selected triangles."""
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
        x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
        y_query = np.asarray(y_query, dtype=np.float64).reshape(-1)
        x_vertices = np.asarray(
            triangulation.x[nodes],
            dtype=np.float64,
        )
        y_vertices = np.asarray(
            triangulation.y[nodes],
            dtype=np.float64,
        )
        denominator = (
            (y_vertices[:, 1] - y_vertices[:, 2])
            * (x_vertices[:, 0] - x_vertices[:, 2])
            + (x_vertices[:, 2] - x_vertices[:, 1])
            * (y_vertices[:, 0] - y_vertices[:, 2])
        )
        if bool(
            np.any(
                np.abs(denominator)
                <= np.finfo(np.float64).eps
            )
        ):
            raise RuntimeError(
                "Degenerate triangle in vertical integration stencil"
            )
        w0 = (
            (y_vertices[:, 1] - y_vertices[:, 2])
            * (x_query - x_vertices[:, 2])
            + (x_vertices[:, 2] - x_vertices[:, 1])
            * (y_query - y_vertices[:, 2])
        ) / denominator
        w1 = (
            (y_vertices[:, 2] - y_vertices[:, 0])
            * (x_query - x_vertices[:, 2])
            + (x_vertices[:, 0] - x_vertices[:, 2])
            * (y_query - y_vertices[:, 2])
        ) / denominator
        return np.column_stack((w0, w1, 1.0 - w0 - w1))

    def _dataset_values_on_mesh_nodes(
        self,
        values: Any,
        *,
        case_id: int,
    ) -> np.ndarray:
        """Map dataset-ordered values to FVM-node order by physical position."""
        if self.mesh_data is None:
            raise RuntimeError("FVM mesh is not loaded")
        values_np = (
            values.detach().cpu().numpy()
            if torch.is_tensor(values)
            else np.asarray(values)
        )
        values_np = np.asarray(values_np, dtype=np.float64).reshape(-1)
        if values_np.size != self.dataset.n_points:
            raise ValueError(
                "Profile values must follow dataset order: "
                f"{values_np.size} != {self.dataset.n_points}"
            )

        case_ids = getattr(self.dataset, "case_id", None)
        if case_ids is None:
            selected = np.ones(self.dataset.n_points, dtype=bool)
        else:
            selected = (
                case_ids.detach().cpu().numpy().reshape(-1)
                == int(case_id)
            )
        if not bool(np.any(selected)):
            raise ValueError(f"case_id={case_id} is absent from the dataset")

        mesh_x = getattr(self.dataset, "mesh_x_phys", None)
        mesh_y = getattr(self.dataset, "mesh_y_phys", None)
        if mesh_x is None or mesh_y is None:
            dataset_coords = np.column_stack(
                (
                    self.dataset.x.detach().cpu().numpy().reshape(-1),
                    self.dataset.y.detach().cpu().numpy().reshape(-1),
                )
            ) * float(self.chord)
        else:
            dataset_coords = np.column_stack(
                (
                    mesh_x.detach().cpu().numpy().reshape(-1),
                    mesh_y.detach().cpu().numpy().reshape(-1),
                )
            )
        dataset_coords = np.asarray(
            dataset_coords[selected],
            dtype=np.float64,
        )
        selected_values = values_np[selected]

        cache_key = (
            id(self.mesh_data.coords),
            id(self.dataset.x),
            int(self.dataset.n_points),
            int(case_id),
        )
        cached = getattr(self, "_profile_mesh_index_cache", None)
        if cached is not None and cached[0] == cache_key:
            node_indices = cached[1]
        else:
            tree = cKDTree(dataset_coords)
            distances, node_indices = tree.query(
                np.asarray(self.mesh_data.coords, dtype=np.float64),
                k=1,
            )
            coordinate_scale = max(
                1.0,
                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                self.mesh_data.coords,
                                dtype=np.float64,
                            )
                        )
                    )
                ),
            )
            tolerance = 1e-10 * coordinate_scale
            max_distance = float(np.max(distances))
            if max_distance > tolerance:
                raise RuntimeError(
                    "Cannot map profile values to every FVM node: "
                    f"max coordinate mismatch={max_distance:.3e}, "
                    f"tolerance={tolerance:.3e}"
                )
            node_indices = np.asarray(node_indices, dtype=np.int64)
            self._profile_mesh_index_cache = (
                cache_key,
                node_indices,
            )
        return selected_values[node_indices]

    def _interpolate_profile_without_mesh(
        self,
        x_target: float,
        values: Any,
        *,
        y_grid: Optional[Any],
        num_points: int,
        y_max_ratio: float,
        case_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compatibility fallback for legacy point clouds without triangles."""
        if not 0.0 < float(y_max_ratio) <= 1.0:
            raise ValueError("y_max_ratio must be in the interval (0, 1]")
        x = self.dataset.x.detach().cpu().numpy().reshape(-1)
        y = self.dataset.y.detach().cpu().numpy().reshape(-1)
        values_np = (
            values.detach().cpu().numpy()
            if torch.is_tensor(values)
            else np.asarray(values)
        )
        values_np = np.asarray(values_np, dtype=np.float64).reshape(-1)
        case_ids = getattr(self.dataset, "case_id", None)
        if case_ids is not None:
            case_mask = (
                case_ids.detach().cpu().numpy().reshape(-1)
                == int(case_id)
            )
        else:
            case_mask = np.ones_like(x, dtype=bool)

        distance = np.abs(x - float(x_target))
        available = distance[case_mask]
        if available.size == 0:
            raise ValueError(f"case_id={case_id} is absent from the dataset")
        nearest = float(np.min(available))
        x_span = max(float(np.max(x) - np.min(x)), 1.0)
        tolerance = max(nearest + 1e-12, 2.5e-3 * x_span)
        mask = case_mask & (distance <= tolerance)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        centre = 0.5 * (y_min + y_max)
        half_height = 0.5 * float(y_max_ratio) * (y_max - y_min)
        if y_grid is None:
            y_query = np.linspace(
                centre - half_height,
                centre + half_height,
                int(num_points),
                dtype=np.float64,
            )
        else:
            y_query = (
                y_grid.detach().cpu().numpy()
                if torch.is_tensor(y_grid)
                else np.asarray(y_grid)
            )
            y_query = np.asarray(y_query, dtype=np.float64).reshape(-1)
        if int(np.count_nonzero(mask)) < 2:
            return y_query, np.full_like(y_query, np.nan)

        order = np.argsort(y[mask])
        y_sorted = y[mask][order]
        value_sorted = values_np[mask][order]
        sampled = np.interp(
            y_query,
            y_sorted,
            value_sorted,
            left=np.nan,
            right=np.nan,
        )
        return y_query, sampled
