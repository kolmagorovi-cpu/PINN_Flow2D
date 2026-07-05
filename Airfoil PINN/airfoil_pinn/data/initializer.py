#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль инициализации расчётной области (Initializer) с поддержкой многопроцессорности.

Возвращает данные в нормализованном виде для PINN:
    - координаты делятся на хорду
    - скорости делятся на U_inf
    - давление переводится в избыточное и делится на 0.5*ρ*U_inf²
    - SDF нормализуется на хорду (sdf_norm = sdf / chord)
    - на стенке скорость обнуляется, на границах симметрии обнуляются соответствующие компоненты
    - добавлена возможность выбора начального поля скорости (constant, linear_x, perturbed)
    - добавлена возможность добавления шума к давлению

Визуализация полей выполняется гладкой интерполяционной заливкой, для SDF
отображается физическое расстояние (умножением на chord).

Example:
    >>> init = Initializer()
    >>> init.load_sampler_points()
    >>> init.select_fluid()
    >>> init.select_conditions()
    >>> init.initialize_fields()
    >>> init.save('initial')

Автор: Колмагоров И.Д.
Версия: 3.260226 (финальная исправленная)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, List, Tuple, Dict, Any
import glob
import multiprocessing as mp
from functools import partial
from pathlib import Path

try:
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from airfoil_pinn.core.type_hints import InitializerMetadataDict

# Импортируем наши модули и классы
from airfoil_pinn.physics.fluids import FluidLibrary, Fluid
from airfoil_pinn.physics.conditions import (
    BoundaryConditions,
    ConditionsManager,
    select_fluid as choose_fluid,
)
from airfoil_pinn.data.sdf_diagnostics import (
    compute_nearest_wall_distances,
    compute_sdf_summary,
    estimate_boundary_layer_delta,
    format_sdf_summary,
)

# Корень проекта остается базой для существующих папок артефактов:
# Geometry/Data, Sampler/Data, Initial/Data и справочников JSON.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# Константы путей (абсолютные)
GEOMETRY_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
SAMPLER_DIR = os.path.join(PROJECT_ROOT, "Sampler", "Data")
FLUIDS_DIR = os.path.join(PROJECT_ROOT, "Fluids")
CONDITIONS_DIR = os.path.join(PROJECT_ROOT, "Conditions")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Initial", "Data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "Initial", "Plots")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = [
    "CONDITIONS_DIR",
    "FLUIDS_DIR",
    "GEOMETRY_DIR",
    "Initializer",
    "InitializerMetadataDict",
    "OUTPUT_DIR",
    "PLOTS_DIR",
    "PROJECT_ROOT",
    "SAMPLER_DIR",
    "SCIPY_AVAILABLE",
    "_compute_sdf_chunk",
    "initialize_menu",
]

# Структура данных для точки с нормализованными полями
point_dtype = np.dtype([
    ('id', 'i4'),
    ('x_norm', 'f8'),
    ('y_norm', 'f8'),
    ('sdf_norm', 'f8'),
    ('boundary_type', 'U20'),
    ('on_boundary', 'bool'),
    ('p_norm', 'f8'),
    ('u_norm', 'f8'),
    ('v_norm', 'f8')
])


# ----------------------------------------------------------------------
# Вспомогательная функция для многопроцессорного вычисления SDF (вынесена на уровень модуля)
# ----------------------------------------------------------------------
def _compute_sdf_chunk(
    indices: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    wall_points: np.ndarray,
    use_scipy: bool
) -> np.ndarray:
    """
    Вычисляет SDF (в метрах) для набора индексов.

    При наличии SciPy строит KDTree внутри процесса и выполняет один
    векторизованный query на весь chunk. Без SciPy использует блочную
    NumPy-векторизацию, чтобы не делать вложенный Python-цикл по точкам.

    Args:
        indices: Индексы точек для вычисления
        xs: X-координаты всех точек
        ys: Y-координаты всех точек
        wall_points: Координаты точек на стенке
        use_scipy: Использовать ли KDTree

    Returns:
        Массив расстояний до стенки
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or len(wall_points) == 0:
        return np.zeros(len(indices), dtype=np.float64)

    query_points = np.column_stack((xs[indices], ys[indices])).astype(np.float64, copy=False)

    if use_scipy and SCIPY_AVAILABLE and len(wall_points) > 0:
        tree = cKDTree(wall_points)
        dist, _ = tree.query(query_points, k=1)
        return np.asarray(dist, dtype=np.float64)

    wall_points = np.asarray(wall_points, dtype=np.float64)
    sdf_chunk = np.empty(len(indices), dtype=np.float64)
    max_pairs = 1_000_000
    block_size = max(1, min(len(query_points), max_pairs // max(1, len(wall_points))))
    for start in range(0, len(query_points), block_size):
        stop = min(start + block_size, len(query_points))
        block = query_points[start:stop]
        delta = block[:, None, :] - wall_points[None, :, :]
        dist_sq = np.sum(delta * delta, axis=2)
        sdf_chunk[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return sdf_chunk


class Initializer:
    """
    Класс для инициализации расчётной области с нормализацией.

    Attributes:
        geometry_dir: Директория с геометрией
        sampler_dir: Директория с точками коллокации
        fluids_dir: Директория с жидкостями
        conditions_dir: Директория с граничными условиями
        output_dir: Директория для выходных файлов
        plots_dir: Директория для графиков
        fluid_lib: Библиотека жидкостей
        cond_manager: Менеджер граничных условий
        points: Загруженные точки коллокации
        points_norm: Нормализованные точки
        wall_points: Точки на стенке для SDF
        wall_tree: KDTree для быстрого поиска расстояний
        fluid: Выбранная жидкость
        conditions: Выбранные граничные условия
        chord: Хорда профиля [м]
        U_inf: Характерная скорость [м/с]
        Re: Число Рейнольдса
    """

    def __init__(self) -> None:
        """Инициализирует класс инициализации расчётной области."""
        self.geometry_dir = GEOMETRY_DIR
        self.sampler_dir = SAMPLER_DIR
        self.fluids_dir = FLUIDS_DIR
        self.conditions_dir = CONDITIONS_DIR
        self.output_dir = OUTPUT_DIR
        self.plots_dir = PLOTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.fluid_lib = FluidLibrary(self.fluids_dir)
        self.cond_manager = ConditionsManager(self.conditions_dir)

        self.points: Optional[np.ndarray] = None
        self.points_norm: Optional[np.ndarray] = None
        self.wall_points: Optional[np.ndarray] = None
        self.wall_tree: Optional[cKDTree] = None  # используется только в однопоточном режиме
        self.fluid: Optional[Fluid] = None
        self.conditions: Optional[BoundaryConditions] = None
        self.fluid_name: Optional[str] = None
        self.conditions_name: Optional[str] = None

        self.chord: Optional[float] = None
        self.U_inf: Optional[float] = None
        self.p_ref: Optional[float] = None
        self.Re: Optional[float] = None
        self.rho: Optional[float] = None
        self.mu: Optional[float] = None
        self.delta_bl_est: Optional[float] = None
        self.delta_bl_est_norm: Optional[float] = None

        self.pressure_mode: Optional[str] = None
        self.velocity_initial_mode: Optional[str] = None
        self.temperature: Optional[float] = None
        self.velocity_perturbation_amplitude: float = 0.01
        self.pressure_perturbation_amplitude: float = 0.0

        self.domain_coords: Optional[np.ndarray] = None
        self.foil_coords: Optional[np.ndarray] = None

    def load_sampler_points(self, filename: Optional[str] = None) -> np.ndarray:
        """
        Загружает точки коллокации из Sampler.

        Args:
            filename: Путь к файлу с точками (по умолчанию выбирается автоматически)

        Returns:
            Массив загруженных точек

        Raises:
            FileNotFoundError: Если нет файлов в директории Sampler
            ValueError: Если файл имеет неверный формат
        """
        if filename is None:
            files = glob.glob(os.path.join(self.sampler_dir, "*.npy"))
            if not files:
                raise FileNotFoundError(f"Нет .npy файлов в {self.sampler_dir}")
            main_files = [f for f in files
                          if not f.endswith('_interior.npy')
                          and not f.endswith('_boundary.npy')
                          and not f.endswith('_metadata.npy')]
            filename = main_files[0] if main_files else files[0]
            print(f"Использую файл точек: {os.path.basename(filename)}")
        data = np.load(filename, allow_pickle=True)

        if data.dtype.names is not None:
            self.points = data
            logger.info("Загружен структурированный массив точек.")
        else:
            if data.ndim == 2 and data.shape[1] == 2:
                arr = np.zeros(len(data), dtype=[('id', 'i4'), ('x', 'f8'), ('y', 'f8')])
                arr['id'] = np.arange(len(data))
                arr['x'] = data[:, 0]
                arr['y'] = data[:, 1]
                self.points = arr
                logger.info("Загружены координаты точек, id присвоены.")
            else:
                raise ValueError("Неизвестный формат файла точек")

        logger.info(f"Загружено {len(self.points)} точек из Sampler.")
        return self.points

    def extract_wall_points(self) -> None:
        """
        Извлекает точки с типом WALL для построения SDF.

        Raises:
            RuntimeError: Если точки не загружены
        """
        if self.points is None:
            raise RuntimeError("Сначала загрузите точки.")
        if 'boundary_type' not in self.points.dtype.names:
            logger.warning("В точках нет информации о boundary_type. Использую все точки (некорректно!)")
            self.wall_points = np.column_stack((self.points['x'], self.points['y']))
            return

        wall_mask = self.points['boundary_type'] == 'wall'
        if not np.any(wall_mask):
            logger.warning("Не найдено ни одной точки с типом WALL. SDF будет нулевым (расстояние до стенки не определено).")
            self.wall_points = np.empty((0, 2))
        else:
            self.wall_points = np.column_stack((self.points['x'][wall_mask], self.points['y'][wall_mask]))
            logger.info(f"Найдено {len(self.wall_points)} точек WALL для построения SDF.")
            if SCIPY_AVAILABLE and len(self.wall_points) > 0:
                self.wall_tree = cKDTree(self.wall_points)
                logger.info("KDTree построен (будет использован в однопоточном режиме).")

    @staticmethod
    def _load_coords(path_base: str) -> Optional[np.ndarray]:
        for ext in (".npy", ".txt", ".csv"):
            path = path_base + ext
            if not os.path.exists(path):
                continue
            try:
                if ext == ".npy":
                    data = np.asarray(np.load(path), dtype=np.float64)
                else:
                    data = np.asarray(
                        np.loadtxt(path, delimiter=',' if ext == ".csv" else None),
                        dtype=np.float64,
                    )
                data = data.reshape(-1, 2)
                if len(data) >= 2:
                    return data
            except Exception as exc:
                logger.debug("Не удалось загрузить координаты из %s: %s", path, exc)
        return None

    @staticmethod
    def _current_geom_name() -> Optional[str]:
        cfg_path = os.path.join(PROJECT_ROOT, "geom_config.yaml")
        if not os.path.exists(cfg_path):
            return None
        try:
            import yaml
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            geom_name = cfg.get('geom_name')
            return str(geom_name) if geom_name else None
        except Exception as exc:
            logger.debug("Не удалось прочитать geom_config.yaml: %s", exc)
            return None

    def _load_geometry_coords(self, suffix: str) -> Optional[np.ndarray]:
        geom_name = self._current_geom_name()
        search_dirs = [self.geometry_dir]
        metadata_dir = os.path.join(PROJECT_ROOT, "Geometry", "Metadata")
        if suffix == "foil":
            search_dirs.append(metadata_dir)

        if geom_name:
            for directory in search_dirs:
                coords = self._load_coords(os.path.join(directory, f"{geom_name}_{suffix}"))
                if coords is not None:
                    return coords

        pattern = os.path.join(self.geometry_dir, f"*_{suffix}.*")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for path in files:
            base, _ = os.path.splitext(path)
            coords = self._load_coords(base)
            if coords is not None:
                return coords
        return None

    def load_domain_coords(self) -> None:
        """Загружает контур расчётного домена для маски визуализации."""
        data = self._load_geometry_coords("domain")
        if data is not None:
            if len(data) > 2 and not np.allclose(data[0], data[-1], atol=1e-12):
                data = np.vstack([data, data[0]])
            self.domain_coords = data
            logger.info("Загружены координаты домена для визуализации.")

    def load_foil_coords(self) -> None:
        """Загружает координаты профиля/пластины для визуализации."""
        data = self._load_geometry_coords("foil")
        if data is not None:
            self.foil_coords = data
            logger.info("Загружены координаты профиля для визуализации.")

    def _load_chord_from_metadata(self) -> float:
        """Определяет хорду из метаданных геометрии или оценивает по разбросу x."""
        metadata_dir = os.path.join(PROJECT_ROOT, "Geometry", "Metadata")
        search_dirs = [metadata_dir, self.geometry_dir]
        geom_name = self._current_geom_name()
        meta_files: List[str] = []

        if geom_name:
            for directory in search_dirs:
                path = os.path.join(directory, f"{geom_name}_metadata.npy")
                if os.path.exists(path):
                    meta_files.append(path)

        if not meta_files:
            seen = set()
            for directory in search_dirs:
                for path in glob.glob(os.path.join(directory, "*_metadata.npy")):
                    if path not in seen:
                        meta_files.append(path)
                        seen.add(path)
            meta_files.sort(key=os.path.getmtime, reverse=True)

        for meta_path in meta_files:
            try:
                meta = np.load(meta_path, allow_pickle=True).item()
                chord = meta.get('chord', None)
                if chord is not None:
                    logger.info(f"Хорда загружена из {os.path.basename(meta_path)}: {chord:.4f} м")
                    return float(chord)
                else:
                    logger.warning(f"Метаданные {os.path.basename(meta_path)} не содержат 'chord', оцениваю по разбросу x.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки метаданных из {os.path.basename(meta_path)}: {e}")

        foil = self.foil_coords if self.foil_coords is not None else self._load_geometry_coords("foil")
        if foil is not None and len(foil) >= 2:
            chord = float(np.max(foil[:, 0]) - np.min(foil[:, 0]))
            if chord > 0.0:
                logger.warning(
                    "Хорда оценена по координатам профиля, metadata не использованы: "
                    f"{chord:.4f} м"
                )
                return chord

        x_range = self.points['x'].max() - self.points['x'].min()
        logger.warning(
            f"Файлы метаданных не найдены или не содержат хорду. "
            f"Используется приближение по разбросу x точек (включает домен, не только хорду): {x_range:.4f} м"
        )
        return float(x_range)

    def select_fluid(self) -> str:
        """
        Интерактивный выбор жидкости из библиотеки.

        Returns:
            Название выбранной жидкости

        Raises:
            RuntimeError: Если библиотека жидкостей пуста
        """
        fluid_name = choose_fluid(self.fluid_lib)
        if fluid_name is None:
            raise RuntimeError("Жидкость не выбрана.")
        self.fluid = self.fluid_lib.get(fluid_name)
        self.fluid_name = fluid_name
        return fluid_name

    def select_conditions(self) -> str:
        """
        Интерактивный выбор граничных условий.

        Returns:
            Название выбранных условий

        Raises:
            RuntimeError: Если библиотека условий пуста
        """
        names = self.cond_manager.list_names()
        if not names:
            raise RuntimeError("Библиотека условий пуста.")
        print("\nДоступные наборы условий:")
        for i, name in enumerate(names, 1):
            cond = self.cond_manager.get(name)
            print(f"{i}. {name}")
            print(f"   {cond.display(self.fluid_lib)}")
        while True:
            try:
                choice = input("Выберите номер условий: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    self.conditions = self.cond_manager.get(names[idx])
                    self.conditions_name = names[idx]
                    return names[idx]
                else:
                    print("Неверный номер.")
            except ValueError:
                print("Введите число.")

    def _compute_normalization_params(self) -> None:
        """Вычисляет параметры нормализации на основе выбранных жидкости и условий."""
        if self.fluid is None or self.conditions is None:
            raise RuntimeError("Сначала выберите жидкость и условия.")

        self.chord = self._load_chord_from_metadata()
        u_in = self.conditions.u
        v_in = self.conditions.v
        self.U_inf = np.sqrt(u_in**2 + v_in**2)
        if self.U_inf == 0:
            self.U_inf = 1.0
            logger.warning("⚠️ СКОРОСТЬ НА ВХОДЕ РАВНА 0! Использую U_inf = 1 для нормализации. "
                          "Это может привести к нефизичным результатам. Рекомендуется изменить граничные условия.")
        self.p_ref = self.conditions.outlet_pressure
        self.rho = self.fluid.density
        self.mu = self.fluid.dynamic_viscosity
        nu = self.fluid.kinematic_viscosity
        self.Re = self.U_inf * self.chord / nu

        logger.info(f"Хорда: {self.chord:.4f} м")
        logger.info(f"U_inf = {self.U_inf:.4f} м/с")
        logger.info(f"Re = {self.Re:.2e}")
        logger.info(f"p_ref = {self.p_ref:.1f} Па (давление на выходе)")

    def print_sdf_diagnostics_after_conditions(self) -> None:
        """
        Выводит оценку delta и SDF-статистику сразу после выбора условий.

        На этом этапе поля ещё не нормализованы, но уже известны жидкость,
        скорость на входе и Re. Статистика считается для interior-точек,
        чтобы WALL-точки с SDF=0 не маскировали разрешение пристенной области.
        """
        if self.points is None:
            raise RuntimeError("Сначала загрузите точки.")
        if self.wall_points is None:
            self.extract_wall_points()

        self._compute_normalization_params()
        self.delta_bl_est, self.delta_bl_est_norm = estimate_boundary_layer_delta(
            self.chord,
            self.Re,
        )

        print("\n" + "=" * 60)
        print("SDF-ДИАГНОСТИКА ПРИСТЕННОЙ СЕТКИ".center(60))
        print("=" * 60)
        if self.delta_bl_est is not None and self.delta_bl_est_norm is not None:
            print(
                "Оценка толщины слоя: "
                f"delta_99/c ~= 5/sqrt(Re) = {self.delta_bl_est_norm:.6g}, "
                f"delta_99 = {self.delta_bl_est:.6g} м"
            )
        else:
            print("WARNING: Не удалось оценить delta_99: проверьте хорду и Re.")

        if self.wall_points is None or len(self.wall_points) == 0:
            print("WARNING: Нет WALL-точек: SDF-статистика не рассчитана.")
            return

        if 'boundary_type' in self.points.dtype.names:
            interior_mask = self.points['boundary_type'] == 'interior'
        elif 'on_boundary' in self.points.dtype.names:
            interior_mask = ~self.points['on_boundary']
        else:
            interior_mask = np.ones(len(self.points), dtype=bool)

        if not np.any(interior_mask):
            print("WARNING: Нет interior-точек для SDF-статистики.")
            return

        query_xy = np.column_stack((
            self.points['x'][interior_mask],
            self.points['y'][interior_mask],
        ))
        sdf_m = compute_nearest_wall_distances(query_xy, self.wall_points)
        summary = compute_sdf_summary(
            sdf_m,
            chord=self.chord,
            delta=self.delta_bl_est,
        )
        for line in format_sdf_summary(
            summary,
            title="SDF-статистика initial-сетки (interior -> WALL)",
        ):
            print(line)

    def initialize_fields(self,
                          pressure_mode: str = 'constant',
                          velocity_initial_mode: str = 'constant',
                          temperature: float = 293.15,
                          perturbation_amplitude: float = 0.01,
                          velocity_perturbation_amplitude: Optional[float] = None,
                          pressure_perturbation_amplitude: float = 0.0,
                          num_processes: Optional[int] = None) -> np.ndarray:
        """
        Инициализирует нормализованные поля.

        Args:
            pressure_mode: 'constant' или 'linear_x'
            velocity_initial_mode: 'constant', 'linear_x', 'perturbed'
            temperature: температура [K]
            perturbation_amplitude: относительная амплитуда шума для скорости (доля от U_inf)
            velocity_perturbation_amplitude: если None, используется perturbation_amplitude
            pressure_perturbation_amplitude: относительная амплитуда шума для давления
                                             (доля от динамического напора 0.5*rho*U_inf^2)
            num_processes: количество процессов для параллельного вычисления SDF

        Returns:
            Нормализованный массив точек.
        """
        if self.points is None:
            raise RuntimeError("Сначала загрузите точки.")
        if self.fluid is None:
            raise RuntimeError("Сначала выберите жидкость.")
        if self.conditions is None:
            raise RuntimeError("Сначала выберите граничные условия.")
        if self.wall_points is None:
            self.extract_wall_points()

        self.pressure_mode = pressure_mode
        self.velocity_initial_mode = velocity_initial_mode
        self.temperature = temperature

        if velocity_perturbation_amplitude is not None:
            self.velocity_perturbation_amplitude = velocity_perturbation_amplitude
        else:
            self.velocity_perturbation_amplitude = perturbation_amplitude
        self.pressure_perturbation_amplitude = pressure_perturbation_amplitude

        self._compute_normalization_params()
        self.delta_bl_est, self.delta_bl_est_norm = estimate_boundary_layer_delta(
            self.chord,
            self.Re,
        )

        N = len(self.points)
        arr_norm = np.zeros(N, dtype=point_dtype)

        # Копируем существующие поля, если они есть
        for name in ['id', 'boundary_type', 'on_boundary']:
            if name in self.points.dtype.names:
                arr_norm[name] = self.points[name]

        if 'id' not in self.points.dtype.names:
            arr_norm['id'] = np.arange(N)

        arr_norm['x_norm'] = self.points['x'] / self.chord
        arr_norm['y_norm'] = self.points['y'] / self.chord

        # --- Генерация начального поля скорости ---
        x = self.points['x']
        x_min = np.min(x)
        x_max = np.max(x)
        u0 = self.conditions.u
        v0 = self.conditions.v

        if velocity_initial_mode == 'constant':
            u_phys = np.full(N, u0)
            v_phys = np.full(N, v0)
        elif velocity_initial_mode == 'linear_x':
            if x_max > x_min:
                slope = -0.5 * u0 / (x_max - x_min)  # убывание на 50% на длине области
                u_phys = u0 + slope * (x - x_min)
                v_phys = np.full(N, v0)
            else:
                u_phys = np.full(N, u0)
                v_phys = np.full(N, v0)
        elif velocity_initial_mode == 'perturbed':
            amp = self.velocity_perturbation_amplitude
            U_mag = self.U_inf  # модуль скорости — не зависит от направления
            u_phys = np.full(N, u0) + amp * U_mag * (2 * np.random.random(N) - 1)
            v_phys = np.full(N, v0) + amp * U_mag * (2 * np.random.random(N) - 1)
        else:
            raise ValueError(f"Неизвестный режим инициализации скорости: {velocity_initial_mode}")

        # --- Давление (базовое распределение) ---
        if pressure_mode == 'constant':
            p_phys = np.full(N, self.conditions.inlet_pressure)
        elif pressure_mode == 'linear_x':
            if x_max > x_min:
                slope = (self.conditions.outlet_pressure - self.conditions.inlet_pressure) / (x_max - x_min)
                p_phys = self.conditions.inlet_pressure + slope * (x - x_min)
            else:
                p_phys = np.full(N, self.conditions.inlet_pressure)
        else:
            raise ValueError(f"Неизвестный режим давления: {pressure_mode}")

        # Добавление шума к давлению
        if self.pressure_perturbation_amplitude > 0:
            q_ref = 0.5 * self.rho * self.U_inf**2
            p_phys += self.pressure_perturbation_amplitude * q_ref * (2 * np.random.random(N) - 1)

        # Применяем граничные условия к начальному полю скорости
        if 'boundary_type' in self.points.dtype.names:
            wall_mask = self.points['boundary_type'] == 'wall'
            u_phys[wall_mask] = 0.0
            v_phys[wall_mask] = 0.0

            sym_mask = self.points['boundary_type'] == 'symmetry'
            v_phys[sym_mask] = 0.0

            axis_mask = self.points['boundary_type'] == 'axis_symmetry'
            u_phys[axis_mask] = 0.0

        # Нормализация
        arr_norm['u_norm'] = u_phys / self.U_inf
        arr_norm['v_norm'] = v_phys / self.U_inf
        arr_norm['p_norm'] = (p_phys - self.p_ref) / (0.5 * self.rho * self.U_inf**2)

        # --- Вычисление SDF (с поддержкой многопроцессорности) ---
        if len(self.wall_points) == 0:
            logger.warning("Нет точек WALL, SDF будет нулевым.")
            sdf_phys = np.zeros(N)
        else:
            logger.info("Вычисление SDF...")
            xs = self.points['x'].copy()
            ys = self.points['y'].copy()

            if num_processes is None:
                num_processes = mp.cpu_count()
            num_processes = max(1, min(int(num_processes), N, 16))

            if num_processes > 1 and mp.cpu_count() > 1:
                logger.info(f"Используется процессов: {num_processes}")
                indices = np.arange(N)
                chunks = [chunk for chunk in np.array_split(indices, num_processes) if len(chunk) > 0]
                with mp.Pool(processes=num_processes) as pool:
                    func = partial(_compute_sdf_chunk,
                                   xs=xs, ys=ys,
                                   wall_points=self.wall_points,
                                   use_scipy=SCIPY_AVAILABLE)
                    results = pool.map(func, chunks)

                sdf_phys = np.zeros(N)
                for chunk_idx, sdf_vals in zip(chunks, results):
                    sdf_phys[chunk_idx] = sdf_vals
            else:
                # Однопоточный режим с использованием заранее построенного дерева
                logger.info("Однопоточный режим.")
                if SCIPY_AVAILABLE and self.wall_tree is not None:
                    sdf_phys, _ = self.wall_tree.query(np.column_stack((xs, ys)), k=1)
                    sdf_phys = np.asarray(sdf_phys, dtype=np.float64)
                else:
                    sdf_phys = _compute_sdf_chunk(
                        np.arange(N),
                        xs=xs,
                        ys=ys,
                        wall_points=self.wall_points,
                        use_scipy=False,
                    )

        arr_norm['sdf_norm'] = sdf_phys / self.chord

        self.points_norm = arr_norm
        logger.info("Нормализованные поля инициализированы.")
        return arr_norm

    def save(self, description: str = "initial") -> None:
        """
        Сохраняет нормализованные данные и метаданные.

        Args:
            description: Описание для сохранения

        Raises:
            RuntimeError: Если нет данных для сохранения
        """
        if self.points_norm is None:
            raise RuntimeError("Нет данных для сохранения.")
        base = os.path.join(self.output_dir, f"{description}.npy")
        np.save(base, self.points_norm)
        logger.info(f"Сохранено: {base}")

        # Вычисляем нормализованные значения скорости на входе
        inlet_u_norm = self.conditions.u / self.U_inf if self.U_inf > 0 else 1.0
        inlet_v_norm = self.conditions.v / self.U_inf if self.U_inf > 0 else 0.0

        meta = {
            'description': description,
            'fluid_name': self.fluid_name,
            'conditions_name': self.conditions_name,
            'temperature': self.temperature,
            'pressure_mode': self.pressure_mode,
            'velocity_initial_mode': self.velocity_initial_mode,
            'velocity_perturbation_amplitude': self.velocity_perturbation_amplitude,
            'pressure_perturbation_amplitude': self.pressure_perturbation_amplitude,
            'num_points': len(self.points_norm),
            'chord': self.chord,
            'U_inf': self.U_inf,
            'p_ref': self.p_ref,
            'rho': self.rho,
            'mu': self.mu,
            'Re': self.Re,
            'delta_bl_99_est': self.delta_bl_est,
            'delta_bl_99_est_norm': self.delta_bl_est_norm,
            'normalized': True,
            'sdf_norm_factor': self.chord,
            'inlet_u_norm': inlet_u_norm,
            'inlet_v_norm': inlet_v_norm,
        }
        meta_path = os.path.join(self.output_dir, f"{description}_metadata.npy")
        np.save(meta_path, meta)
        logger.info(f"Метаданные сохранены: {meta_path}")

    @staticmethod
    def _deduplicate_field_points(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.column_stack([x, y])
        rounded = np.round(xy, decimals=12)
        uniq, inverse = np.unique(rounded, axis=0, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        val_sum = np.bincount(inverse, weights=values.astype(np.float64))
        return uniq[:, 0], uniq[:, 1], val_sum / np.maximum(counts, 1.0)

    @staticmethod
    def _clean_polyline(polyline: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if polyline is None:
            return None
        line = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(line).all(axis=1)
        line = line[finite]
        if len(line) < 2:
            return None
        keep = np.ones(len(line), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-12
        line = line[keep]
        return line if len(line) >= 2 else None

    @classmethod
    def _clip_polygon(cls, clip_polygon: Optional[np.ndarray]) -> Optional[np.ndarray]:
        poly = cls._clean_polyline(clip_polygon)
        if poly is None or len(poly) < 3:
            return None
        if not np.allclose(poly[0], poly[-1], atol=1e-12):
            poly = np.vstack([poly, poly[0]])
        return poly

    @classmethod
    def _closed_mask_polygon(cls, mask_polygon: Optional[np.ndarray]) -> Optional[np.ndarray]:
        poly = cls._clean_polyline(mask_polygon)
        if poly is None or len(poly) < 3:
            return None
        if not np.allclose(poly[0], poly[-1], atol=1e-12):
            return None
        return poly

    @classmethod
    def _polyline(cls, polyline: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return cls._clean_polyline(polyline)

    @staticmethod
    def _points_inside_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
        try:
            from matplotlib.path import Path
            return Path(poly).contains_points(pts).reshape(np.shape(x))
        except Exception:
            return np.zeros(np.shape(x), dtype=bool)

    @classmethod
    def _apply_grid_masks(
        cls,
        Xi: np.ndarray,
        Yi: np.ndarray,
        Zi: np.ndarray,
        clip_poly: Optional[np.ndarray],
        mask_poly: Optional[np.ndarray],
    ) -> np.ndarray:
        masked = np.array(Zi, copy=True)
        if clip_poly is not None:
            masked[~cls._points_inside_polygon(Xi, Yi, clip_poly)] = np.nan
        if mask_poly is not None:
            masked[cls._points_inside_polygon(Xi, Yi, mask_poly)] = np.nan
        return masked

    @staticmethod
    def _plot_smooth_field(
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        label: str,
        title: str,
        cmap: str = 'jet',
        clip_polygon: Optional[np.ndarray] = None,
        mask_polygon: Optional[np.ndarray] = None,
        outline_polylines: Optional[List[np.ndarray]] = None,
        grid_size: int = 420,
        levels: int = 96,
    ) -> None:
        """Рисует гладкую карту поля вместо точечной карты."""
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        values = np.asarray(values, dtype=np.float64).flatten()
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        x, y, values = x[finite], y[finite], values[finite]
        if len(values) == 0:
            raise ValueError("нет конечных значений поля")

        x, y, values = Initializer._deduplicate_field_points(x, y, values)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if abs(vmax - vmin) < 1e-30:
            pad = max(abs(vmax), 1.0) * 1e-6
            vmin -= pad
            vmax += pad
        level_values = np.linspace(vmin, vmax, levels)

        artist = None
        clip_poly = Initializer._clip_polygon(clip_polygon)
        mask_poly = Initializer._closed_mask_polygon(mask_polygon)
        if SCIPY_AVAILABLE and len(values) >= 4:
            xi = np.linspace(float(x.min()), float(x.max()), grid_size)
            yi = np.linspace(float(y.min()), float(y.max()), grid_size)
            Xi, Yi = np.meshgrid(xi, yi)
            method = 'cubic' if len(values) >= 16 else 'linear'
            try:
                Zi = griddata((x, y), values, (Xi, Yi), method=method)
                if np.all(~np.isfinite(Zi)):
                    Zi = griddata((x, y), values, (Xi, Yi), method='linear')
                if np.any(np.isfinite(Zi)):
                    Zi = Initializer._apply_grid_masks(Xi, Yi, Zi, clip_poly, mask_poly)
                    artist = plt.contourf(Xi, Yi, Zi, levels=level_values,
                                          cmap=cmap, extend='both')
            except Exception as exc:
                logger.debug("griddata не сработал для %s: %s", title, exc)

        if artist is None:
            try:
                import matplotlib.tri as mtri
                triang = mtri.Triangulation(x, y)
                if clip_poly is not None or mask_poly is not None:
                    tri_mask = np.zeros(len(triang.triangles), dtype=bool)
                    centroids_x = x[triang.triangles].mean(axis=1)
                    centroids_y = y[triang.triangles].mean(axis=1)
                    if clip_poly is not None:
                        tri_mask |= ~Initializer._points_inside_polygon(centroids_x, centroids_y, clip_poly)
                    if mask_poly is not None:
                        tri_mask |= Initializer._points_inside_polygon(centroids_x, centroids_y, mask_poly)
                    triang.set_mask(tri_mask)
                artist = plt.tricontourf(triang, values, levels=level_values,
                                         cmap=cmap, extend='both')
            except Exception as exc:
                raise RuntimeError(f"Не удалось построить гладкую заливку для поля {title}") from exc

        if clip_poly is not None:
            plt.plot(clip_poly[:, 0], clip_poly[:, 1], color='black', linewidth=0.9)
        if outline_polylines:
            for line in outline_polylines:
                line_arr = Initializer._polyline(line)
                if line_arr is not None:
                    plt.plot(line_arr[:, 0], line_arr[:, 1], color='black', linewidth=1.2)
        elif mask_polygon is not None:
            line_arr = Initializer._polyline(mask_polygon)
            if line_arr is not None:
                plt.plot(line_arr[:, 0], line_arr[:, 1], color='black', linewidth=1.2)

        plt.colorbar(artist, label=label)
        plt.title(title)
        plt.xlabel('x, м')
        plt.ylabel('y, м')
        plt.axis('equal')
        plt.grid(alpha=0.15)

    def plot_field(
        self,
        field_name: str,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False
    ) -> None:
        """
        Строит гладкую карту указанного поля.

        Args:
            field_name: Имя поля для визуализации
            title: Заголовок графика
            save: Сохранить ли график
            show: Показать ли график сразу
        """
        if self.points_norm is None:
            raise RuntimeError("Нет данных для визуализации.")
        plt.figure(figsize=(12, 8))
        x = self.points_norm['x_norm'] * self.chord
        y = self.points_norm['y_norm'] * self.chord

        if field_name == 'sdf_norm':
            values = self.points_norm['sdf_norm'] * self.chord
            label = 'SDF (расстояние до стенки, м)'
        else:
            values = self.points_norm[field_name]
            label = field_name

        self._plot_smooth_field(
            x,
            y,
            values,
            label=label,
            title=title or f"Поле {field_name}",
            cmap='jet',
            clip_polygon=self.domain_coords,
            mask_polygon=self.foil_coords,
            outline_polylines=[self.foil_coords] if self.foil_coords is not None else None,
        )

        if save:
            filename = os.path.join(self.plots_dir, f"{field_name}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранён: {filename}")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_fields(self) -> None:
        """
        Строит графики для всех основных полей.

        Визуализирует: p_norm, u_norm, v_norm, sdf_norm.
        """
        self.load_domain_coords()
        self.load_foil_coords()
        fields = [
            ('p_norm', 'Нормализованное давление'),
            ('u_norm', 'Нормализованная скорость U'),
            ('v_norm', 'Нормализованная скорость V'),
            ('sdf_norm', 'SDF (расстояние до стенки, м)')
        ]
        for fname, label in fields:
            self.plot_field(fname, title=label)


def initialize_menu() -> None:
    """
    Интерактивное меню для запуска инициализации.

    Example:
        >>> if __name__ == "__main__":
        ...     initialize_menu()
    """
    init = Initializer()

    print("\n" + "=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ РАСЧЁТНОЙ ОБЛАСТИ".center(60))
    print("=" * 60)

    try:
        init.load_sampler_points()
        init.extract_wall_points()
        if len(init.wall_points) == 0:
            print("⚠️ Не найдено точек WALL. SDF будет нулевым (расстояние до стенки не определено).")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        input("Нажмите Enter для возврата...")
        return

    try:
        fluid_name = init.select_fluid()
        print(f"Выбрана жидкость: {fluid_name}")
    except Exception as e:
        print(f"❌ {e}")
        input("Нажмите Enter для возврата...")
        return

    try:
        cond_name = init.select_conditions()
        print(f"Выбраны условия: {cond_name}")
    except Exception as e:
        print(f"❌ {e}")
        input("Нажмите Enter для возврата...")
        return

    try:
        init.print_sdf_diagnostics_after_conditions()
    except Exception as e:
        print(f"WARNING: Не удалось вывести SDF-диагностику: {e}")

    print("\nПараметры инициализации:")
    print("Режим давления:")
    print("   1 - постоянное")
    print("   2 - линейное по X")
    press_mode_choice = input("Выберите (1/2): ").strip()
    pressure_mode = 'constant' if press_mode_choice == '1' else 'linear_x'

    print("\nРежим начального поля скорости:")
    print("   1 - постоянное (равное входной скорости)")
    print("   2 - линейное убывание по X")
    print("   3 - постоянное + случайный шум")
    vel_choice = input("Выберите (1/2/3): ").strip()
    vel_map = {'1': 'constant', '2': 'linear_x', '3': 'perturbed'}
    velocity_initial_mode = vel_map.get(vel_choice, 'constant')

    velocity_perturbation_amplitude = 0.01
    if velocity_initial_mode == 'perturbed':
        try:
            amp = float(input("Амплитуда шума для скорости (относительная, по умолч. 0.01): ").strip() or "0.01")
            velocity_perturbation_amplitude = amp
        except ValueError:
            velocity_perturbation_amplitude = 0.01
            print("Использую значение по умолчанию 0.01")

    add_pressure_noise = input("\nДобавить шум к давлению? (y/N): ").strip().lower()
    pressure_perturbation_amplitude = 0.0
    if add_pressure_noise in ('y', 'yes'):
        try:
            amp = float(input("Амплитуда шума для давления (относительная доля от динамического напора, по умолч. 0.1): ").strip() or "0.1")
            pressure_perturbation_amplitude = amp
        except ValueError:
            pressure_perturbation_amplitude = 0.1
            print("Использую значение по умолчанию 0.1")

    try:
        temp = float(input("Температура [K] (по умолчанию 293.15): ").strip() or "293.15")
    except ValueError:
        temp = 293.15
        print("Использую значение по умолчанию 293.15 K")

    try:
        num_proc_input = input("Количество процессов для параллельных вычислений (Enter для авто): ").strip()
        num_proc = int(num_proc_input) if num_proc_input else None
    except ValueError:
        num_proc = None
        print("Использую автоматическое определение числа процессов.")

    print("\n🔧 Вычисление нормализованных полей...")
    try:
        init.initialize_fields(pressure_mode=pressure_mode,
                               velocity_initial_mode=velocity_initial_mode,
                               temperature=temp,
                               velocity_perturbation_amplitude=velocity_perturbation_amplitude,
                               pressure_perturbation_amplitude=pressure_perturbation_amplitude,
                               num_processes=num_proc)
    except Exception as e:
        print(f"❌ Ошибка при вычислении полей: {e}")
        input("Нажмите Enter для возврата...")
        return

    desc = input("Введите описание для сохранения (по умолчанию 'initial'): ").strip() or "initial"
    init.save(description=desc)

    plot_choice = input("Построить графики полей? (Y/n): ").strip().lower()
    if plot_choice == '' or plot_choice == 'y' or plot_choice == 'yes':
        init.plot_all_fields()
        print(f"Графики сохранены в {init.plots_dir}")

    print("\n✅ Инициализация завершена. Нормализованные данные готовы для PINN.")
    input("Нажмите Enter для возврата...")


if __name__ == "__main__":
    initialize_menu()
