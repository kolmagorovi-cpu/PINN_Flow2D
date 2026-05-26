#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль дискретизации S2DR (Sampler 2D).

Генерация точек коллокации для PINNs с классификацией граничных условий.

Область задаётся внешним прямоугольником (domain) и профилем (foil).
Точки на внешних границах, на профиле и внутри области генерируются раздельно.

Поддерживаются методы:
    - regular: равномерная сетка
    - irregular: равномерное случайное распределение
    - adaptive_foil: сгущение у профиля (на основе расстояния до профиля)
    - adaptive_all: сгущение ко всем границам (на основе расстояния до ближайшей границы)

Все методы поддерживают многопроцессорность (с корректной работой на Windows).
Пользователь может задать параметры alpha (степень сгущения) и epsilon_factor (масштаб сгущения),
а также выбрать тип верхней границы (INLET или OUTLET).

Example:
    >>> config = SamplerConfig(step_size=0.01, num_processes=4)
    >>> sampler = CollocationPoints(config)
    >>> domain, foil = sampler.load_geometries()
    >>> sampler.generate_all_points(domain, foil)
    >>> sampler.save_points()

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная)
"""

import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from matplotlib.path import Path
from multiprocessing import Process, Queue, freeze_support
import os
import time
import logging
import yaml
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict
import sys
import queue
from scipy.spatial import cKDTree

from type_hints import (
    CollocationPointDict,
    SamplerConfigDict,
)

# Попытка импорта tqdm для прогресс-бара
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEOMETRY_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
DEFAULT_METADATA_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Metadata")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Sampler", "Data")

# Импортируем константы после определения локальных констант
from constants import (
    CLASSIFICATION_TOLERANCE_FACTOR,
    MAX_ATTEMPTS_MULTIPLIER,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_PROCESSES,
    QUEUE_TIMEOUT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    AXIS_SYMMETRY = "axis_symmetry"
    INTERIOR = "interior"


@dataclass
class CollocationPoint:
    """
    Точка коллокации с классификацией граничных условий.

    Attributes:
        id: Уникальный идентификатор точки
        x: X-координата [м]
        y: Y-координата [м]
        boundary_type: Тип границы
        on_airfoil: Находится ли точка на профиле
        on_boundary: Находится ли точка на границе
        segment_id: ID сегмента границы
    """
    id: int
    x: float
    y: float
    boundary_type: BoundaryType
    on_airfoil: bool = False
    on_boundary: bool = False
    segment_id: int = -1

    def to_dict(self) -> CollocationPointDict:
        """
        Преобразует точку в словарь.

        Returns:
            Словарь с данными точки
        """
        return CollocationPointDict(
            id=self.id,
            x=self.x,
            y=self.y,
            boundary_type=self.boundary_type.value,
            on_airfoil=self.on_airfoil,
            on_boundary=self.on_boundary,
            segment_id=self.segment_id
        )

    def to_tuple(self) -> Tuple[float, float]:
        """
        Возвращает координаты точки в виде кортежа.

        Returns:
            Кортеж (x, y)
        """
        return (self.x, self.y)


_VALID_BOUNDARY_TYPES = ('inlet', 'outlet', 'symmetry', 'axis_symmetry', 'wall')


@dataclass
class SamplerConfig:
    """
    Конфигурация для дискретизации расчётной области.

    Attributes:
        num_processes: Количество процессов для параллельных вычислений
        step_size: Шаг дискретизации
        step_units: Единицы шага ('m' или 'mm')
        use_multiprocessing: Использовать ли многопроцессорность
        show_progress: Показывать ли прогресс-бар
        save_format: Формат сохранения ('npy', 'csv')
        plot_results: Построить ли результаты
        plot_dpi: DPI для графиков
        geometry_dir: Директория с геометрией
        output_dir: Директория для выходных файлов
        boundary_point_density: Плотность точек на границах
        interior_point_density: Плотность внутренних точек
        left_boundary_type: Тип левой границы (x=x_min); одно из: inlet, outlet, symmetry, axis_symmetry, wall
        right_boundary_type: Тип правой границы (x=x_max)
        top_boundary_type: Тип верхней границы (y=y_max)
        bottom_boundary_type: Тип нижней границы (y=y_min)
        sampling_method: Метод генерации ('regular', 'irregular', 'adaptive_foil', 'adaptive_all')
        adaptive_alpha: Степень сгущения (для адаптивных методов)
        adaptive_epsilon_factor: Масштаб сгущения (для адаптивных методов)
        geom_type: Тип геометрии (только 'airfoil')
        geom_name: Имя конфигурации (например, 'VMFL010')
    """
    num_processes: int = 4
    step_size: float = 0.01
    step_units: str = 'm'
    use_multiprocessing: bool = True
    show_progress: bool = True
    save_format: str = 'npy'
    plot_results: bool = True
    plot_dpi: int = 300
    geometry_dir: str = DEFAULT_GEOMETRY_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    boundary_point_density: float = 1.0
    interior_point_density: float = 1.0
    left_boundary_type: str = "inlet"
    right_boundary_type: str = "outlet"
    top_boundary_type: str = "outlet"
    bottom_boundary_type: str = "symmetry"
    sampling_method: str = "regular"
    adaptive_alpha: float = 1.0
    adaptive_epsilon_factor: float = 0.01

    # Тип геометрии (только 'airfoil')
    geom_type: str = "airfoil"
    geom_name: str = None

    def __post_init__(self) -> None:
        """
        Инициализация после создания dataclass.

        Создаёт директории и проверяет корректность типов границ.

        Raises:
            ValueError: Если любой из типов границ не входит в допустимые значения
        """
        os.makedirs(self.geometry_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        for field_name, value in [
            ('left_boundary_type', self.left_boundary_type),
            ('right_boundary_type', self.right_boundary_type),
            ('top_boundary_type', self.top_boundary_type),
            ('bottom_boundary_type', self.bottom_boundary_type),
        ]:
            if value not in _VALID_BOUNDARY_TYPES:
                raise ValueError(
                    f"{field_name}='{value}' не входит в допустимые значения: {_VALID_BOUNDARY_TYPES}"
                )

    def to_dict(self) -> SamplerConfigDict:
        """
        Преобразует конфигурацию в словарь.

        Returns:
            Словарь с параметрами конфигурации
        """
        return SamplerConfigDict(
            num_processes=self.num_processes,
            step_size=self.step_size,
            step_units=self.step_units,
            use_multiprocessing=self.use_multiprocessing,
            show_progress=self.show_progress,
            save_format=self.save_format,
            plot_results=self.plot_results,
            plot_dpi=self.plot_dpi,
            geometry_dir=self.geometry_dir,
            output_dir=self.output_dir,
            boundary_point_density=self.boundary_point_density,
            interior_point_density=self.interior_point_density,
            left_boundary_type=self.left_boundary_type,
            right_boundary_type=self.right_boundary_type,
            top_boundary_type=self.top_boundary_type,
            bottom_boundary_type=self.bottom_boundary_type,
            sampling_method=self.sampling_method,  # type: ignore
            adaptive_alpha=self.adaptive_alpha,
            adaptive_epsilon_factor=self.adaptive_epsilon_factor
        )


# ----------------------------------------------------------------------
# Вспомогательные функции для многопроцессорной обработки (вынесены на уровень модуля)
# ----------------------------------------------------------------------

def _regular_worker(
    domain: np.ndarray,
    step: float,
    bounds: Tuple[float, float, float, float],
    proc_id: int,
    total_procs: int,
    result_queue: Queue,
    progress_queue: Queue
) -> None:
    """
    Worker для регулярной сетки (использует векторизованную проверку Path).

    Args:
        domain: Массив точек области
        step: Шаг сетки
        bounds: Границы (x_min, x_max, y_min, y_max)
        proc_id: ID процесса
        total_procs: Общее количество процессов
        result_queue: Очередь для результатов
        progress_queue: Очередь для прогресса
    """
    # Создаём Path для области
    path = Path(domain)
    x_min, x_max, y_min, y_max = bounds
    # Генерируем сетку
    x_coords = np.arange(x_min, x_max, step)
    y_coords = np.arange(y_min, y_max, step)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.ravel(), Y.ravel()))
    # Проверяем, какие точки внутри области
    inside = path.contains_points(points)
    valid_points = points[inside]
    # Отправляем результат
    result_queue.put(valid_points)
    progress_queue.put((proc_id, len(valid_points), len(points)))


def _irregular_worker(
    domain: np.ndarray,
    bounds: Tuple[float, float, float, float],
    target: int,
    proc_id: int,
    result_queue: Queue,
    progress_queue: Queue
) -> None:
    """
    Worker для нерегулярной (случайной) сетки.

    Args:
        domain: Массив точек области
        bounds: Границы (x_min, x_max, y_min, y_max)
        target: Целевое количество точек
        proc_id: ID процесса
        result_queue: Очередь для результатов
        progress_queue: Очередь для прогресса
    """
    path = Path(domain)
    x_min, x_max, y_min, y_max = bounds
    points = []
    attempts = 0
    max_attempts = target * MAX_ATTEMPTS_MULTIPLIER
    last_report = 0
    while len(points) < target and attempts < max_attempts:
        # Генерируем батч случайных точек
        batch_size = min(DEFAULT_BATCH_SIZE, max_attempts - attempts)
        xs = np.random.uniform(x_min, x_max, batch_size)
        ys = np.random.uniform(y_min, y_max, batch_size)
        candidates = np.column_stack((xs, ys))
        inside = path.contains_points(candidates)
        valid = candidates[inside]
        for pt in valid:
            if len(points) < target:
                points.append(tuple(pt))
            else:
                break
        if len(points) - last_report >= 100:
            progress_queue.put((proc_id, len(points), target))
            last_report = len(points)
        attempts += batch_size
    progress_queue.put((proc_id, len(points), target))
    result_queue.put(points)


def _adaptive_worker(
    domain: np.ndarray,
    boundary_points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    target: int,
    alpha: float,
    epsilon_factor: float,
    proc_id: int,
    result_queue: Queue,
    progress_queue: Queue
) -> None:
    """
    Worker для адаптивного метода (сгущение к заданным граничным точкам).

    Args:
        domain: Массив точек области
        boundary_points: Массив координат точек, к которым выполняется сгущение
        bounds: Границы (x_min, x_max, y_min, y_max)
        target: Целевое количество точек
        alpha: Степень сгущения
        epsilon_factor: Масштаб сгущения
        proc_id: ID процесса
        result_queue: Очередь для результатов
        progress_queue: Очередь для прогресса
    """
    path = Path(domain)
    # Строим дерево внутри процесса
    tree = cKDTree(boundary_points)
    x_min, x_max, y_min, y_max = bounds
    char_length = (x_max - x_min + y_max - y_min) / 2
    eps = char_length * epsilon_factor

    points = []
    batch_size = DEFAULT_BATCH_SIZE
    max_attempts = target * MAX_ATTEMPTS_MULTIPLIER
    total_attempts = 0
    last_report = 0

    while len(points) < target and total_attempts < max_attempts:
        xs = np.random.uniform(x_min, x_max, batch_size)
        ys = np.random.uniform(y_min, y_max, batch_size)
        candidates = np.column_stack((xs, ys))
        inside = path.contains_points(candidates)
        valid = candidates[inside]
        if len(valid) == 0:
            total_attempts += batch_size
            continue
        dists, _ = tree.query(valid)
        probs = (eps / (dists + eps)) ** alpha
        accept = np.random.random(len(valid)) < probs
        accepted = valid[accept]
        for pt in accepted:
            if len(points) < target:
                points.append(tuple(pt))
            else:
                break
        if len(points) - last_report >= 100:
            progress_queue.put((proc_id, len(points), target))
            last_report = len(points)
        total_attempts += batch_size
    progress_queue.put((proc_id, len(points), target))
    result_queue.put(points)


class CollocationPoints:
    """
    Класс для генерации точек коллокации с классификацией граничных условий.

    Attributes:
        config: Конфигурация дискретизации
        points: Список всех точек
        points_interior: Список внутренних точек
        points_boundary: Список граничных точек
        metadata: Метаданные расчётной области
        foil_metadata: Метаданные профиля
        foil_coords: Координаты профиля
        domain_coords: Координаты области
        foil_is_closed: Флаг замкнутости профиля
    """

    def __init__(self, config: SamplerConfig) -> None:
        """
        Инициализирует генератор точек коллокации.

        Args:
            config: Конфигурация дискретизации
        """
        self.config = config
        self.points: List[CollocationPoint] = []
        self.points_interior: List[CollocationPoint] = []
        self.points_boundary: List[CollocationPoint] = []
        self.metadata: Dict[str, Any] = {}
        self.foil_metadata: Dict[str, Any] = {}
        self.next_id = 0
        self.foil_coords: Optional[np.ndarray] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.foil_is_closed = False
        self.output_base: Optional[str] = None

        # Загрузка конфигурации геометрии
        self.load_geometry_config()

        os.makedirs(config.output_dir, exist_ok=True)

    def _get_next_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def load_geometry_config(self, config_path: str = None) -> None:
        """
        Загрузка конфигурации геометрии из YAML-файла.

        Args:
            config_path: Путь к geom_config.yaml (по умолчанию:
                         'Airfoil PINN/geom_config.yaml')
        """
        if config_path is None:
            # Ищем файл в директории проекта
            possible_paths = [
                os.path.join(PROJECT_ROOT, "geom_config.yaml"),
                os.path.join(PROJECT_ROOT, "Airfoil PINN", "geom_config.yaml"),
                "geom_config.yaml"
            ]
            logger.info(f"🔍 Поиск geom_config.yaml по путям:")
            for path in possible_paths:
                exists = "✅" if os.path.exists(path) else "❌"
                logger.info(f"   {exists} {path}")

            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    logger.info(f"✅ Найден: {config_path}")
                    break

        if config_path is None or not os.path.exists(config_path):
            logger.warning("⚠️ geom_config.yaml не найден, используется конфигурация по умолчанию (airfoil)")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                geom_config = yaml.safe_load(f)

            self.config.geom_type = geom_config.get('geom_type', 'airfoil')
            # Не перезаписываем geom_name если он уже задан (например, через меню S2DR)
            if self.config.geom_name is None:
                self.config.geom_name = geom_config.get('geom_name')

            logger.info(f"✅ Конфигурация геометрии загружена: {config_path}")
            logger.info(f"   geom_type: {self.config.geom_type}")
            logger.info(f"   geom_name: {self.config.geom_name or 'default'}")

        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить geom_config.yaml: {e}")
            logger.info("   Используется конфигурация по умолчанию (airfoil)")

    def find_geometry_files(self) -> Dict[str, Optional[str]]:
        """
        Поиск файлов геометрии по geom_name из конфига.

        domain   → Geometry/Data/       ({geom_name}_domain.npy)
        foil     → Geometry/Metadata/   ({geom_name}_foil.npy)
        metadata → Geometry/Metadata/   ({geom_name}_metadata.npy)

        Returns:
            Словарь с путями к файлам {'foil': path, 'domain': path, 'metadata': path}
        """
        result = {'foil': None, 'domain': None, 'metadata': None}
        geom_name = getattr(self.config, 'geom_name', None)
        if not geom_name:
            return result

        data_dir = self.config.geometry_dir          # Geometry/Data/
        meta_dir = DEFAULT_METADATA_DIR              # Geometry/Metadata/

        # domain — только в Data/
        for ext in ('_domain.npy', '_domain.txt'):
            path = os.path.join(data_dir, f"{geom_name}{ext}")
            if os.path.exists(path):
                result['domain'] = path
                print(f"   ✅ Область:    {os.path.basename(path)}")
                break

        if result['domain'] is None:
            return result

        # foil — только в Metadata/
        for ext in ('.npy', '.txt'):
            path = os.path.join(meta_dir, f"{geom_name}_foil{ext}")
            if os.path.exists(path):
                result['foil'] = path
                print(f"   ✅ Профиль:   {os.path.basename(path)}")
                break

        # metadata — только в Metadata/
        meta = os.path.join(meta_dir, f"{geom_name}_metadata.npy")
        if os.path.exists(meta):
            result['metadata'] = meta
            print(f"   ✅ Метаданные: {os.path.basename(meta)}")

        return result

    def load_geometries(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Загрузка файлов геометрии и метаданных.

        Returns:
            Кортеж (domain, foil) — массивы координат области и профиля

        Raises:
            FileNotFoundError: Если файл области не найден
        """
        print("\n📂 Поиск файлов геометрии...")

        files = self.find_geometry_files()
        if files['domain'] is None:
            raise FileNotFoundError(f"Файл области не найден в {self.config.geometry_dir}")

        print(f"\n📂 Загрузка области: {os.path.basename(files['domain'])}")
        domain = self._load_coords(files['domain'])
        # Замыкаем контур области, если необходимо
        if not np.allclose(domain[0], domain[-1], atol=1e-6):
            logger.warning("⚠️ Контур области не замкнут! Замыкаю.")
            domain = np.vstack([domain, domain[0]])
        self.domain_coords = domain

        foil = None
        if files['foil']:
            print(f"📂 Загрузка профиля: {os.path.basename(files['foil'])}")
            foil = self._load_coords(files['foil'])
            self.foil_coords = foil
            self.foil_is_closed = np.allclose(foil[0], foil[-1], atol=1e-6)
            logger.info(f"📐 Профиль замкнутый: {'✅' if self.foil_is_closed else '❌'}")
            print(f"   Профиль: {len(foil)} точек, x_min={np.min(foil[:,0]):.4f}, x_max={np.max(foil[:,0]):.4f}")

        if files['metadata']:
            print(f"📂 Загрузка метаданных: {os.path.basename(files['metadata'])}")
            try:
                self.foil_metadata = np.load(files['metadata'], allow_pickle=True).item()
            except Exception as e:
                print(f"⚠️ Не удалось загрузить метаданные: {e}")

        self.output_base = os.path.join(self.config.output_dir, "collocation_points")
        return domain, foil

    def _load_coords(self, filepath: str) -> np.ndarray:
        """
        Загрузка координат из файла.

        Args:
            filepath: Путь к файлу (.npy, .csv, .txt)

        Returns:
            Массив координат формы (N, 2)
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npy':
            return np.load(filepath)
        elif ext == '.csv':
            return np.loadtxt(filepath, delimiter=',', skiprows=1)
        else:
            return np.loadtxt(filepath, delimiter='\t')

    # ------------------------------------------------------------------
    # Генерация точек на внешнем контуре (прямоугольник)
    # ------------------------------------------------------------------
    def sample_boundary_external(
        self,
        contour: np.ndarray,
        step: float,
        bounds: Dict[str, float]
    ) -> List[CollocationPoint]:
        """
        Точки на внешнем контуре области (равномерно).

        Args:
            contour: Массив координат контура
            step: Шаг дискретизации [м]
            bounds: Границы области

        Returns:
            Список точек на внешней границе
        """
        line = LineString(contour)
        length = line.length
        # Удаляем замыкающую точку для расчёта длины
        if np.allclose(contour[0], contour[-1], atol=1e-6):
            line = LineString(contour[:-1])
            length = line.length

        num_points = int(length / step) + 1
        points = []

        for i in range(num_points):
            distance = i * step
            if distance <= length:
                pt = line.interpolate(distance)
                x, y = pt.x, pt.y
                btype = self._classify_external_boundary(x, y, bounds, step)
                points.append(CollocationPoint(
                    id=self._get_next_id(),
                    x=x, y=y,
                    boundary_type=btype,
                    on_airfoil=False,
                    on_boundary=True
                ))
        type_counts = {}
        for p in points:
            type_counts[p.boundary_type] = type_counts.get(p.boundary_type, 0) + 1
        logger.info(f"✅ Внешняя граница: {len(points)} точек")
        for bt, cnt in type_counts.items():
            logger.info(f"   {bt.value}: {cnt}")
        return points

    def _classify_external_boundary(
        self,
        x: float,
        y: float,
        bounds: Dict[str, float],
        step: float
    ) -> BoundaryType:
        """
        Определяет тип точки на внешнем прямоугольном контуре.

        Args:
            x: X-координата точки [м]
            y: Y-координата точки [м]
            bounds: Границы области (x_min, x_max, y_min, y_max)
            step: Шаг дискретизации [м]

        Returns:
            Тип границы (INLET, OUTLET, WALL, SYMMETRY)
        """
        tol = step * CLASSIFICATION_TOLERANCE_FACTOR
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']

        # Проверяем принадлежность к одной из 4 сторон прямоугольника
        # Тип каждой стороны задаётся пользователем в SamplerConfig
        if abs(x - x_min) < tol:
            return BoundaryType(self.config.left_boundary_type)

        if abs(x - x_max) < tol:
            return BoundaryType(self.config.right_boundary_type)

        if abs(y - y_max) < tol:
            return BoundaryType(self.config.top_boundary_type)

        if abs(y - y_min) < tol:
            return BoundaryType(self.config.bottom_boundary_type)

        # Точка не лежит ни на одной из сторон — внутренняя
        return BoundaryType.INTERIOR

    # ------------------------------------------------------------------
    # Генерация точек на профиле (WALL)
    # ------------------------------------------------------------------
    def sample_boundary_foil(self, foil: np.ndarray, step: float) -> List[CollocationPoint]:
        """
        Точки на профиле (равномерно по длине).

        Args:
            foil: Массив координат профиля
            step: Шаг дискретизации [м]

        Returns:
            Список точек на профиле с типом WALL
        """
        line = LineString(foil)
        length = line.length
        if self.foil_is_closed:
            line = LineString(foil[:-1])
            length = line.length

        num_steps = int(length / step)
        distances = [i * step for i in range(num_steps + 1)]
        if distances[-1] < length:
            distances.append(length)  # гарантируем конечную точку профиля
        points = []
        for distance in distances:
            pt = line.interpolate(distance)
            points.append(CollocationPoint(
                id=self._get_next_id(),
                x=pt.x, y=pt.y,
                boundary_type=BoundaryType.WALL,
                on_airfoil=True,
                on_boundary=True,
            ))
        logger.info(f"✅ Профиль (WALL): {len(points)} точек")
        return points

    # ------------------------------------------------------------------
    # Генерация нижней границы (отдельно, если профиль не доходит до левой границы)
    # ------------------------------------------------------------------
    def sample_bottom_edge(
        self,
        x_min_dom: float,
        x_min_foil: float,
        y_min_dom: float,
        step: float
    ) -> List[CollocationPoint]:
        """
        Генерация точек на нижней границе слева от профиля.

        ВСЕГДА используется равномерное распределение для граничных точек.

        Args:
            x_min_dom: Минимальная X-координата области [м]
            x_min_foil: Минимальная X-координата профиля [м]
            y_min_dom: Минимальная Y-координата области [м]
            step: Шаг дискретизации [м]

        Returns:
            Список точек на нижней границе
        """
        if x_min_foil <= x_min_dom:
            return []
        bottom_length = x_min_foil - x_min_dom
        num_points = int(bottom_length / step) + 1
        # Всегда равномерно, независимо от sampling_method
        xs = np.linspace(x_min_dom, x_min_foil, num_points)

        btype = BoundaryType(self.config.bottom_boundary_type)

        points = []
        for x in xs:
            points.append(CollocationPoint(
                id=self._get_next_id(),
                x=x, y=y_min_dom,
                boundary_type=btype,
                on_airfoil=False,
                on_boundary=True
            ))
        logger.info(f"✅ Нижняя граница ({btype.value}): {len(points)} точек")
        return points

    # ------------------------------------------------------------------
    # Внутренние точки: регулярный метод (векторизованный)
    # ------------------------------------------------------------------
    def sample_interior_regular_sequential(
        self,
        domain: np.ndarray,
        step: float
    ) -> List[CollocationPoint]:
        """
        Генерация регулярных точек с использованием векторизованной проверки Path.

        Args:
            domain: Массив точек области
            step: Шаг сетки [м]

        Returns:
            Список внутренних точек с типом INTERIOR
        """
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds  # или np.min/max domain
        # Генерируем сетку
        x_coords = np.arange(minx, maxx, step)
        y_coords = np.arange(miny, maxy, step)
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.column_stack((X.ravel(), Y.ravel()))
        # Проверяем принадлежность области
        inside = path.contains_points(points)
        valid_points = points[inside]
        # Преобразуем в список CollocationPoint
        interior_points = []
        for x, y in valid_points:
            interior_points.append(CollocationPoint(
                id=self._get_next_id(), x=x, y=y,
                boundary_type=BoundaryType.INTERIOR,
                on_airfoil=False, on_boundary=False
            ))
        logger.info(f"✅ Внутренних точек (регулярный): {len(interior_points)}")
        return interior_points

    def sample_interior_regular_parallel(
        self,
        domain: np.ndarray,
        step: float,
        num_processes: int
    ) -> List[CollocationPoint]:
        """
        Параллельная версия для регулярной сетки (разбиение по X).

        Args:
            domain: Массив точек области
            step: Шаг сетки [м]
            num_processes: Количество процессов

        Returns:
            Список внутренних точек с типом INTERIOR
        """
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        x_step = (maxx - minx) / num_processes
        result_queue = Queue()
        progress_queue = Queue()
        procs = []
        for i in range(num_processes):
            x_min = minx + i * x_step
            x_max = minx + (i + 1) * x_step if i < num_processes - 1 else maxx
            p = Process(target=_regular_worker,
                        args=(domain, step, (x_min, x_max, miny, maxy),
                              i, num_processes, result_queue, progress_queue))
            procs.append(p)
            p.start()
            logger.info(f"🚀 Запущен процесс {i + 1}/{num_processes}")

        stop_monitor = threading.Event()

        def monitor():
            progress_data = defaultdict(lambda: (0, 0))
            last_print = time.time()
            while not stop_monitor.is_set() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.5)
                    proc_id, current, total = msg
                    progress_data[proc_id] = (current, total)
                except queue.Empty:
                    pass  # нет новых сообщений, продолжаем ждать
                except Exception as e:
                    logger.error(f"Неожиданная ошибка в мониторе: {e}")
                if time.time() - last_print > 2:
                    lines = []
                    for pid in sorted(progress_data.keys()):
                        cur, tot = progress_data[pid]
                        percent = 100.0 * cur / tot if tot > 0 else 0
                        lines.append(f"  Процесс {pid + 1}: {cur}/{tot} ({percent:.1f}%)")
                    if lines:
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.write("Текущий прогресс (регулярный):\n")
                        sys.stdout.write("\n".join(lines) + "\n")
                        sys.stdout.flush()
                    last_print = time.time()
            print("✅ Регулярные процессы завершены.")

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        all_coords = []
        for _ in range(num_processes):
            try:
                chunk = result_queue.get(timeout=600)
                all_coords.extend(chunk)
            except queue.Empty:
                logger.warning("⚠️ Таймаут получения данных от одного из процессов")
            except Exception as e:
                logger.error(f"Ошибка при получении данных: {e}")
        for p in procs:
            p.join(timeout=5)
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        points = [CollocationPoint(self._get_next_id(), x, y,
                                   BoundaryType.INTERIOR, False, False)
                  for x, y in all_coords]
        logger.info(f"✅ Внутренних точек (регулярный параллельный): {len(points)}")
        return points

    # ------------------------------------------------------------------
    # Внутренние точки: нерегулярный метод (равномерный случайный)
    # ------------------------------------------------------------------
    def sample_interior_irregular(
        self,
        domain: np.ndarray,
        num_points: int
    ) -> List[CollocationPoint]:
        """
        Генерация внутренних точек методом нерегулярной выборки.

        Args:
            domain: Массив точек области
            num_points: Целевое количество точек

        Returns:
            Список внутренних точек
        """
        if self.config.use_multiprocessing and self.config.num_processes > 1:
            return self._sample_interior_irregular_parallel(domain, num_points)
        else:
            return self._sample_interior_irregular_sequential(domain, num_points)

    def _sample_interior_irregular_sequential(
        self,
        domain: np.ndarray,
        num_points: int
    ) -> List[CollocationPoint]:
        """
        Генерация внутренних точек методом нерегулярной выборки (последовательно).

        Args:
            domain: Массив точек, определяющих область
            num_points: Целевое количество точек

        Returns:
            Список внутренних точек
        """
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        points = []
        attempts = 0
        max_attempts = num_points * MAX_ATTEMPTS_MULTIPLIER
        
        if TQDM_AVAILABLE and self.config.show_progress:
            pbar = tqdm(total=num_points, desc="Внутренние (нерегулярный)")
        else:
            pbar = None
            last_log = time.time()
            
        while len(points) < num_points and attempts < max_attempts:
            # Генерируем батч
            batch_size = min(DEFAULT_BATCH_SIZE, max_attempts - attempts)
            xs = np.random.uniform(minx, maxx, batch_size)
            ys = np.random.uniform(miny, maxy, batch_size)
            candidates = np.column_stack((xs, ys))
            inside = path.contains_points(candidates)
            valid = candidates[inside]
            
            for x, y in valid:
                if len(points) < num_points:
                    points.append(CollocationPoint(
                        id=self._get_next_id(), x=x, y=y,
                        boundary_type=BoundaryType.INTERIOR,
                        on_airfoil=False, on_boundary=False
                    ))
                    if pbar:
                        pbar.update(1)
                else:
                    break
            attempts += batch_size
            
            if not pbar and self.config.show_progress and time.time() - last_log > QUEUE_TIMEOUT:
                logger.info(f"⏳ Прогресс: {len(points)}/{num_points}")
                last_log = time.time()
                
        if pbar:
            pbar.close()
        if len(points) < num_points:
            logger.warning(f"⚠️ Сгенерировано только {len(points)} из {num_points}")
        return points

    def _sample_interior_irregular_parallel(self, domain: np.ndarray,
                                            num_points: int) -> List[CollocationPoint]:
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        num_proc = self.config.num_processes
        target_per_proc = num_points // num_proc
        remainder = num_points % num_proc
        x_edges = np.linspace(minx, maxx, num_proc + 1)

        result_queue = Queue()
        progress_queue = Queue()
        procs = []
        for i in range(num_proc):
            x_min = x_edges[i]
            x_max = x_edges[i + 1]
            target = target_per_proc + (1 if i < remainder else 0)
            p = Process(target=_irregular_worker,
                        args=(domain, (x_min, x_max, miny, maxy),
                              target, i, result_queue, progress_queue))
            procs.append(p)
            p.start()
            logger.info(f"🚀 Запущен процесс {i + 1}/{num_proc}, цель: {target} точек")

        stop_monitor = threading.Event()

        def monitor():
            progress_data = defaultdict(lambda: (0, 0))
            last_print = time.time()
            while not stop_monitor.is_set() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.5)
                    proc_id, current, target = msg
                    progress_data[proc_id] = (current, target)
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Неожиданная ошибка в мониторе: {e}")
                if time.time() - last_print > 2:
                    lines = []
                    for pid in sorted(progress_data.keys()):
                        cur, tgt = progress_data[pid]
                        percent = 100.0 * cur / tgt if tgt > 0 else 0
                        lines.append(f"  Процесс {pid + 1}: {cur}/{tgt} ({percent:.1f}%)")
                    if lines:
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.write("Текущий прогресс (нерегулярный):\n")
                        sys.stdout.write("\n".join(lines) + "\n")
                        sys.stdout.flush()
                    last_print = time.time()
            print("✅ Нерегулярные процессы завершены.")

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        all_points = []
        for _ in range(num_proc):
            try:
                chunk = result_queue.get(timeout=600)
                all_points.extend(chunk)
            except queue.Empty:
                logger.warning("⚠️ Таймаут получения от процесса")
            except Exception as e:
                logger.error(f"Ошибка при получении данных: {e}")

        for p in procs:
            p.join(timeout=5)
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        points = [CollocationPoint(self._get_next_id(), x, y,
                                   BoundaryType.INTERIOR, False, False)
                  for x, y in all_points]
        logger.info(f"✅ Внутренних точек (нерегулярный параллельный): {len(points)}")
        return points

    # ------------------------------------------------------------------
    # Внутренние точки: адаптивные методы (сгущение)
    # ------------------------------------------------------------------
    def sample_interior_adaptive(
        self,
        domain: np.ndarray,
        num_points: int,
        alpha: float,
        epsilon_factor: float,
        mode: str
    ) -> List[CollocationPoint]:
        """
        Генерация внутренних точек адаптивным методом (сгущение).

        Args:
            domain: Массив точек области
            num_points: Целевое количество точек
            alpha: Степень сгущения (чем больше, тем сильнее концентрация)
            epsilon_factor: Масштаб сгущения (доля от характерного размера)
            mode: Режим сгущения ('foil' — к профилю, 'all' — ко всем границам)

        Returns:
            Список внутренних точек

        Raises:
            ValueError: Если режим адаптивного метода неизвестен
        """
        # Определяем, к каким точкам будем сгущаться
        if mode == 'foil':
            if self.foil_coords is None:
                logger.warning("⚠️ Для сгущения к профилю нужен профиль. Использую нерегулярный.")
                return self.sample_interior_irregular(domain, num_points)
            # Используем только точки профиля (WALL)
            wall_mask = [p.boundary_type == BoundaryType.WALL for p in self.points_boundary]
            boundary_points = np.array([(p.x, p.y) for p, m in zip(self.points_boundary, wall_mask) if m])
            if len(boundary_points) == 0:
                logger.warning("⚠️ Нет точек WALL для сгущения. Использую нерегулярный.")
                return self.sample_interior_irregular(domain, num_points)
        elif mode == 'all':
            # Используем все граничные точки
            boundary_points = np.array([(p.x, p.y) for p in self.points_boundary])
            if len(boundary_points) == 0:
                logger.warning("⚠️ Нет граничных точек для сгущения. Использую нерегулярный.")
                return self.sample_interior_irregular(domain, num_points)
        else:
            raise ValueError(f"Неизвестный режим адаптивного метода: {mode}")

        if self.config.use_multiprocessing and self.config.num_processes > 1:
            return self._sample_interior_adaptive_parallel(domain, boundary_points, num_points,
                                                            alpha, epsilon_factor, mode)
        else:
            return self._sample_interior_adaptive_sequential(domain, boundary_points, num_points,
                                                              alpha, epsilon_factor, mode)

    def _sample_interior_adaptive_sequential(self, domain: np.ndarray, boundary_points: np.ndarray,
                                             num_points: int,
                                             alpha: float,
                                             epsilon_factor: float,
                                             mode: str) -> List[CollocationPoint]:
        """
        Генерация внутренних точек с адаптивным сгущением (последовательно).
        
        Args:
            domain: Массив точек, определяющих область
            boundary_points: Граничные точки для сгущения
            num_points: Целевое количество точек
            alpha: Степень сгущения
            epsilon_factor: Масштаб сгущения
            mode: Режим ('adaptive_foil' или 'adaptive_all')
            
        Returns:
            Список внутренних точек
        """
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        char_length = (maxx - minx + maxy - miny) / 2
        eps = char_length * epsilon_factor

        # Строим дерево по переданным граничным точкам
        tree = cKDTree(boundary_points)

        points = []
        batch_size = DEFAULT_BATCH_SIZE
        max_attempts = num_points * MAX_ATTEMPTS_MULTIPLIER
        total_attempts = 0

        if TQDM_AVAILABLE and self.config.show_progress:
            pbar = tqdm(total=num_points, desc=f"Адаптивный ({mode})")
        else:
            pbar = None
            last_log = time.time()

        while len(points) < num_points and total_attempts < max_attempts:
            xs = np.random.uniform(minx, maxx, batch_size)
            ys = np.random.uniform(miny, maxy, batch_size)
            candidates = np.column_stack((xs, ys))

            inside = path.contains_points(candidates)
            valid = candidates[inside]
            if len(valid) == 0:
                total_attempts += batch_size
                continue

            dists, _ = tree.query(valid)
            probs = (eps / (dists + eps)) ** alpha
            accept = np.random.random(len(valid)) < probs
            accepted = valid[accept]

            for x, y in accepted:
                if len(points) < num_points:
                    points.append(CollocationPoint(
                        id=self._get_next_id(), x=x, y=y,
                        boundary_type=BoundaryType.INTERIOR,
                        on_airfoil=False, on_boundary=False
                    ))
                    if pbar:
                        pbar.update(1)
                else:
                    break
            total_attempts += batch_size

            if not pbar and self.config.show_progress and time.time() - last_log > 2:
                logger.info(f"⏳ Прогресс: {len(points)}/{num_points}")
                last_log = time.time()

        if pbar:
            pbar.close()

        if len(points) < num_points:
            logger.warning(f"⚠️ Адаптивный метод ({mode}): только {len(points)} из {num_points}")
        return points

    def _sample_interior_adaptive_parallel(self, domain: np.ndarray, boundary_points: np.ndarray,
                                           num_points: int,
                                           alpha: float,
                                           epsilon_factor: float,
                                           mode: str) -> List[CollocationPoint]:
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        num_proc = self.config.num_processes
        target_per_proc = num_points // num_proc
        remainder = num_points % num_proc
        x_edges = np.linspace(minx, maxx, num_proc + 1)

        result_queue = Queue()
        progress_queue = Queue()
        procs = []
        for i in range(num_proc):
            x_min = x_edges[i]
            x_max = x_edges[i + 1]
            target = target_per_proc + (1 if i < remainder else 0)
            p = Process(target=_adaptive_worker,
                        args=(domain, boundary_points, (x_min, x_max, miny, maxy),
                              target, alpha, epsilon_factor, i, result_queue, progress_queue))
            procs.append(p)
            p.start()
            logger.info(f"🚀 Запущен процесс {i + 1}/{num_proc}, цель: {target} точек")

        stop_monitor = threading.Event()

        def monitor():
            progress_data = defaultdict(lambda: (0, 0))
            last_print = time.time()
            while not stop_monitor.is_set() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.5)
                    proc_id, current, target = msg
                    progress_data[proc_id] = (current, target)
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Неожиданная ошибка в мониторе: {e}")
                if time.time() - last_print > 2:
                    lines = []
                    for pid in sorted(progress_data.keys()):
                        cur, tgt = progress_data[pid]
                        percent = 100.0 * cur / tgt if tgt > 0 else 0
                        lines.append(f"  Процесс {pid + 1}: {cur}/{tgt} ({percent:.1f}%)")
                    if lines:
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.write("Текущий прогресс (адаптивный параллельный):\n")
                        sys.stdout.write("\n".join(lines) + "\n")
                        sys.stdout.flush()
                    last_print = time.time()
            print("✅ Адаптивные процессы завершены.")

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        all_points = []
        for _ in range(num_proc):
            try:
                chunk = result_queue.get(timeout=1200)
                all_points.extend(chunk)
            except queue.Empty:
                logger.warning("⚠️ Таймаут получения от адаптивного процесса")
            except Exception as e:
                logger.error(f"Ошибка при получении данных: {e}")

        for p in procs:
            p.join(timeout=5)
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        points = [CollocationPoint(self._get_next_id(), x, y,
                                   BoundaryType.INTERIOR, False, False)
                  for x, y in all_points]
        logger.info(f"✅ Внутренних точек (адаптивный параллельный): {len(points)}")
        return points

    # ------------------------------------------------------------------
    # Основной метод генерации всех точек
    # ------------------------------------------------------------------
    def generate_all_points(
        self,
        domain: np.ndarray,
        foil: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Основной метод генерации всех точек коллокации.

        Генерирует точки на внешнем контуре, профиле (если есть) и внутренние точки.

        Args:
            domain: Массив координат расчётной области
            foil: Массив координат профиля (или None)

        Returns:
            Словарь с результатами генерации
        """
        logger.info("=" * 60)
        logger.info("ГЕНЕРАЦИЯ ТОЧЕК КОЛЛОКАЦИИ")
        logger.info("=" * 60)

        step_m = self.config.step_size
        if self.config.step_units == 'mm':
            step_m /= 1000.0
        logger.info(f"📏 Шаг сетки: {self.config.step_size} {self.config.step_units} = {step_m:.6f} м")
        logger.info(f"🔧 Метод внутренних точек: {self.config.sampling_method}")
        if 'adaptive' in self.config.sampling_method:
            logger.info(f"   alpha = {self.config.adaptive_alpha}")
            logger.info(f"   epsilon_factor = {self.config.adaptive_epsilon_factor}")
        logger.info(f"🔲 Типы границ: ←{self.config.left_boundary_type} →{self.config.right_boundary_type} "
                    f"↑{self.config.top_boundary_type} ↓{self.config.bottom_boundary_type}")

        x_min_dom, y_min_dom = np.min(domain, axis=0)
        x_max_dom, y_max_dom = np.max(domain, axis=0)
        bounds = {'x_min': x_min_dom, 'x_max': x_max_dom,
                  'y_min': y_min_dom, 'y_max': y_max_dom}
        logger.info(f"📐 Область: x=[{x_min_dom:.4f}, {x_max_dom:.4f}], y=[{y_min_dom:.4f}, {y_max_dom:.4f}]")

        poly_domain = Polygon(domain)
        if not poly_domain.is_valid:
            poly_domain = poly_domain.buffer(0)
        foil_area = 0.0
        if foil is not None and len(foil) >= 3:
            poly_foil = Polygon(foil)
            if not poly_foil.is_valid:
                poly_foil = poly_foil.buffer(0)
            if poly_foil.is_valid:
                foil_area = poly_foil.area
        effective_area = poly_domain.area - foil_area
        num_interior = int(effective_area / step_m ** 2 * self.config.interior_point_density)

        # 1. Точки на внешнем контуре
        logger.info("\n📌 Внешний контур:")
        self.points_boundary = self.sample_boundary_external(domain, step_m, bounds)

        # 2. Точки на профиле (если есть)
        if foil is not None:
            logger.info("\n📌 Профиль (WALL):")
            foil_points = self.sample_boundary_foil(foil, step_m)
            self.points_boundary.extend(foil_points)

        # 3. Нижняя граница слева от профиля
        if foil is not None:
            x_min_foil = np.min(foil[:, 0])
            if x_min_foil > x_min_dom + 1e-12:
                logger.info("\n📌 Нижняя граница (слева от профиля):")
                bottom_points = self.sample_bottom_edge(x_min_dom, x_min_foil, y_min_dom, step_m)
                # Исключить точки, уже покрытые внешним контуром (совпадение < step/2).
                # Без этой фильтрации два набора точек на одной прямой y=y_min создают
                # взаимно-коллинеарные CDT-рёбра → нарушение консервативности FVM.
                if bottom_points and self.points_boundary:
                    _exist_xy = np.array([[p.x, p.y] for p in self.points_boundary])
                    _tree = cKDTree(_exist_xy)
                    filtered = [bp for bp in bottom_points
                                if _tree.query([bp.x, bp.y])[0] > step_m * 0.5]
                    n_dup = len(bottom_points) - len(filtered)
                    if n_dup:
                        logger.info("   Исключено %d дублирующих точек (уже покрыты внешним контуром)", n_dup)
                    bottom_points = filtered
                self.points_boundary.extend(bottom_points)

        # 4. Внутренние точки
        logger.info("\n📌 Внутренние точки:")
        if self.config.sampling_method == 'regular':
            if self.config.use_multiprocessing and self.config.num_processes > 1:
                interior = self.sample_interior_regular_parallel(domain, step_m, self.config.num_processes)
            else:
                interior = self.sample_interior_regular_sequential(domain, step_m)
        elif self.config.sampling_method == 'irregular':
            interior = self.sample_interior_irregular(domain, num_interior)
        elif self.config.sampling_method == 'adaptive_foil':
            interior = self.sample_interior_adaptive(domain, num_interior,
                                                     self.config.adaptive_alpha,
                                                     self.config.adaptive_epsilon_factor,
                                                     'foil')
        elif self.config.sampling_method == 'adaptive_all':
            interior = self.sample_interior_adaptive(domain, num_interior,
                                                     self.config.adaptive_alpha,
                                                     self.config.adaptive_epsilon_factor,
                                                     'all')
        else:
            raise ValueError(f"Неизвестный метод: {self.config.sampling_method}")

        self.points_interior = interior
        self.points = self.points_boundary + self.points_interior

        type_counts = {}
        for p in self.points:
            type_counts[p.boundary_type] = type_counts.get(p.boundary_type, 0) + 1

        self.metadata = {
            'num_points': len(self.points),
            'num_interior': len(self.points_interior),
            'num_boundary': len(self.points_boundary),
            'type_counts': {k.value: v for k, v in type_counts.items()},
            'step_size': self.config.step_size,
            'step_units': self.config.step_units,
            'step_m': step_m,
            'bounds': bounds,
            'foil_closed': self.foil_is_closed,
            'left_boundary_type': self.config.left_boundary_type,
            'right_boundary_type': self.config.right_boundary_type,
            'top_boundary_type': self.config.top_boundary_type,
            'bottom_boundary_type': self.config.bottom_boundary_type,
            'sampling_method': self.config.sampling_method,
            'adaptive_alpha': self.config.adaptive_alpha,
            'adaptive_epsilon_factor': self.config.adaptive_epsilon_factor
        }

        logger.info("\n" + "=" * 60)
        logger.info("ИТОГО:")
        logger.info(f"  Внутри области: {self.metadata['num_interior']}")
        logger.info(f"  На границе: {self.metadata['num_boundary']}")
        logger.info(f"  ВСЕГО: {self.metadata['num_points']}")
        logger.info("\n  По типам граничных условий:")
        for bt, cnt in type_counts.items():
            if bt != BoundaryType.INTERIOR:
                logger.info(f"    {bt.value}: {cnt}")
        logger.info("=" * 60)

        return {'points': self.points, 'interior': self.points_interior,
                'boundary': self.points_boundary, 'metadata': self.metadata}

    # ------------------------------------------------------------------
    # Визуализация и сохранение
    # ------------------------------------------------------------------
    def plot_points(
        self,
        domain: np.ndarray,
        foil: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Визуализация точек коллокации.

        Args:
            domain: Массив координат области
            foil: Массив координат профиля (или None)
            save_path: Путь для сохранения изображения (опционально)
        """
        plt.figure(figsize=(14, 10))
        plt.plot(domain[:, 0], domain[:, 1], 'k-', lw=1.5, alpha=0.5, label='Граница области')
        if foil is not None:
            plt.plot(foil[:, 0], foil[:, 1], 'r-', lw=2, label='Профиль')

        colors = {
            BoundaryType.INLET: 'blue',
            BoundaryType.OUTLET: 'cyan',
            BoundaryType.WALL: 'red',
            BoundaryType.SYMMETRY: 'purple',
            BoundaryType.AXIS_SYMMETRY: 'magenta',
            BoundaryType.INTERIOR: 'green'
        }
        sizes = {
            BoundaryType.INLET: 10,
            BoundaryType.OUTLET: 10,
            BoundaryType.WALL: 15,
            BoundaryType.SYMMETRY: 10,
            BoundaryType.AXIS_SYMMETRY: 10,
            BoundaryType.INTERIOR: 1
        }
        points_by_type = {}
        for p in self.points:
            points_by_type.setdefault(p.boundary_type, {'x': [], 'y': []})
            points_by_type[p.boundary_type]['x'].append(p.x)
            points_by_type[p.boundary_type]['y'].append(p.y)

        for bt, coords in points_by_type.items():
            plt.scatter(coords['x'], coords['y'],
                        c=colors[bt], s=sizes[bt],
                        alpha=0.6 if bt == BoundaryType.INTERIOR else 0.8,
                        label=f'{bt.value} ({len(coords["x"])})',
                        zorder=3 if bt == BoundaryType.INTERIOR else 4)

        plt.legend(loc='upper right', fontsize=10, ncol=2)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        step_info = f"шаг={self.config.step_size}{self.config.step_units}, метод={self.config.sampling_method}"
        if 'adaptive' in self.config.sampling_method:
            step_info += f", α={self.config.adaptive_alpha}, ε_factor={self.config.adaptive_epsilon_factor}"
        plt.title(f"Точки коллокации (всего: {self.metadata['num_points']}, {step_info})", fontsize=14)
        plt.xlabel('X, м')
        plt.ylabel('Y, м')
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"✅ График сохранён: {save_path}")
        if self.config.plot_results:
            plt.show()
        else:
            plt.close()

    def save_points(self) -> None:
        """
        Сохранение точек коллокации и метаданных.

        Сохраняет все точки, внутренние точки, граничные точки и метаданные
        в форматах npy или csv в зависимости от конфигурации.
        """
        if self.output_base is None:
            raise RuntimeError(
                "output_base не задан — сначала вызовите load_geometries()"
            )
        base = self.output_base
        if self.config.save_format == 'npy':
            dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'),
                     ('boundary_type', 'U20'), ('on_airfoil', 'bool'),
                     ('on_boundary', 'bool'), ('segment_id', 'i4')]
            arr = np.array([(p.id, p.x, p.y, p.boundary_type.value,
                             p.on_airfoil, p.on_boundary, p.segment_id)
                            for p in self.points], dtype=dtype)
            np.save(f"{base}.npy", arr)
            interior_coords = [(p.x, p.y) for p in self.points_interior]
            np.save(f"{base}_interior.npy",
                    np.array(interior_coords) if interior_coords else np.empty((0, 2)))
            boundary_coords = [(p.x, p.y) for p in self.points_boundary]
            np.save(f"{base}_boundary.npy",
                    np.array(boundary_coords) if boundary_coords else np.empty((0, 2)))
            logger.info(f"✅ Сохранено: {base}.npy")
        elif self.config.save_format == 'csv':
            with open(f"{base}.csv", 'w') as f:
                f.write("id,x,y,boundary_type,on_airfoil,on_boundary,segment_id\n")
                for p in self.points:
                    f.write(f"{p.id},{p.x},{p.y},{p.boundary_type.value},"
                            f"{p.on_airfoil},{p.on_boundary},{p.segment_id}\n")
            logger.info(f"✅ Сохранено: {base}.csv")
        np.save(f"{base}_metadata.npy", self.metadata)
        logger.info(f"✅ Метаданные сохранены")

    def get_points_by_type(self, btype: BoundaryType) -> List[CollocationPoint]:
        """
        Возвращает точки указанного типа.

        Args:
            btype: Тип границы для фильтрации

        Returns:
            Список точек заданного типа
        """
        return [p for p in self.points if p.boundary_type == btype]


def _select_geometry_data() -> Optional[str]:
    """
    Меню выбора расчётной области из папки Geometry/Data/.

    Сканирует директорию на файлы *_domain.npy / *_domain.txt и предлагает
    пользователю выбрать нужный по номеру.

    Returns:
        Базовое имя геометрии (например 'plate') или None при отмене.
    """
    geom_dir = DEFAULT_GEOMETRY_DIR

    candidates: List[Tuple[str, str]] = []
    if os.path.isdir(geom_dir):
        for fname in sorted(os.listdir(geom_dir)):
            if fname.endswith('_domain.npy') or fname.endswith('_domain.txt'):
                base = fname.replace('_domain.npy', '').replace('_domain.txt', '')
                candidates.append((base, os.path.join(geom_dir, fname)))

    print("\n" + "=" * 60)
    print("  ВЫБОР РАСЧЁТНОЙ ОБЛАСТИ".center(60))
    print("=" * 60)

    if not candidates:
        print(f"\n  Файлы геометрии (*_domain.npy) не найдены.")
        print(f"  Папка: {geom_dir}")
        print(f"\n  Сначала запустите модуль G2D.")
        print("=" * 60)
        return None

    print(f"  Папка: Geometry/Data/    Найдено: {len(candidates)}")
    print("-" * 60)
    for i, (base, path) in enumerate(candidates, 1):
        size_kb = os.path.getsize(path) / 1024
        # Проверяем наличие профиля в Metadata/
        has_foil = any(
            os.path.exists(os.path.join(DEFAULT_METADATA_DIR, f"{base}_foil{ext}"))
            for ext in ('.npy', '.txt')
        )
        tags = "  [+ профиль]" if has_foil else ""
        print(f"  [{i:2d}]  {base:<36}  {size_kb:6.1f} кБ{tags}")
    print("-" * 60)
    print("  [ 0]  Отмена")
    print("=" * 60)

    while True:
        try:
            idx = int(input("\n  Номер геометрии: ").strip())
            if idx == 0:
                return None
            if 1 <= idx <= len(candidates):
                base, domain_path = candidates[idx - 1]
                print(f"\n  Выбрана геометрия: {base}")
                _show_geometry_preview(base, domain_path, geom_dir)
                return base
            print(f"  Введите число от 0 до {len(candidates)}")
        except ValueError:
            print("  Ошибка: введите число")
        except KeyboardInterrupt:
            print()
            return None


def _show_geometry_preview(base: str, domain_path: str, geom_dir: str) -> None:
    """Загружает и показывает основные параметры выбранной геометрии."""
    print("\n" + "-" * 60)
    print("  ПАРАМЕТРЫ ГЕОМЕТРИИ".center(60))
    print("-" * 60)
    try:
        domain = np.load(domain_path) if domain_path.endswith('.npy') else np.loadtxt(domain_path)
        x_min, y_min = domain[:, 0].min(), domain[:, 1].min()
        x_max, y_max = domain[:, 0].max(), domain[:, 1].max()
        print(f"  Область : {len(domain)} точек")
        print(f"  X: [{x_min:.4f}, {x_max:.4f}] м  →  {x_max - x_min:.4f} м")
        print(f"  Y: [{y_min:.4f}, {y_max:.4f}] м  →  {y_max - y_min:.4f} м")
    except Exception as e:
        print(f"  Не удалось прочитать область: {e}")
        print("-" * 60)
        return

    # Профиль — в Geometry/Metadata/
    for ext in ('.npy', '.txt'):
        foil_path = os.path.join(DEFAULT_METADATA_DIR, f"{base}_foil{ext}")
        if os.path.exists(foil_path):
            try:
                foil = np.load(foil_path) if ext == '.npy' else np.loadtxt(foil_path)
                chord = foil[:, 0].max() - foil[:, 0].min()
                is_closed = np.allclose(foil[0], foil[-1], atol=1e-6)
                print(f"  Профиль : {len(foil)} точек  |  хорда {chord:.4f} м"
                      f"  |  {'замкнутый' if is_closed else 'незамкнутый'}")
            except Exception:
                pass
            break

    print("-" * 60)


def S2DR() -> None:
    """
    Главная функция модуля дискретизации S2DR (интерактивный режим).

    Запускает интерактивный сеанс для генерации точек коллокации
    с классификацией граничных условий.

    Example:
        >>> if __name__ == "__main__":
        ...     S2DR()
    """
    print('\033c' + "=" * 60 + "\nМОДУЛЬ ДИСКРЕТИЗАЦИИ S2DR".center(60) + "\n" + "=" * 60)
    try:
        if os.name == 'nt':
            freeze_support()

        # Выбор расчётной области (первым делом, до всех других вопросов)
        selected_geom = _select_geometry_data()
        if selected_geom is None:
            return

        num_proc = int(input("\nКоличество ядер процессора (1-16): "))
        if not 1 <= num_proc <= 16:
            raise ValueError("Некорректное число ядер")

        print("\nЕдиницы шага:\n   1 - метры\n   2 - миллиметры")
        units_choice = input("Выберите (1/2): ").strip()
        step_units = 'm' if units_choice == '1' else 'mm'
        step = float(input(f"Шаг сетки ({step_units}): "))
        if step <= 0:
            raise ValueError("Шаг должен быть >0")

        _BOUNDARY_MENU = (
            "   1 - inlet         (вход потока)\n"
            "   2 - outlet        (выход потока)\n"
            "   3 - symmetry      (плоская симметрия)\n"
            "   4 - axis_symmetry (осевая симметрия)\n"
            "   5 - wall          (твёрдая стенка)"
        )
        _BOUNDARY_MAP = {'1': 'inlet', '2': 'outlet', '3': 'symmetry',
                         '4': 'axis_symmetry', '5': 'wall'}

        print("\nТип ЛЕВОЙ границы (x = x_min):")
        print(_BOUNDARY_MENU)
        left_boundary_type = _BOUNDARY_MAP.get(input("Выберите (1-5): ").strip(), 'inlet')

        print("\nТип ПРАВОЙ границы (x = x_max):")
        print(_BOUNDARY_MENU)
        right_boundary_type = _BOUNDARY_MAP.get(input("Выберите (1-5): ").strip(), 'outlet')

        print("\nТип ВЕРХНЕЙ границы (y = y_max):")
        print(_BOUNDARY_MENU)
        top_boundary_type = _BOUNDARY_MAP.get(input("Выберите (1-5): ").strip(), 'outlet')

        print("\nТип НИЖНЕЙ границы (y = y_min):")
        print(_BOUNDARY_MENU)
        bottom_boundary_type = _BOUNDARY_MAP.get(input("Выберите (1-5): ").strip(), 'symmetry')

        print("\nМетод генерации внутренних точек:")
        print("   1 - регулярный (равномерная сетка)")
        print("   2 - нерегулярный (равномерный случайный)")
        print("   3 - адаптивный (сгущение к профилю)")
        print("   4 - адаптивный (сгущение ко всем границам)")
        method_choice = input("Выберите (1/2/3/4): ").strip()
        method_map = {
            '1': 'regular',
            '2': 'irregular',
            '3': 'adaptive_foil',
            '4': 'adaptive_all'
        }
        sampling_method = method_map.get(method_choice, 'regular')

        adaptive_alpha = 1.0
        adaptive_epsilon_factor = 0.01
        if sampling_method in ('adaptive_foil', 'adaptive_all'):
            print("\nДля адаптивных методов используются два параметра:")
            print("  - alpha (степень сгущения): чем больше, тем сильнее концентрация у границ.")
            print("    Рекомендуемый диапазон: 0.5–2.0.")
            print("  - epsilon_factor (масштаб сгущения): доля от характерного размера области,")
            print("    определяющая расстояние, на котором сгущение заметно. Например, 0.01 даёт")
            print("    масштаб около 1 см для области размером 1 м. Меньшие значения делают")
            print("    сгущение более резким у самой границы.")
            alpha_input = input("alpha (по умолчанию 1.0): ").strip()
            if alpha_input:
                adaptive_alpha = float(alpha_input)
            eps_input = input("epsilon_factor (по умолчанию 0.01): ").strip()
            if eps_input:
                adaptive_epsilon_factor = float(eps_input)

        use_mp_input = input("\nИспользовать многопроцессорность? (Y/n): ").strip().lower()
        use_mp = use_mp_input == '' or use_mp_input in ('y', 'yes')
        plot_input = input("Показать график? (Y/n): ").strip().lower()
        plot_res = plot_input == '' or plot_input in ('y', 'yes')

        config = SamplerConfig(
            num_processes=num_proc,
            step_size=step,
            step_units=step_units,
            use_multiprocessing=use_mp and num_proc > 1,
            plot_results=plot_res,
            save_format='npy',
            left_boundary_type=left_boundary_type,
            right_boundary_type=right_boundary_type,
            top_boundary_type=top_boundary_type,
            bottom_boundary_type=bottom_boundary_type,
            sampling_method=sampling_method,
            adaptive_alpha=adaptive_alpha,
            adaptive_epsilon_factor=adaptive_epsilon_factor,
            geom_name=selected_geom,
        )

        sampler = CollocationPoints(config)

        print("\n📂 Загрузка геометрии...")
        try:
            domain, foil = sampler.load_geometries()
            print(f"✅ Область: {len(domain)} точек")
            if foil is not None:
                print(f"✅ Профиль: {len(foil)} точек")
        except FileNotFoundError as e:
            print(f"\n❌ {e}\n💡 Сначала запустите G2D")
            input("\nНажмите Enter...")
            return

        print("\n🔧 Генерация точек коллокации...")
        sampler.generate_all_points(domain, foil)

        print("\n📊 Построение графика точек коллокации...")
        plot_path = os.path.join(config.output_dir, "collocation_points.png")
        sampler.plot_points(domain, foil, save_path=plot_path)
        print(f"✅ График сохранён: {plot_path}")

        print("\n💾 Сохранение...")
        sampler.save_points()

        mesh_input = input("\nПостроить FVM-сетку (mesh2d)? (y/N): ").strip().lower()
        if mesh_input in ('y', 'yes'):
            print("\n🔧 Построение FVM-сетки (CDT)...")
            try:
                from mesh2d import Mesh2D
            except ImportError:
                print("\n❌ Модуль mesh2d.py не найден — убедитесь, что файл находится в директории проекта")
                logger.error("ImportError: mesh2d не найден")
            else:
                try:
                    mesh = Mesh2D.from_sampler(sampler)
                    mesh.build()
                    mesh_path = os.path.join(config.output_dir, "mesh2d.npz")
                    mesh.save(mesh_path)
                    s = mesh.summary()
                    print(f"✅ Сетка сохранена: {mesh_path}")
                    print(f"   Узлов: {s['num_nodes']}  |  Треугольников: {s['num_triangles']}")
                    print(f"   Внутренних граней: {s['num_internal_faces']}  |  Граничных граней: {s['num_boundary_faces']}")
                    print(f"   CV min/max/mean: {s['cv_volume_min']:.3e} / {s['cv_volume_max']:.3e} / {s['cv_volume_mean']:.3e} м²")

                    # ── Диагностика объёмов CV ───────────────────────────────
                    vols = mesh.cv_volumes
                    min_vol = float(vols.min())
                    min_vol_idx = int(vols.argmin())
                    print(f"   Минимальный объём CV: {min_vol:.3e} (узел {min_vol_idx})")

                    print("\n📊 Построение графика сетки...")
                    mesh_plot_path = os.path.join(config.output_dir, "mesh2d.png")
                    mesh.plot(save_path=mesh_plot_path, dpi=config.plot_dpi,
                              show=config.plot_results)
                    print(f"✅ График сохранён: {mesh_plot_path}")

                    print("\n📊 Построение карты объёмов CV...")
                    cv_map_path = os.path.join(config.output_dir, "mesh2d_cv_map.png")
                    mesh.plot_cv_volume_map(
                        save_path=cv_map_path,
                        dpi=config.plot_dpi,
                        show=config.plot_results,
                    )
                    print(f"✅ Карта CV сохранена: {cv_map_path}")

                    print("\n📊 Построение гистограммы распределения объёмов CV...")
                    cv_hist_path = os.path.join(config.output_dir, "mesh2d_cv_hist.png")
                    mesh.plot_cv_volume_hist(
                        bins=100,
                        save_path=cv_hist_path,
                        dpi=config.plot_dpi,
                        show=config.plot_results,
                    )
                    print(f"✅ Гистограмма CV сохранена: {cv_hist_path}")
                except Exception as e:
                    print(f"\n❌ Ошибка построения сетки: {e}")
                    logger.error("Ошибка mesh2d.build()", exc_info=True)

        print("\n" + "=" * 60 + "\n✅ МОДУЛЬ S2DR УСПЕШНО ВЫПОЛНЕН\n" + "=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        logger.error("Ошибка", exc_info=True)

    input("\nНажмите Enter для возврата...")


if __name__ == "__main__":
    S2DR()