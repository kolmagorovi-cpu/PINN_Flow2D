#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль дискретизации S2DR (Sampler 2D)
Генерация точек коллокации для PINNs с классификацией граничных условий

Область задаётся внешним прямоугольником (domain) и профилем (foil).
Точки на внешних границах, на профиле и внутри области генерируются раздельно.

Поддерживаются методы:
- regular      : равномерная сетка
- irregular    : равномерное случайное распределение
- adaptive_foil: сгущение у профиля (на основе расстояния до профиля)
- adaptive_all : сгущение ко всем границам (на основе расстояния до ближайшей границы)

Все методы поддерживают многопроцессорность (с корректной работой на Windows).
Пользователь может задать параметры alpha (степень сгущения) и epsilon_factor (масштаб сгущения),
а также выбрать тип верхней границы (INLET или OUTLET).

Исправленная версия: улучшена производительность регулярной сетки (векторизованная проверка),
исправлены обработчики исключений (замена голых except на конкретные).
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
from matplotlib.path import Path  # для векторизованной проверки внутри области
from multiprocessing import Process, Queue, freeze_support
import os
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict
import sys
import queue  # для queue.Empty
from scipy.spatial import cKDTree

# Попытка импорта tqdm для прогресс-бара
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEOMETRY_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Sampler", "Data")

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
    id: int
    x: float
    y: float
    boundary_type: BoundaryType
    on_airfoil: bool = False
    on_boundary: bool = False
    segment_id: int = -1

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'boundary_type': self.boundary_type.value,
            'on_airfoil': self.on_airfoil,
            'on_boundary': self.on_boundary,
            'segment_id': self.segment_id
        }

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class SamplerConfig:
    num_processes: int = 4
    step_size: float = 0.01
    step_units: str = 'm'           # 'm' или 'mm'
    use_multiprocessing: bool = True
    show_progress: bool = True
    save_format: str = 'npy'         # 'npy', 'csv'
    plot_results: bool = True
    plot_dpi: int = 300

    geometry_dir: str = DEFAULT_GEOMETRY_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR

    boundary_point_density: float = 1.0
    interior_point_density: float = 1.0

    # Тип симметрии для нижней границы (для разомкнутого профиля)
    symmetry_type: str = "symmetry"   # "symmetry" или "axis_symmetry"

    # Тип верхней границы (может быть "inlet" или "outlet")
    top_boundary_type: str = "outlet"

    # Метод генерации внутренних точек
    sampling_method: str = "regular"  # "regular", "irregular", "adaptive_foil", "adaptive_all"

    # Параметры адаптивного сгущения
    adaptive_alpha: float = 1.0               # степень сгущения
    adaptive_epsilon_factor: float = 0.01     # доля характерного размера для масштаба сгущения

    def __post_init__(self):
        os.makedirs(self.geometry_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.top_boundary_type not in ('inlet', 'outlet'):
            raise ValueError("top_boundary_type должно быть 'inlet' или 'outlet'")


# ----------------------------------------------------------------------
# Вспомогательные функции для многопроцессорной обработки (вынесены на уровень модуля)
# ----------------------------------------------------------------------

def _regular_worker(domain, step, bounds, proc_id, total_procs, result_queue, progress_queue):
    """Worker для регулярной сетки (использует векторизованную проверку Path)."""
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


def _irregular_worker(domain, bounds, target, proc_id, result_queue, progress_queue):
    """Worker для нерегулярной (случайной) сетки."""
    path = Path(domain)
    x_min, x_max, y_min, y_max = bounds
    points = []
    attempts = 0
    max_attempts = target * 20
    last_report = 0
    while len(points) < target and attempts < max_attempts:
        # Генерируем батч случайных точек
        batch_size = min(10000, max_attempts - attempts)
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


def _adaptive_worker(domain, boundary_points, bounds, target, alpha, epsilon_factor, proc_id,
                     result_queue, progress_queue):
    """
    Worker для адаптивного метода (сгущение к заданным граничным точкам).
    boundary_points – массив координат точек, к которым выполняется сгущение.
    """
    path = Path(domain)
    # Строим дерево внутри процесса
    tree = cKDTree(boundary_points)
    x_min, x_max, y_min, y_max = bounds
    char_length = (x_max - x_min + y_max - y_min) / 2
    eps = char_length * epsilon_factor

    points = []
    batch_size = 10000
    max_attempts = target * 20
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
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.points: List[CollocationPoint] = []
        self.points_interior: List[CollocationPoint] = []
        self.points_boundary: List[CollocationPoint] = []
        self.metadata = {}
        self.foil_metadata = {}
        self.next_id = 0
        self.foil_coords = None
        self.domain_coords = None
        self.foil_is_closed = False

        os.makedirs(config.output_dir, exist_ok=True)

    def _get_next_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def find_geometry_files(self) -> Dict[str, str]:
        """Поиск файлов геометрии"""
        result = {'foil': None, 'domain': None, 'metadata': None}
        foil_names = ["airfoil_foil.npy", "airfoil_foil.txt"]
        domain_names = ["airfoil_domain.npy", "airfoil_domain.txt"]
        meta_names = ["airfoil_metadata.npy"]

        for name in foil_names:
            path = os.path.join(self.config.geometry_dir, name)
            if os.path.exists(path):
                result['foil'] = path
                print(f"   ✅ Найден профиль: {name}")
                break
        for name in domain_names:
            path = os.path.join(self.config.geometry_dir, name)
            if os.path.exists(path):
                result['domain'] = path
                print(f"   ✅ Найдена область: {name}")
                break
        for name in meta_names:
            path = os.path.join(self.config.geometry_dir, name)
            if os.path.exists(path):
                result['metadata'] = path
                print(f"   ✅ Найдены метаданные: {name}")
                break
        return result

    def load_geometries(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
    def sample_boundary_external(self, contour: np.ndarray, step: float,
                                  bounds: Dict) -> List[CollocationPoint]:
        """Точки на внешнем контуре области (равномерно)"""
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

    def _classify_external_boundary(self, x: float, y: float,
                                    bounds: Dict, step: float) -> BoundaryType:
        """Определение типа точки на внешнем прямоугольном контуре"""
        tol = step * 0.1
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']

        if abs(x - x_min) < tol:
            return BoundaryType.INLET
        elif abs(x - x_max) < tol:
            return BoundaryType.OUTLET
        elif abs(y - y_max) < tol:
            return BoundaryType(self.config.top_boundary_type)
        elif abs(y - y_min) < tol:
            if self.foil_is_closed:
                return BoundaryType.INLET
            else:
                return BoundaryType(self.config.symmetry_type)
        else:
            return BoundaryType.OUTLET

    # ------------------------------------------------------------------
    # Генерация точек на профиле (WALL)
    # ------------------------------------------------------------------
    def sample_boundary_foil(self, foil: np.ndarray, step: float) -> List[CollocationPoint]:
        """Точки на профиле (равномерно по длине)"""
        line = LineString(foil)
        length = line.length
        if self.foil_is_closed:
            line = LineString(foil[:-1])
            length = line.length

        num_points = int(length / step) + 1
        points = []
        for i in range(num_points):
            distance = i * step
            if distance <= length:
                pt = line.interpolate(distance)
                points.append(CollocationPoint(
                    id=self._get_next_id(),
                    x=pt.x, y=pt.y,
                    boundary_type=BoundaryType.WALL,
                    on_airfoil=True,
                    on_boundary=True
                ))
        logger.info(f"✅ Профиль (WALL): {len(points)} точек")
        return points

    # ------------------------------------------------------------------
    # Генерация нижней границы (отдельно, если профиль не доходит до левой границы)
    # ------------------------------------------------------------------
    def sample_bottom_edge(self, x_min_dom: float, x_min_foil: float,
                           y_min_dom: float, step: float) -> List[CollocationPoint]:
        """
        Генерация точек на нижней границе слева от профиля.
        ВСЕГДА используется равномерное распределение для граничных точек.
        """
        if x_min_foil <= x_min_dom:
            return []
        bottom_length = x_min_foil - x_min_dom
        num_points = int(bottom_length / step) + 1
        # Всегда равномерно, независимо от sampling_method
        xs = np.linspace(x_min_dom, x_min_foil, num_points)

        btype = BoundaryType.SYMMETRY if not self.foil_is_closed else BoundaryType.INLET
        if self.config.symmetry_type == 'axis_symmetry' and not self.foil_is_closed:
            btype = BoundaryType.AXIS_SYMMETRY

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
    def sample_interior_regular_sequential(self, domain: np.ndarray, step: float) -> List[CollocationPoint]:
        """
        Генерация регулярных точек с использованием векторизованной проверки Path.
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

    def sample_interior_regular_parallel(self, domain: np.ndarray, step: float,
                                         num_processes: int) -> List[CollocationPoint]:
        """
        Параллельная версия для регулярной сетки (разбиение по X).
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
    def sample_interior_irregular(self, domain: np.ndarray, num_points: int) -> List[CollocationPoint]:
        if self.config.use_multiprocessing and self.config.num_processes > 1:
            return self._sample_interior_irregular_parallel(domain, num_points)
        else:
            return self._sample_interior_irregular_sequential(domain, num_points)

    def _sample_interior_irregular_sequential(self, domain: np.ndarray,
                                              num_points: int) -> List[CollocationPoint]:
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        points = []
        attempts = 0
        max_attempts = num_points * 10
        if TQDM_AVAILABLE and self.config.show_progress:
            pbar = tqdm(total=num_points, desc="Внутренние (нерегулярный)")
        else:
            pbar = None
            last_log = time.time()
        while len(points) < num_points and attempts < max_attempts:
            # Генерируем батч
            batch_size = min(10000, max_attempts - attempts)
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
            if not pbar and self.config.show_progress and time.time() - last_log > 2:
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
    def sample_interior_adaptive(self, domain: np.ndarray, num_points: int,
                                 alpha: float, epsilon_factor: float, mode: str) -> List[CollocationPoint]:
        """
        mode: 'foil' - сгущение к профилю, 'all' - сгущение ко всем границам
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
        path = Path(domain)
        minx, miny, maxx, maxy = path.get_extents().bounds
        char_length = (maxx - minx + maxy - miny) / 2
        eps = char_length * epsilon_factor

        # Строим дерево по переданным граничным точкам
        tree = cKDTree(boundary_points)

        points = []
        batch_size = 10000
        max_attempts = num_points * 50
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
    def generate_all_points(self, domain: np.ndarray, foil: Optional[np.ndarray]) -> Dict:
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
        logger.info(f"🔼 Верхняя граница: {self.config.top_boundary_type}")

        x_min_dom, y_min_dom = np.min(domain, axis=0)
        x_max_dom, y_max_dom = np.max(domain, axis=0)
        bounds = {'x_min': x_min_dom, 'x_max': x_max_dom,
                  'y_min': y_min_dom, 'y_max': y_max_dom}
        logger.info(f"📐 Область: x=[{x_min_dom:.4f}, {x_max_dom:.4f}], y=[{y_min_dom:.4f}, {y_max_dom:.4f}]")

        poly_domain = Polygon(domain)
        if not poly_domain.is_valid:
            poly_domain = poly_domain.buffer(0)
        area = poly_domain.area
        num_interior = int(area / step_m ** 2 * self.config.interior_point_density)

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
            'symmetry_type': self.config.symmetry_type,
            'top_boundary_type': self.config.top_boundary_type,
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
    def plot_points(self, domain: np.ndarray, foil: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
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

    def save_points(self):
        base = self.output_base
        if self.config.save_format == 'npy':
            dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'),
                     ('boundary_type', 'U20'), ('on_airfoil', 'bool'),
                     ('on_boundary', 'bool'), ('segment_id', 'i4')]
            arr = np.array([(p.id, p.x, p.y, p.boundary_type.value,
                             p.on_airfoil, p.on_boundary, p.segment_id)
                            for p in self.points], dtype=dtype)
            np.save(f"{base}.npy", arr)
            np.save(f"{base}_interior.npy",
                    np.array([(p.x, p.y) for p in self.points_interior]))
            np.save(f"{base}_boundary.npy",
                    np.array([(p.x, p.y) for p in self.points_boundary]))
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
        return [p for p in self.points if p.boundary_type == btype]


def S2DR():
    print('\033c' + "=" * 60 + "\nМОДУЛЬ ДИСКРЕТИЗАЦИИ S2DR".center(60) + "\n" + "=" * 60)
    try:
        if os.name == 'nt':
            freeze_support()

        num_proc = int(input("Количество ядер процессора (1-16): "))
        if not 1 <= num_proc <= 16:
            raise ValueError("Некорректное число ядер")

        print("\nЕдиницы шага:\n   1 - метры\n   2 - миллиметры")
        units_choice = input("Выберите (1/2): ").strip()
        step_units = 'm' if units_choice == '1' else 'mm'
        step = float(input(f"Шаг сетки ({step_units}): "))
        if step <= 0:
            raise ValueError("Шаг должен быть >0")

        print("\nТип симметрии нижней границы:")
        print("   1 - symmetry\n   2 - axis_symmetry")
        sym_choice = input("Выберите (1/2): ").strip()
        symmetry_type = 'symmetry' if sym_choice == '1' else 'axis_symmetry'

        print("\nТип верхней границы:")
        print("   1 - outlet (обычно для выхода потока)")
        print("   2 - inlet  (если поток входит сверху)")
        top_choice = input("Выберите (1/2): ").strip()
        top_boundary_type = 'outlet' if top_choice == '1' else 'inlet'

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
            symmetry_type=symmetry_type,
            top_boundary_type=top_boundary_type,
            sampling_method=sampling_method,
            adaptive_alpha=adaptive_alpha,
            adaptive_epsilon_factor=adaptive_epsilon_factor
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

        if config.plot_results:
            print("\n📊 Построение графика...")
            plot_path = os.path.join(config.output_dir, "collocation_points.png")
            sampler.plot_points(domain, foil, save_path=plot_path)

        print("\n💾 Сохранение...")
        sampler.save_points()

        print("\n" + "=" * 60 + "\n✅ МОДУЛЬ S2DR УСПЕШНО ВЫПОЛНЕН\n" + "=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        logger.error("Ошибка", exc_info=True)

    input("\nНажмите Enter для возврата...")


if __name__ == "__main__":
    S2DR()