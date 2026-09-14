from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль инициализации расчётной области для Pipe-Bend PINN.

Вычисляет нормализованные поля для PINN:
- координаты делятся на pipe_width
- скорости делятся на U_inf
- давление переводится в избыточное и делится на ρU_inf²
- SDF = расстояние до ближайшей стенки (внешней или внутренней)
- на стенке скорость обнуляется

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, List, Tuple, Dict, Any
import glob
import multiprocessing as mp
from functools import partial

from scipy.spatial import cKDTree
from pipe_bend_pinn.geometry.sdf import PolylineDistanceField
from pipe_bend_pinn.data.normalization import ReferenceScales
from pipe_bend_pinn.sampling.core import load_domain
from pipe_bend_pinn.physics.fluids import FluidLibrary, Fluid
from pipe_bend_pinn.physics.conditions import ConditionsManager, BoundaryConditions
GEOMETRY_DIR = GEOMETRY_DOMAINS_DIR
SAMPLER_DIR = SAMPLING_COLLOCATION_DIR
FLUIDS_DIR = FLUIDS_DIR
CONDITIONS_DIR = CONDITIONS_DIR
OUTPUT_DIR = INITIAL_FIELDS_DIR
PLOTS_DIR = INITIAL_PLOTS_DIR

logger = logging.getLogger(__name__)

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


class Initializer:
    """
    Класс для инициализации расчётной области с нормализацией.

    Для pipe bend:
    - Нормализация на pipe_width (вместо chord для профиля)
    - SDF вычисляется до обеих стенок (внешней и внутренней)
    - Начальное поле скорости: параболический профиль (Пуазейль) на входе
    - Начальное давление: линейное от inlet_pressure к outlet_pressure
    """

    def __init__(self) -> None:
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
        self.fluid: Optional[Fluid] = None
        self.conditions: Optional[BoundaryConditions] = None
        self.fluid_name: Optional[str] = None
        self.conditions_name: Optional[str] = None

        self.pipe_width: Optional[float] = None
        self.U_inf: Optional[float] = None
        self.p_ref: Optional[float] = None
        self.Re: Optional[float] = None
        self.rho: Optional[float] = None
        self.mu: Optional[float] = None
        self.inlet_y_min: Optional[float] = None
        self.inlet_y_max: Optional[float] = None
        self.velocity_initial_mode: Optional[str] = None
        self.psi_initial: Optional[np.ndarray] = None

    def load_sampler_points(self, filename: Optional[str] = None) -> np.ndarray:
        """Загружает точки коллокации из Sampler."""
        if filename is None:
            files = glob.glob(os.path.join(self.sampler_dir, "*.npy"))
            if not files:
                raise FileNotFoundError(f"Нет .npy файлов в {self.sampler_dir}")
            main_files = [f for f in files if not f.endswith('_interior.npy') and not f.endswith('_boundary.npy')]
            filename = main_files[0] if main_files else files[0]
            print(f"Использую файл точек: {os.path.basename(filename)}")
        data = np.load(filename, allow_pickle=True)
        if data.dtype.names is not None:
            self.points = data
        else:
            if data.ndim == 2 and data.shape[1] == 2:
                arr = np.zeros(len(data), dtype=[('id', 'i4'), ('x', 'f8'), ('y', 'f8')])
                arr['id'] = np.arange(len(data))
                arr['x'] = data[:, 0]
                arr['y'] = data[:, 1]
                self.points = arr
            else:
                raise ValueError("Неизвестный формат файла точек")
        logger.info(f"Загружено {len(self.points)} точек из Sampler.")
        return self.points

    def extract_wall_points(self) -> None:
        """Извлекает точки с типом WALL для построения SDF."""
        if self.points is None:
            raise RuntimeError("Сначала загрузите точки.")
        if 'boundary_type' not in self.points.dtype.names:
            logger.warning("В точках нет boundary_type. Использую все точки.")
            self.wall_points = np.column_stack((self.points['x'], self.points['y']))
            return
        wall_mask = self.points['boundary_type'] == 'wall'
        if not np.any(wall_mask):
            logger.warning("Не найдено точек WALL. SDF будет нулевым.")
            self.wall_points = np.empty((0, 2))
        else:
            self.wall_points = np.column_stack((self.points['x'][wall_mask], self.points['y'][wall_mask]))
            logger.info(f"Найдено {len(self.wall_points)} точек WALL для SDF.")

    def _load_pipe_width_from_metadata(self) -> float:
        """Определяет pipe_width из метаданных или оценивает по разбросу y на inlet."""
        meta_path = os.path.join(GEOMETRY_DIR, "pipe_bend_metadata.npy")
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path, allow_pickle=True).item()
                params = meta.get('pipe_bend_params', {})
                pw = params.get('pipe_width', None)
                if pw is not None:
                    return float(pw)
            except Exception as e:
                logger.warning(f"Ошибка загрузки метаданных: {e}")
        # Оценка по разбросу y точек inlet
        if self.points is not None and 'boundary_type' in self.points.dtype.names:
            inlet_mask = self.points['boundary_type'] == 'inlet'
            if np.any(inlet_mask):
                y_vals = self.points['y'][inlet_mask]
                pw = y_vals.max() - y_vals.min()
                logger.warning(f"pipe_width оценён по точкам inlet: {pw:.4f} м")
                return float(pw)
        logger.warning("pipe_width не найден, используется 0.5 м по умолчанию")
        return 0.5

    def _ordered_wall_chains(self, tol_factor: float = 2.0) -> List[np.ndarray]:
        """Разбивает точки стенок на упорядоченные цепочки по большим разрывам.

        Точки стенок в коллокации сгенерированы по периметру контура, поэтому в
        массиве они идут подряд вдоль стенки. Между концом внутренней стенки и
        началом внешней (через вход/выход) есть разрыв размером ~pipe_width.
        Разделяем по этим разрывам, получая отдельные открытые полилинии.
        """
        if self.wall_points is None or len(self.wall_points) < 2:
            return []
        pts = np.asarray(self.wall_points, dtype=np.float64)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0:1]])
        delta = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        median_gap = float(np.median(delta))
        threshold = tol_factor * max(median_gap, 1e-12)
        breaks = np.where(delta > threshold)[0]
        chains = []
        start = 0
        for boundary in np.append(breaks, len(pts) - 1):
            chain = pts[start:boundary + 1]
            if len(chain) >= 2:
                chains.append(chain)
            start = boundary + 1
        return chains

    def _compute_segment_sdf(self, xs: np.ndarray, ys: np.ndarray) -> Optional[np.ndarray]:
        """Точный сегментный SDF через PolylineDistanceField (в метрах).

        Возвращает массив расстояний до ближайшей стенки; None — если
        упорядоченные цепочки стенок построить не удалось.
        """
        chains = self._ordered_wall_chains()
        if not chains:
            return None
        query = np.column_stack((xs, ys))
        distance = None
        for chain in chains:
            field = PolylineDistanceField(chain, closed=False, segment_chunk_size=256)
            d = field.evaluate_numpy(query, point_chunk_size=8192)
            distance = d if distance is None else np.minimum(distance, d)
        return distance

    def _stream_function_geometry(self) -> Tuple[Optional[float], Optional[float]]:
        """Возвращает (R_outer, R_inner) из геометрии отвода или (None, None).

        Контур домена лежит в абсолютных координатах с центром изгиба в (0, 0):
        внешняя дуга радиусом R_outer, внутренняя — R_inner.
        """
        try:
            domain = load_domain()
        except Exception:
            return None, None
        if domain is None or len(domain) < 4:
            return None, None
        r_outer = float(-np.min(domain[:, 1]))
        width = max(self.pipe_width or 0.0, 1e-12)
        return r_outer, r_outer - width

    def _compute_stream_function_field(
        self, u0: float, v_in: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Начальное поле через функцию тока ψ: u = ∂ψ/∂y, v = −∂ψ/∂x.

        Поле по построению бездивергентно (div u = 0 тождественно), что даёт
        физически согласованную стартовую точку для PINN. Профиль поперёк
        канала — параболический (Пуазейль): F(t) = 2t² − (4/3)t³.

        В прямых участках поперечная координата t отсчитывается от внешней
        стенки, в изгибе — радиально от внешней дуги. ψ непрерывна на стыках.
        """
        x = self.points['x']
        y = self.points['y']
        N = len(x)
        r_outer, r_inner = self._stream_function_geometry()
        width = self.pipe_width if self.pipe_width else 1.0

        def poiseuille(t: np.ndarray) -> np.ndarray:
            t = np.clip(t, 0.0, 1.0)
            return 2.0 * t**2 - (4.0 / 3.0) * t**3

        if r_outer is None or r_inner is None:
            # Запасной вариант: поперечная координата только по y (входной канал).
            if 'boundary_type' in self.points.dtype.names:
                inlet_mask = self.points['boundary_type'] == 'inlet'
                if np.any(inlet_mask):
                    y0, y1 = float(np.min(y[inlet_mask])), float(np.max(y[inlet_mask]))
                else:
                    y0, y1 = float(np.min(y)), float(np.max(y))
            else:
                y0, y1 = float(np.min(y)), float(np.max(y))
            channel = max(y1 - y0, 1e-12)
            t = np.clip((y - y0) / channel, 0.0, 1.0)
            psi = u0 * channel * poiseuille(t)
            u = u0 * (4.0 * t - 4.0 * t**2)
            v = np.full(N, v_in)
            return u, v, psi

        r = np.hypot(x, y)
        eps = 1e-9
        psi = np.zeros(N)
        u = np.zeros(N)
        v = np.zeros(N)
        for i in range(N):
            xi, yi = x[i], y[i]
            if xi <= eps and yi <= eps:
                # Входной канал: x <= 0, y in [-R_outer, -R_inner]
                t = float(np.clip((yi + r_outer) / width, 0.0, 1.0))
                psi[i] = u0 * width * poiseuille(np.array(t))
                u[i] = u0 * (4.0 * t - 4.0 * t**2)
            elif xi >= -eps and yi >= -eps:
                # Выходной канал: y >= 0, x in [R_inner, R_outer]
                t = float(np.clip((xi - r_inner) / width, 0.0, 1.0))
                psi[i] = u0 * width * (poiseuille(np.array(1.0)) - poiseuille(np.array(t)))
                v[i] = u0 * (4.0 * t - 4.0 * t**2)
            else:
                # Изгиб: x > 0, y < 0, r in [R_inner, R_outer]
                ri = max(r[i], 1e-12)
                t = float(np.clip((r_outer - ri) / width, 0.0, 1.0))
                psi[i] = u0 * width * poiseuille(np.array(t))
                fprime = u0 * (4.0 * t - 4.0 * t**2)
                u[i] = -fprime * yi / ri
                v[i] = fprime * xi / ri
        return u, v, psi

    def select_fluid(self) -> str:
        """Интерактивный выбор жидкости."""
        fluids = self.fluid_lib.list_fluids()
        if not fluids:
            raise RuntimeError("Библиотека жидкостей пуста.")
        print("\nДоступные жидкости:")
        for i, name in enumerate(fluids, 1):
            fluid = self.fluid_lib.get(name)
            print(f"{i}. {fluid.display()}")
        while True:
            try:
                choice = input("Выберите номер жидкости: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(fluids):
                    self.fluid = self.fluid_lib.get(fluids[idx])
                    self.fluid_name = fluids[idx]
                    return fluids[idx]
                print("Неверный номер.")
            except ValueError:
                print("Введите число.")

    def select_conditions(self) -> str:
        """Интерактивный выбор граничных условий."""
        names = self.cond_manager.list_names()
        if not names:
            raise RuntimeError("Библиотека условий пуста.")
        print("\nДоступные наборы условий:")
        for i, name in enumerate(names, 1):
            cond = self.cond_manager.get(name)
            print(f"{i}. {name}")
            if cond:
                print(f"   {cond.display(self.fluid_lib)}")
        while True:
            try:
                choice = input("Выберите номер условий: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    self.conditions = self.cond_manager.get(names[idx])
                    self.conditions_name = names[idx]
                    return names[idx]
                print("Неверный номер.")
            except ValueError:
                print("Введите число.")

    def initialize_fields(self, pressure_mode: str = 'linear_x',
                          velocity_initial_mode: str = 'parabolic',
                          temperature: float = 293.15,
                          perturbation_amplitude: float = 0.01,
                          num_processes: Optional[int] = None) -> np.ndarray:
        """Инициализирует нормализованные поля для pipe bend."""
        if self.points is None:
            raise RuntimeError("Сначала загрузите точки.")
        if self.fluid is None:
            raise RuntimeError("Сначала выберите жидкость.")
        if self.conditions is None:
            raise RuntimeError("Сначала выберите граничные условия.")
        if self.wall_points is None:
            self.extract_wall_points()

        self.pipe_width = self._load_pipe_width_from_metadata()
        u_in = self.conditions.inlet_u
        v_in = self.conditions.inlet_v
        self.U_inf = np.sqrt(u_in**2 + v_in**2) or 1.0
        self.p_ref = self.conditions.outlet_pressure
        self.rho = self.fluid.density
        self.mu = self.fluid.dynamic_viscosity
        nu = self.fluid.kinematic_viscosity
        self.Re = self.U_inf * self.pipe_width / nu

        logger.info(f"pipe_width: {self.pipe_width:.4f} м")
        logger.info(f"U_inf = {self.U_inf:.4f} м/с")
        logger.info(f"Re = {self.Re:.2e}")

        N = len(self.points)
        arr_norm = np.zeros(N, dtype=point_dtype)
        for name in ['id', 'boundary_type', 'on_boundary']:
            if name in self.points.dtype.names:
                arr_norm[name] = self.points[name]
        if 'id' not in self.points.dtype.names:
            arr_norm['id'] = np.arange(N)

        arr_norm['x_norm'] = self.points['x'] / self.pipe_width
        arr_norm['y_norm'] = self.points['y'] / self.pipe_width

        # Начальное поле скорости
        x = self.points['x']
        x_min, x_max = np.min(x), np.max(x)
        u0 = self.conditions.inlet_u

        if velocity_initial_mode == 'parabolic':
            self.velocity_initial_mode = 'parabolic'
            # Параболический профиль на входе, линейное затухание по длине канала
            y = self.points['y']
            if 'boundary_type' in self.points.dtype.names:
                inlet_mask = self.points['boundary_type'] == 'inlet'
                if np.any(inlet_mask):
                    y_inlet = y[inlet_mask]
                    y_min_inlet, y_max_inlet = np.min(y_inlet), np.max(y_inlet)
                    y_range = y_max_inlet - y_min_inlet
                    if y_range > 0:
                        y_local = (y - y_min_inlet) / y_range  # 0..1
                        u_base = self.conditions.get_u_profile(y_local * self.pipe_width) if hasattr(self.conditions, 'get_u_profile') else u0 * (1 - (2 * y_local - 1) ** 2)
                    else:
                        u_base = np.full(N, u0)
                else:
                    u_base = np.full(N, u0)
            else:
                u_base = np.full(N, u0)
            # Затухание вдоль канала (линейное 0% на входе, 30% на выходе)
            if x_max > x_min:
                decay = 1.0 - 0.3 * (x - x_min) / (x_max - x_min)
            else:
                decay = 1.0
            u_phys = u_base * decay  # параболический профиль * затухание по длине
            v_phys = np.full(N, v_in)

            # ВЫХОД: pressure-outlet. Скорость НЕ задаётся (ни парабола, ни uniform),
            # профиль на выходе развивается из PDE. Баланс массы обеспечивается
            # integral_condition в LossComputer. В данных на выходе остаётся
            # только p = outlet_pressure (p_norm = 0).
        elif velocity_initial_mode == 'stream_function':
            # Функция тока: psi = ∫u dy, u = ∂ψ/∂y, v = −∂ψ/∂x.
            # Начальное поле бездивергентно по построению.
            self.velocity_initial_mode = 'stream_function'
            u_phys, v_phys, psi_phys = self._compute_stream_function_field(u0, v_in)
            self.psi_initial = psi_phys
        else:
            self.velocity_initial_mode = 'constant'
            u_phys = np.full(N, u0)
            v_phys = np.full(N, v_in)

        # Граничные условия
        if 'boundary_type' in self.points.dtype.names:
            wall_mask = self.points['boundary_type'] == 'wall'
            u_phys[wall_mask] = 0.0
            v_phys[wall_mask] = 0.0

        # Давление (линейное от inlet к outlet)
        if pressure_mode == 'constant':
            p_phys = np.full(N, self.conditions.inlet_pressure)
        else:  # linear_x
            if x_max > x_min:
                slope = (self.conditions.outlet_pressure - self.conditions.inlet_pressure) / (x_max - x_min)
                p_phys = self.conditions.inlet_pressure + slope * (x - x_min)
            else:
                p_phys = np.full(N, self.conditions.inlet_pressure)
        # На выходе давление строго равно outlet_pressure (избыточное p_norm = 0)
        if 'boundary_type' in self.points.dtype.names:
            outlet_mask = self.points['boundary_type'] == 'outlet'
            p_phys[outlet_mask] = self.conditions.outlet_pressure

        # Нормализация (согласовано с Airfoil PINN: p_norm = Cp = (p - p_ref) / (0.5 * rho * U_inf^2))
        arr_norm['u_norm'] = u_phys / self.U_inf
        arr_norm['v_norm'] = v_phys / self.U_inf
        arr_norm['p_norm'] = (p_phys - self.p_ref) / (0.5 * self.rho * self.U_inf**2)

        # SDF
        if len(self.wall_points) == 0:
            sdf_phys = np.zeros(N)
        else:
            logger.info("Вычисление SDF...")
            xs = self.points['x'].copy()
            ys = self.points['y'].copy()
            # Основной путь: точный сегментный SDF по полилиниям стенок.
            sdf_phys = self._compute_segment_sdf(xs, ys)
            if sdf_phys is None:
                # Запасной путь: cKDTree по точкам стенок.
                logger.info("Сегментный SDF недоступен, использую cKDTree по точкам стенок.")
                if num_processes is None:
                    num_processes = mp.cpu_count()
                num_processes = min(num_processes, N, 16)
                if num_processes > 1:
                    logger.info(f"Используется процессов: {num_processes}")
                    indices = np.arange(N)
                    chunks = np.array_split(indices, num_processes)
                    sdf_phys = np.zeros(N)
                    for chunk_idx in chunks:
                        for i in chunk_idx:
                            dx = xs[i] - self.wall_points[:, 0]
                            dy = ys[i] - self.wall_points[:, 1]
                            dist_sq = dx**2 + dy**2
                            sdf_phys[i] = np.sqrt(np.min(dist_sq))
                else:
                    tree = cKDTree(self.wall_points)
                    sdf_phys = np.zeros(N)
                    for i in range(N):
                        dist, _ = tree.query(np.array([xs[i], ys[i]]), k=1)
                        sdf_phys[i] = dist

        arr_norm['sdf_norm'] = sdf_phys / self.pipe_width
        self.points_norm = arr_norm
        logger.info("Нормализованные поля инициализированы.")
        return arr_norm

    def save(self, description: str = "initial") -> None:
        if self.points_norm is None:
            raise RuntimeError("Нет данных для сохранения.")
        base = os.path.join(self.output_dir, f"{description}.npy")
        np.save(base, self.points_norm)
        logger.info(f"Сохранено: {base}")

        inlet_u_norm = self.conditions.inlet_u / self.U_inf if self.U_inf > 0 else 1.0
        inlet_v_norm = self.conditions.inlet_v / self.U_inf if self.U_inf > 0 else 0.0
        # Единая система масштабов (порт ReferenceScales из Airfoil PINN)
        try:
            reference = ReferenceScales(
                pipe_width=self.pipe_width,
                velocity=self.U_inf,
                density=self.rho,
                dynamic_viscosity=self.mu,
                static_pressure=self.p_ref,
            )
            normalization_reference = reference.to_metadata()
        except Exception as exc:
            logger.warning(f"Не удалось построить ReferenceScales: {exc}")
            normalization_reference = None
        meta = {
            'description': description,
            'fluid_name': self.fluid_name,
            'conditions_name': self.conditions_name,
            'num_points': len(self.points_norm),
            'pipe_width': self.pipe_width,
            'U_inf': self.U_inf,
            'p_ref': self.p_ref,
            'rho': self.rho,
            'mu': self.mu,
            'Re': self.Re,
            'normalized': True,
            'sdf_norm_factor': self.pipe_width,
            'inlet_u_norm': inlet_u_norm,
            'inlet_v_norm': inlet_v_norm,
            'geom_type': 'pipe_bend',
            'velocity_initial_mode': self.velocity_initial_mode,
            'field_role': 'initial_guess_only',
            'supervised_reference': False,
            'normalization_reference': normalization_reference,
        }
        meta_path = os.path.join(self.output_dir, f"{description}_metadata.npy")
        np.save(meta_path, meta)
        logger.info(f"Метаданные сохранены: {meta_path}")
        # Первичное поле через функцию тока: сохраняем ψ отдельным файлом.
        if self.psi_initial is not None:
            psi_path = os.path.join(self.output_dir, f"{description}_psi.npy")
            np.save(psi_path, np.asarray(self.psi_initial, dtype=np.float64))
            logger.info(f"Функция тока сохранена: {psi_path}")

    def plot_field(self, field_name: str, title: Optional[str] = None,
                   save: bool = True, show: bool = False) -> None:
        if self.points_norm is None:
            raise RuntimeError("Нет данных для визуализации.")
        plt.figure(figsize=(12, 8))
        x = self.points_norm['x_norm'] * self.pipe_width
        y = self.points_norm['y_norm'] * self.pipe_width
        if field_name == 'sdf_norm':
            values = self.points_norm['sdf_norm'] * self.pipe_width
            label = 'SDF (расстояние до стенки, м)'
        else:
            values = self.points_norm[field_name]
            label = field_name
        plt.scatter(x, y, c=values, s=1, cmap='jet', alpha=0.6)
        plt.colorbar(label=label)
        plt.title(title or f"Поле {field_name}")
        plt.xlabel('x, м')
        plt.ylabel('y, м')
        plt.axis('equal')
        plt.grid(alpha=0.3)
        if save:
            filename = os.path.join(self.plots_dir, f"{field_name}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранён: {filename}")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_fields(self) -> None:
        fields = [
            ('p_norm', 'Нормализованное давление'),
            ('u_norm', 'Нормализованная скорость U'),
            ('v_norm', 'Нормализованная скорость V'),
            ('sdf_norm', 'SDF (расстояние до стенки, м)')
        ]
        for fname, label in fields:
            self.plot_field(fname, title=label)


def initialize_menu() -> None:
    """Интерактивное меню инициализации."""
    init = Initializer()
    print("\n" + "=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ РАСЧЁТНОЙ ОБЛАСТИ".center(60))
    print("=" * 60)
    try:
        init.load_sampler_points()
        init.extract_wall_points()
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        input("Нажмите Enter для возврата...")
        return
    try:
        init.select_fluid()
    except Exception as e:
        print(f"❌ {e}")
        input("Нажмите Enter для возврата...")
        return
    try:
        init.select_conditions()
    except Exception as e:
        print(f"❌ {e}")
        input("Нажмите Enter для возврата...")
        return
    print("\nРежим начального поля скорости:")
    print("   1 - параболический профиль (Пуазейль)")
    print("   2 - постоянное")
    print("   3 - функция тока (бездивергентное поле)")
    vel_choice = input("Выберите (1/2/3): ").strip()
    if vel_choice == '1':
        vel_mode = 'parabolic'
    elif vel_choice == '3':
        vel_mode = 'stream_function'
    else:
        vel_mode = 'constant'
    print("\n🔧 Вычисление нормализованных полей...")
    try:
        init.initialize_fields(pressure_mode='linear_x', velocity_initial_mode=vel_mode)
    except Exception as e:
        print(f"❌ Ошибка при вычислении полей: {e}")
        input("Нажмите Enter для возврата...")
        return
    desc = input("Введите описание для сохранения (по умолчанию 'initial'): ").strip() or "initial"
    init.save(description=desc)
    plot_choice = input("Построить графики полей? (Y/n): ").strip().lower()
    if plot_choice == '' or plot_choice in ('y', 'yes'):
        init.plot_all_fields()
    print("\n✅ Инициализация завершена.")
    input("Нажмите Enter для возврата...")


if __name__ == "__main__":
    initialize_menu()