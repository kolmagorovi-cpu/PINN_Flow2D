#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Модуль геометрии (G2D) для работы с аэродинамическими профилями.

ВСЕ КООРДИНАТЫ В МЕТРАХ (без нормализации).

Источник геометрии задаётся явно:
    - ``load_from_file`` загружает конкретный файл координат;
    - ``load_from_yaml`` загружает профиль по имени из библиотеки.

Example:
    >>> geom = Geometry2D()
    >>> geom.load_from_file("workspace/geometry/foils/naca0012.dat")
    >>> geom.create_domain(front=1.5, rear=3.0, top=1.2, bottom=1.2)
    >>> geom.save('my_geometry')

Автор: Колмагоров И.Д.
Версия: 4.1 (только профили)
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
import logging
from scipy import interpolate

from airfoil_pinn.app.paths import (
    GEOMETRY_DOMAINS_DIR,
    GEOMETRY_FOILS_DIR,
    GEOMETRY_METADATA_DIR,
    FOIL_GEOMS_PATH,
    PATHS,
)
from airfoil_pinn.core.type_hints import (
    GeometryConfigDict,
    DomainParamsDict,
    ClosedDomainParamsDict,
    GeometryMetadataDict,
)

# Геометрические артефакты хранятся в workspace/geometry.
# Константы оставлены строками, потому что большая часть модуля использует os.path.
PROJECT_ROOT = str(PATHS.project_root)
DEFAULT_OUTPUT_DIR = GEOMETRY_DOMAINS_DIR
DEFAULT_METADATA_DIR = GEOMETRY_METADATA_DIR
DEFAULT_PLOT_DPI = 300

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GeometryConfig:
    """
    Конфигурация для геометрических параметров.

    Attributes:
        save_format: Формат сохранения ('npy', 'csv', 'txt')
        plot_enabled: Включена ли визуализация
        plot_show: Показывать ли график сразу
        plot_dpi: DPI для сохранения графиков
        output_dir: Директория для выходных файлов
        verbose: Подробный вывод в консоль
    """
    save_format: str = 'npy'
    plot_enabled: bool = True
    plot_show: bool = True
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: str = DEFAULT_OUTPUT_DIR
    verbose: bool = True

    def to_dict(self) -> GeometryConfigDict:
        """Преобразует конфигурацию в словарь."""
        return GeometryConfigDict(
            save_format=self.save_format,
            plot_enabled=self.plot_enabled,
            plot_show=self.plot_show,
            plot_dpi=self.plot_dpi,
            output_dir=self.output_dir,
            verbose=self.verbose
        )


class AirfoilGeometry:
    """
    Класс для работы с геометрией профиля и расчётной области.

    ВСЕ КООРДИНАТЫ В МЕТРАХ.

    Attributes:
        foil_coords: Координаты точек профиля (N, 2)
        domain_coords: Координаты точек расчётной области
        is_closed: Флаг замкнутости профиля
        config: Конфигурация геометрии
        chord: Хорда профиля [м]
    """

    def __init__(self, config: Optional[GeometryConfig] = None) -> None:
        """
        Инициализирует геометрию профиля.

        Args:
            config: Конфигурация геометрии (по умолчанию используется GeometryConfig())
        """
        self.foil_coords: Optional[np.ndarray] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.is_closed: Optional[bool] = None
        self.config = config or GeometryConfig()
        self.chord: float = 1.0
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_airfoil(self, filepath: str) -> bool:
        """
        Загрузка профиля из файла (координаты в метрах).

        Args:
            filepath: Путь к файлу с координатами (.npy, .csv, .txt, .dat)

        Returns:
            True если загрузка успешна, иначе False

        Example:
            >>> geom = AirfoilGeometry()
            >>> success = geom.load_airfoil('naca0012.dat')
        """
        try:
            ext = os.path.splitext(filepath)[1].lower()

            if ext == '.npy':
                self.foil_coords = np.load(filepath)
            elif ext == '.csv':
                self.foil_coords = np.loadtxt(filepath, delimiter=',', skiprows=1)
            else:
                self.foil_coords = self._load_text_file(filepath)

            if self.foil_coords.shape[1] != 2:
                logger.error(f"Файл должен содержать 2 столбца, получено {self.foil_coords.shape[1]}")
                return False

            # Вычисляем хорду
            x_min = np.min(self.foil_coords[:, 0])
            x_max = np.max(self.foil_coords[:, 0])
            self.chord = x_max - x_min

            # Проверяем замкнутость
            self.is_closed = self._check_closed()

            if self.config.verbose:
                logger.info(f"✅ Профиль загружен: {len(self.foil_coords)} точек")
                logger.info(f"📏 Хорда профиля: {self.chord:.4f} м")
                logger.info(f"📐 Замкнутый: {'✅' if self.is_closed else '❌'}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке профиля: {e}")
            return False

    def _load_text_file(self, filepath: str) -> np.ndarray:
        """
        Загрузка из текстового файла (столбцы, разделённые пробельными символами).

        Args:
            filepath: Путь к текстовому файлу с координатами

        Returns:
            Массив координат формы (N, 2)
        """
        coords = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue
        return np.array(coords)

    def _check_closed(self, tolerance: float = 1e-6) -> bool:
        """
        Проверка замкнутости профиля.

        Args:
            tolerance: Допуск для сравнения первой и последней точек [м]

        Returns:
            True если профиль замкнут, иначе False
        """
        if len(self.foil_coords) < 2:
            return False
        first = self.foil_coords[0]
        last = self.foil_coords[-1]
        return np.allclose(first, last, atol=tolerance)

    def create_domain(self, **kwargs: Union[float, int]) -> Optional[np.ndarray]:
        """
        Создание расчётной области (интерактивный ввод параметров).

        Args:
            **kwargs: Параметры области:
                - front: Длина домена перед профилем [м]
                - rear: Длина домена за профилем [м] (для замкнутого профиля)
                - top: Высота домена над профилем [м]
                - bottom: Высота домена под профилем [м] (для замкнутого профиля)

        Returns:
            Массив координат области формы (N, 2) или None в случае ошибки

        Example:
            >>> domain = geom.create_domain(front=0.5, top=0.3)
        """
        if self.foil_coords is None:
            logger.error("❌ Сначала загрузите профиль")
            return None

        if self.is_closed:
            self.domain_coords = self._create_closed_domain(kwargs)
            if self.config.verbose:
                logger.info(f"✅ Замкнутая область создана: {len(self.domain_coords)} вершин")
        else:
            self.domain_coords = self._create_unclosed_domain(kwargs)
            if self.config.verbose:
                logger.info(f"✅ Область создана: {len(self.domain_coords)} точек (замкнутый контур)")

        return self.domain_coords

    def _create_unclosed_domain(self, params: DomainParamsDict) -> np.ndarray:
        """
        Создание области для незамкнутого профиля (крыло).

        Создаёт ЗАМКНУТЫЙ контур, объединяя профиль и внешние границы.
        Используется исходный порядок точек профиля (без сортировки).

        Соглашение по проектированию: для незамкнутого профиля (открытая задняя кромка)
        носовая точка профиля ВСЕГДА лежит на оси симметрии y=0. Это не ошибка — это
        конструктивное ограничение: профиль задаётся в системе координат, где нижняя
        граница области совпадает с плоскостью симметрии y=0, а носок профиля
        находится в точке [x_min, 0]. Точка [-length, 0] — угол внешней границы
        на той же линии симметрии.

        Args:
            params: Параметры области (front, top)

        Returns:
            Массив координат замкнутого контура формы (N, 2)
        """
        # Запрашиваем параметры в метрах
        length = params.get('front')
        if length is None:
            length = float(input("Введите длину домена перед профилем, м: "))

        height = params.get('top')
        if height is None:
            height = float(input("Введите высоту домена над профилем, м: "))

        # Границы профиля (в метрах)
        x_min = np.min(self.foil_coords[:, 0])
        x_max = np.max(self.foil_coords[:, 0])
        nose_idx = np.argmin(self.foil_coords[:, 0])
        y_trailing = self._get_y_at_x(x_max)

        if self.config.verbose:
            print(f"\n📐 Параметры области (в метрах):")
            print(f"   Длина перед профилем: {length:.3f} м")
            print(f"   Высота над профилем: {height:.3f} м")
            print(f"   Носок профиля: ({x_min:.4f}, {self.foil_coords[nose_idx, 1]:.4f})")
            print(f"   Хвостик профиля: ({x_max:.4f}, {y_trailing:.4f})")

        # Строим ЗАМКНУТЫЙ контур, используя исходный порядок точек профиля
        domain_points = []

        # 1. Добавляем ВЕСЬ профиль в том порядке, как он задан в файле
        domain_points.extend(self.foil_coords.tolist())

        # 2. От хвостика вверх до верхней границы
        domain_points.append([x_max, height])

        # 3. Влево до левой границы
        domain_points.append([-length, height])

        # 4. Вниз до линии симметрии y=0 (ось/граница симметрии)
        domain_points.append([-length, 0.0])

        # 5. Вправо до носка профиля (на линии симметрии)
        domain_points.append([x_min, 0.0])

        # 6. Явное замыкание контура на первую точку
        domain_points.append(domain_points[0])

        return np.array(domain_points)

    def _create_closed_domain(self, params: ClosedDomainParamsDict) -> np.ndarray:
        """
        Создание ЗАМКНУТОЙ области для замкнутого профиля (профиль в трубе).

        Args:
            params: Параметры области (front, rear, top, bottom)

        Returns:
            Массив координат прямоугольной области формы (5, 2)
        """
        front = params.get('front')
        if front is None:
            front = float(input("Введите длину домена перед профилем, м: "))

        rear = params.get('rear')
        if rear is None:
            rear = float(input("Введите длину домена за профилем, м: "))

        top = params.get('top')
        if top is None:
            top = float(input("Введите высоту домена над профилем, м: "))

        bottom = params.get('bottom')
        if bottom is None:
            bottom = float(input("Введите высоту домена под профилем, м: "))

        # Прямоугольная область (замкнутая)
        domain = np.array([
            [-front, -bottom],
            [-front, top],
            [rear, top],
            [rear, -bottom],
            [-front, -bottom]  # Замыкаем
        ])

        return domain

    def _get_y_at_x(self, x_target: float) -> float:
        """
        Находит координату y на профиле при заданном x (линейная интерполяция).

        Args:
            x_target: Целевая X-координата [м]

        Returns:
            Y-координата на профиле [м]
        """
        # Сортируем точки по x для интерполяции
        sorted_indices = np.argsort(self.foil_coords[:, 0])
        sorted_coords = self.foil_coords[sorted_indices]

        # Удаляем дубликаты x (если есть)
        unique_x, unique_indices = np.unique(sorted_coords[:, 0], return_index=True)
        unique_coords = sorted_coords[unique_indices]

        if len(unique_coords) < 2:
            return self.foil_coords[0, 1]

        f = interpolate.interp1d(unique_coords[:, 0], unique_coords[:, 1],
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        return float(f(x_target))

    def plot(self, title: str = "Геометрия задачи", save_path: Optional[str] = None) -> None:
        """
        Визуализация геометрии.

        Args:
            title: Заголовок графика
            save_path: Путь для сохранения изображения (опционально)

        Example:
            >>> geom.plot(title="Профиль NACA 0012", save_path="geometry.png")
        """
        if self.foil_coords is None:
            logger.error("❌ Нет данных для визуализации")
            return

        plt.figure(figsize=(12, 8))

        # Профиль
        plt.plot(self.foil_coords[:, 0], self.foil_coords[:, 1],
                 'b-', linewidth=2, label='Профиль', zorder=3)
        plt.scatter(self.foil_coords[:, 0], self.foil_coords[:, 1],
                    c='blue', s=20, alpha=0.5, zorder=2)

        # Область
        if self.domain_coords is not None:
            plt.plot(self.domain_coords[:, 0], self.domain_coords[:, 1],
                     'ro-', linewidth=2, markersize=8, label='Граница области', zorder=1)
            plt.scatter(self.domain_coords[:, 0], self.domain_coords[:, 1],
                        c='red', s=100, marker='s', alpha=0.7, zorder=2)

        # Аннотации для важных точек
        x_max = np.max(self.foil_coords[:, 0])
        nose_idx = np.argmin(self.foil_coords[:, 0])
        plt.annotate('Носок', (self.foil_coords[nose_idx, 0], self.foil_coords[nose_idx, 1]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10, color='blue')
        plt.annotate('Хвостик', (x_max, self._get_y_at_x(x_max)),
                     xytext=(5, 5), textcoords='offset points', fontsize=10, color='blue')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X, м', fontsize=12)
        plt.ylabel('Y, м', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')

        if self.config.plot_show:
            plt.show()
        else:
            plt.close()

    def save(self, filename: Optional[str] = None, format: Optional[str] = None,
             metadata_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Сохранение геометрии (координаты в метрах).

        domain  → config.output_dir  (workspace/geometry/domains/)
        foil    → metadata_dir        (workspace/geometry/metadata/)
        metadata→ metadata_dir        (workspace/geometry/metadata/)

        Args:
            filename: Базовое имя файла (без расширения)
            format: Формат сохранения ('npy', 'csv', 'txt')
            metadata_dir: Директория для foil и metadata (по умолчанию output_dir)

        Returns:
            Кортеж (путь_к_профилю, путь_к_области)

        Example:
            >>> foil_path, domain_path = geom.save('naca0012', format='npy',
            ...                                      metadata_dir='workspace/geometry/metadata')
        """
        fmt = format or self.config.save_format
        base_name = filename or "airfoil"
        meta_dir = metadata_dir or self.config.output_dir

        foil_path = None
        domain_path = None

        if self.foil_coords is not None:
            foil_path = self._save_coords(self.foil_coords, f"{base_name}_foil", fmt,
                                          directory=meta_dir)

        if self.domain_coords is not None:
            domain_path = self._save_coords(self.domain_coords, f"{base_name}_domain", fmt)

            metadata = {
                'geom_type': 'airfoil',
                'chord': self.chord,
                'is_closed': self.is_closed,
                'num_foil_points': len(self.foil_coords) if self.foil_coords is not None else 0,
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(meta_dir, f"{base_name}_metadata.npy")
            os.makedirs(meta_dir, exist_ok=True)
            np.save(meta_path, metadata)
            if self.config.verbose:
                logger.info(f"✅ Метаданные сохранены: {meta_path}")

        return foil_path, domain_path

    def _save_coords(self, coords: np.ndarray, basename: str, fmt: str,
                     directory: Optional[str] = None) -> str:
        """
        Сохранение координат в указанном формате.

        Args:
            coords: Массив координат формы (N, 2)
            basename: Базовое имя файла
            fmt: Формат файла ('npy', 'csv', 'txt')
            directory: Директория сохранения (по умолчанию config.output_dir)

        Returns:
            Полный путь к сохранённому файлу
        """
        save_dir = directory or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{basename}.{fmt}")

        if fmt == 'npy':
            np.save(filepath, coords)
        elif fmt == 'csv':
            header = "x,y"
            np.savetxt(filepath, coords, delimiter=',', header=header, comments='')
        else:  # txt или другой текстовый
            np.savetxt(filepath, coords, delimiter='\t', fmt='%.8f')

        if self.config.verbose:
            logger.info(f"✅ Сохранено: {filepath}")
        return filepath

    def print_stats(self) -> None:
        """
        Вывод статистики геометрии в консоль.

        Example:
            >>> geom.print_stats()
        """
        print("\n" + "="*50)
        print("СТАТИСТИКА ГЕОМЕТРИИ".center(50))
        print("="*50)

        if self.foil_coords is not None:
            x_min, y_min = np.min(self.foil_coords, axis=0)
            x_max, y_max = np.max(self.foil_coords, axis=0)

            print(f"\n📐 ПРОФИЛЬ:")
            print(f"   Точек: {len(self.foil_coords)}")
            print(f"   Замкнутый: {'✅' if self.is_closed else '❌'}")
            print(f"   X: [{x_min:.4f}, {x_max:.4f}] м, ширина: {x_max-x_min:.4f} м")
            print(f"   Y: [{y_min:.4f}, {y_max:.4f}] м, высота: {y_max-y_min:.4f} м")
            print(f"   Хорда: {self.chord:.4f} м")

        if self.domain_coords is not None:
            print(f"\n🗺️  РАСЧЁТНАЯ ОБЛАСТЬ:")
            print(f"   Точек: {len(self.domain_coords)}")
            print(f"   Замкнутый контур: ✅")
            print(f"   Вершины:")
            for i, (x, y) in enumerate(self.domain_coords):
                print(f"     {i+1}: ({x:.4f}, {y:.4f}) м")

        print("="*50)


# ============================================================================
# УНИВЕРСАЛЬНЫЙ КЛАСС GEOMETRY2D
# ============================================================================

class Geometry2D:
    """
    Класс для работы с геометрией аэродинамических профилей.

    ВСЕ КООРДИНАТЫ В МЕТРАХ.

    Attributes:
        geometry_coords: Координаты точек профиля
        domain_coords: Координаты точек расчётной области
        is_closed: Флаг замкнутости профиля
        config: Конфигурация геометрии
        chord: Хорда профиля [м]
        geom_metadata: Метаданные геометрии из YAML
    """

    def __init__(self, config: Optional[GeometryConfig] = None,
                 geom_type: str = 'airfoil') -> None:
        """
        Args:
            config: Конфигурация геометрии
            geom_type: Должен быть 'airfoil' (junction удалён)
        """
        if geom_type != 'airfoil':
            raise ValueError(f"Поддерживается только geom_type='airfoil', получено: '{geom_type}'")
        self.geom_type = 'airfoil'
        self.config = config or GeometryConfig()

        self.geometry_coords: Optional[np.ndarray] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.geom_metadata: Optional[Dict] = None
        self.is_closed: Optional[bool] = None
        self.chord: float = 1.0

        self._airfoil_geom = AirfoilGeometry(config)

        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.config.verbose:
            logger.info("✅ Инициализирована геометрия профиля")

    def load_from_file(self, filepath: str) -> bool:
        """
        Загружает профиль из явно указанного файла координат.

        Относительные пути разрешаются от корня проекта. После успешной загрузки
        координаты, хорда и признак замкнутости синхронизируются с внутренним
        объектом :class:`AirfoilGeometry`.
        """
        resolved_path = filepath
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(PROJECT_ROOT, resolved_path)

        result = self._airfoil_geom.load_airfoil(resolved_path)
        if result:
            self.geometry_coords = self._airfoil_geom.foil_coords
            self.is_closed = self._airfoil_geom.is_closed
            self.chord = self._airfoil_geom.chord
            self.geom_metadata = {
                "source_type": "file",
                "file_path": os.path.abspath(resolved_path),
            }
        return result

    def load_from_yaml(self, yaml_path: str = None, geom_name: str = None) -> bool:
        """
        Загрузка профиля из YAML-библиотеки foil_geoms.yaml.

        Args:
            yaml_path: Путь к YAML-файлу библиотеки (по умолчанию FOIL_GEOMS_PATH)
            geom_name: Имя профиля в библиотеке

        Returns:
            True если загрузка успешна
        """
        if yaml_path is None:
            yaml_path = FOIL_GEOMS_PATH

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            geom_data = None
            for item in data.get('foils', []):
                if item['name'] == geom_name:
                    geom_data = item
                    break

            if geom_data is None:
                logger.error(f"❌ Профиль '{geom_name}' не найден в {yaml_path}")
                return False

            self.geom_metadata = geom_data
            return self._load_airfoil_from_yaml(geom_data)

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке геометрии: {e}")
            return False

    def _load_airfoil_from_yaml(self, geom_data: Dict) -> bool:
        """Загрузка профиля из YAML-данных."""
        source_type = geom_data.get('source_type', 'inline')

        if source_type == 'file':
            file_path = geom_data.get('file_path')
            file_format = geom_data.get('file_format', 'txt')  # читается из конфига, но не используется напрямую —
            # формат определяется автоматически по расширению файла внутри load_airfoil()

            if file_path is None:
                logger.error("❌ Не указан file_path для source_type='file'")
                return False

            if not os.path.isabs(file_path):
                file_path = os.path.join(PROJECT_ROOT, file_path)

            success = self._airfoil_geom.load_airfoil(file_path)
            if success:
                self.geometry_coords = self._airfoil_geom.foil_coords
                self.is_closed = self._airfoil_geom.is_closed
                self.chord = self._airfoil_geom.chord
            return success

        elif source_type == 'inline':
            X = geom_data.get('X', [])
            Y = geom_data.get('Y', [])

            if not X or not Y:
                logger.error("❌ Не указаны координаты X и/или Y")
                return False

            self._airfoil_geom.foil_coords = np.column_stack((X, Y))

            x_min = np.min(self._airfoil_geom.foil_coords[:, 0])
            x_max = np.max(self._airfoil_geom.foil_coords[:, 0])
            self._airfoil_geom.chord = x_max - x_min
            self._airfoil_geom.is_closed = self._airfoil_geom._check_closed()

            self.geometry_coords = self._airfoil_geom.foil_coords
            self.is_closed = self._airfoil_geom.is_closed
            self.chord = self._airfoil_geom.chord

            if self.config.verbose:
                logger.info(f"✅ Профиль загружен из inline-данных")
                logger.info(f"   Точек: {len(self.geometry_coords)}")
                logger.info(f"   Хорда: {self.chord:.4f} м")

            return True

        return False

    def create_domain(self, **kwargs) -> Optional[np.ndarray]:
        """Создание расчётной области."""
        self.domain_coords = self._airfoil_geom.create_domain(**kwargs)
        return self.domain_coords

    def plot(self, title: str = "Геометрия задачи", save_path: Optional[str] = None) -> None:
        """Визуализация геометрии."""
        self._airfoil_geom.plot(title=title, save_path=save_path)

    def save(self, filename: Optional[str] = None, format: Optional[str] = None,
             metadata_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Сохранение геометрии."""
        return self._airfoil_geom.save(filename=filename, format=format,
                                       metadata_dir=metadata_dir)

    def print_stats(self) -> None:
        """Вывод статистики геометрии."""
        self._airfoil_geom.print_stats()


FOILS_DIR = GEOMETRY_FOILS_DIR


def _select_geometry_file() -> Optional[str]:
    """
    Меню выбора файла профиля из папки workspace/geometry/foils/.

    Сканирует workspace/geometry/foils/ на файлы .txt и .csv с координатами профиля
    и предлагает пользователю выбрать нужный по номеру.

    Returns:
        Полный путь к выбранному файлу или None при отмене.
    """
    allowed_ext = {'.txt', '.csv'}
    os.makedirs(FOILS_DIR, exist_ok=True)

    files: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(FOILS_DIR)):
        if os.path.splitext(fname)[1].lower() in allowed_ext:
            full_path = os.path.join(FOILS_DIR, fname)
            files.append((fname, full_path))

    print("\n" + "=" * 60)
    print("  ВЫБОР ФАЙЛА ПРОФИЛЯ".center(60))
    print("=" * 60)

    if not files:
        print(f"\n  Файлы координат (.txt, .csv) не найдены.")
        print(f"  Поместите файл в папку: {FOILS_DIR}")
        print("=" * 60)
        return None

    print(f"  Папка: workspace/geometry/foils/    Найдено файлов: {len(files)}")
    print("-" * 60)
    for i, (fname, full_path) in enumerate(files, 1):
        size_kb = os.path.getsize(full_path) / 1024
        print(f"  [{i:2d}]  {fname:<40}  {size_kb:6.1f} кБ")
    print("-" * 60)
    print("  [ 0]  Отмена")
    print("=" * 60)

    while True:
        try:
            idx = int(input("\n  Номер файла: ").strip())
            if idx == 0:
                return None
            if 1 <= idx <= len(files):
                fname, full_path = files[idx - 1]
                print(f"\n  Выбран: {fname}")
                return full_path
            print(f"  Введите число от 0 до {len(files)}")
        except ValueError:
            print("  Ошибка: введите число")
        except KeyboardInterrupt:
            print()
            return None


def g2d() -> None:
    """
    Главная функция модуля геометрии.

    Показывает меню выбора файла из workspace/geometry/foils, затем загружает
    профиль, создаёт расчётную область, визуализирует и сохраняет.
    """
    print('\033c')
    print("=" * 60)
    print("МОДУЛЬ ГЕОМЕТРИИ G2D".center(60))
    print("=" * 60)

    # Выбор файла профиля
    selected_file = _select_geometry_file()
    if selected_file is None:
        return

    config = GeometryConfig(
        save_format='npy',
        plot_show=True,
        output_dir=DEFAULT_OUTPUT_DIR,
        verbose=True,
    )

    # Имя геометрии = имя файла без расширения
    geom_name = os.path.splitext(os.path.basename(selected_file))[0]

    print(f"\n  Файл профиля : {os.path.relpath(selected_file, PROJECT_ROOT)}")
    print(f"  Имя геометрии: {geom_name}")

    # Загрузка профиля
    geom = Geometry2D(config=config, geom_type='airfoil')
    if not geom.load_from_file(selected_file):
        input("\n❌ Ошибка загрузки. Нажмите Enter...")
        return

    geom.print_stats()

    print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ")
    print("-" * 40)

    # Размеры области запрашиваются явно в интерактивном режиме. Они больше не
    # берутся из глобального файла и поэтому всегда относятся к выбранному профилю.
    geom.create_domain()

    geom.print_stats()

    print("\n📊 Визуализация геометрии...")
    plot_path = os.path.join(config.output_dir, f"geometry_{geom_name}.png")
    geom.plot(save_path=plot_path)

    print("\n💾 Сохранение...")
    os.makedirs(DEFAULT_METADATA_DIR, exist_ok=True)
    foil_path, domain_path = geom.save(geom_name, metadata_dir=DEFAULT_METADATA_DIR)

    print(f"\n✅ Сохранено:")
    print(f"   Область:     {domain_path}")
    print(f"   Профиль:     {foil_path}")
    print(f"   Метаданные:  {os.path.join(DEFAULT_METADATA_DIR, f'{geom_name}_metadata.npy')}")

    print("\n" + "=" * 60)
    print("✅ РАБОТА МОДУЛЯ ЗАВЕРШЕНА".center(60))
    print("=" * 60)
    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    g2d()
