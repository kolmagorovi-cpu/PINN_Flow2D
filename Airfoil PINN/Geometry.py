#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Модуль геометрии (G2D) для работы с аэродинамическими профилями.

ВСЕ КООРДИНАТЫ В МЕТРАХ (без нормализации).

Автоматическая загрузка из geom_config.yaml:
    - source_type: 'file' → загрузка из файла (file_path, file_format)
    - source_type: 'inline' → загрузка из foil_geoms.yaml по имени (geom_name)

Example:
    >>> geom = load_geometry_from_config()
    >>> geom.create_domain()
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

from type_hints import (
    GeometryConfigDict,
    DomainParamsDict,
    ClosedDomainParamsDict,
    GeometryMetadataDict,
)

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
DEFAULT_PLOT_DPI = 300

# Путь к YAML-библиотеке профилей
FOIL_GEOMS_PATH = os.path.join(PROJECT_ROOT, "Geometry", "foil_geoms.yaml")

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

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Сохранение геометрии (координаты в метрах).

        Args:
            filename: Базовое имя файла (без расширения)
            format: Формат сохранения ('npy', 'csv', 'txt')

        Returns:
            Кортеж (путь_к_профилю, путь_к_области)

        Example:
            >>> foil_path, domain_path = geom.save('naca0012', format='npy')
        """
        fmt = format or self.config.save_format
        base_name = filename or "airfoil"

        foil_path = None
        domain_path = None

        if self.foil_coords is not None:
            foil_path = self._save_coords(self.foil_coords, f"{base_name}_foil", fmt)

        if self.domain_coords is not None:
            domain_path = self._save_coords(self.domain_coords, f"{base_name}_domain", fmt)

            # Сохраняем метаданные
            metadata = {
                'geom_type': 'airfoil',
                'chord': self.chord,
                'is_closed': self.is_closed,
                'num_foil_points': len(self.foil_coords) if self.foil_coords is not None else 0,
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(self.config.output_dir, f"{base_name}_metadata.npy")
            np.save(meta_path, metadata)
            if self.config.verbose:
                logger.info(f"✅ Метаданные сохранены: {meta_path}")

        return foil_path, domain_path

    def _save_coords(self, coords: np.ndarray, basename: str, fmt: str) -> str:
        """
        Сохранение координат в указанном формате.

        Args:
            coords: Массив координат формы (N, 2)
            basename: Базовое имя файла
            fmt: Формат файла ('npy', 'csv', 'txt')

        Returns:
            Полный путь к сохранённому файлу
        """
        filepath = os.path.join(self.config.output_dir, f"{basename}.{fmt}")

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

    def load_from_config(self, config_path: str = None) -> bool:
        """
        Загрузка профиля напрямую из geom_config.yaml.

        Args:
            config_path: Путь к geom_config.yaml (по умолчанию PROJECT_ROOT/geom_config.yaml)

        Returns:
            True если загрузка успешна
        """
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "geom_config.yaml")

        if not os.path.exists(config_path):
            logger.error(f"❌ Конфигурационный файл не найден: {config_path}")
            return False

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                geom_config = yaml.safe_load(f)
            return self._load_airfoil_from_config(geom_config)
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке конфигурации: {e}")
            return False

    def _load_airfoil_from_config(self, geom_config: Dict) -> bool:
        """
        Загрузка профиля из параметров geom_config.yaml.
        
        Args:
            geom_config: Словарь конфигурации
            
        Returns:
            True если загрузка успешна
        """
        airfoil_params = geom_config.get('airfoil_params', {})
        source_type = airfoil_params.get('source_type', 'inline')
        geom_name = geom_config.get('geom_name')
        
        if source_type == 'file':
            file_path = airfoil_params.get('file_path')
            file_format = airfoil_params.get('file_format', 'txt')  # читается из конфига, но не используется напрямую —
            # формат определяется автоматически по расширению файла внутри load_airfoil()

            if file_path is None:
                logger.error("❌ Не указан file_path в airfoil_params")
                return False

            # Если путь относительный, добавляем PROJECT_ROOT
            if not os.path.isabs(file_path):
                file_path = os.path.join(PROJECT_ROOT, file_path)

            if self.config.verbose:
                logger.info(f"📂 Загрузка профиля из файла: {file_path}")

            result = self._airfoil_geom.load_airfoil(file_path)
            if result:
                self.geometry_coords = self._airfoil_geom.foil_coords
                self.is_closed = self._airfoil_geom.is_closed
                self.chord = self._airfoil_geom.chord
            return result
        
        elif source_type == 'inline':
            # Загрузка из foil_geoms.yaml по имени
            if geom_name is None:
                logger.error("❌ Не указан geom_name для inline профиля")
                return False
            
            if self.config.verbose:
                logger.info(f"📂 Загрузка inline-профиля '{geom_name}' из {FOIL_GEOMS_PATH}")
            
            return self.load_from_yaml(yaml_path=FOIL_GEOMS_PATH, geom_name=geom_name)
        
        else:
            logger.error(f"❌ Неизвестный source_type: {source_type}")
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

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Сохранение геометрии."""
        return self._airfoil_geom.save(filename=filename, format=format)

    def print_stats(self) -> None:
        """Вывод статистики геометрии."""
        self._airfoil_geom.print_stats()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def load_geometry_from_config(config_path: str = None, verbose: bool = True) -> Geometry2D:
    """
    Автоматическая загрузка профиля из geom_config.yaml.

    Args:
        config_path: Путь к geom_config.yaml (по умолчанию PROJECT_ROOT/geom_config.yaml)
        verbose: Выводить ли информацию о процессе

    Returns:
        Geometry2D: Инициализированный объект геометрии

    Example:
        >>> geom = load_geometry_from_config()
        >>> geom.create_domain()
        >>> geom.save('my_geometry')
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "geom_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        geom_config = yaml.safe_load(f)

    geom_name = geom_config.get('geom_name')
    if geom_name is None:
        raise ValueError("В geom_config.yaml не указан параметр 'geom_name'")

    config = GeometryConfig(verbose=verbose)
    geom = Geometry2D(config=config, geom_type='airfoil')

    if not geom.load_from_config(config_path):
        raise RuntimeError(f"Не удалось загрузить геометрию '{geom_name}'")

    if verbose:
        logger.info(f"✅ Геометрия загружена: airfoil/{geom_name}")

    return geom


def g2d() -> None:
    """
    Главная функция модуля геометрии (автоматический режим).

    Автоматически загружает конфигурацию из geom_config.yaml,
    создаёт геометрию, визуализирует и сохраняет результаты.
    """
    print('\033c')
    print("="*60)
    print("МОДУЛЬ ГЕОМЕТРИИ G2D".center(60))
    print("="*60)

    config = GeometryConfig(
        save_format='npy',
        plot_show=True,
        output_dir=DEFAULT_OUTPUT_DIR,
        verbose=True
    )

    # Загрузка конфигурации из geom_config.yaml
    geom_config_path = os.path.join(PROJECT_ROOT, "geom_config.yaml")
    
    if not os.path.exists(geom_config_path):
        logger.error(f"❌ Конфигурационный файл не найден: {geom_config_path}")
        input("\nНажмите Enter для выхода...")
        return

    try:
        with open(geom_config_path, 'r', encoding='utf-8') as f:
            geom_config = yaml.safe_load(f)
        
        geom_type = geom_config.get('geom_type', 'airfoil')
        geom_name = geom_config.get('geom_name')
        
        if geom_name is None:
            logger.error("❌ В geom_config.yaml не указан параметр 'geom_name'")
            input("\nНажмите Enter для выхода...")
            return
        
        print(f"\n✅ Конфигурация загружена: {geom_config_path}")
        print(f"   Тип геометрии: {geom_type}")
        print(f"   Имя геометрии: {geom_name}")

    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке конфигурации: {e}")
        input("\nНажмите Enter для выхода...")
        return

    if geom_type != 'airfoil':
        logger.error(f"❌ Поддерживается только geom_type='airfoil', получено: '{geom_type}'")
        input("\nНажмите Enter для выхода...")
        return

    # Инициализация геометрии
    geom = Geometry2D(config=config, geom_type='airfoil')

    if not geom.load_from_config(geom_config_path):
        input("\n❌ Ошибка загрузки. Нажмите Enter...")
        return

    geom.print_stats()

    print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ")
    print("-" * 40)

    airfoil_params = geom_config.get('airfoil_params', {})
    domain_params = airfoil_params.get('domain_params', {})
    geom.create_domain(**domain_params)

    geom.print_stats()

    print("\n📊 Визуализация геометрии...")
    plot_path = os.path.join(config.output_dir, f"geometry_{geom_name}.png")
    geom.plot(save_path=plot_path)

    print("\n💾 Сохранение...")
    foil_path, domain_path = geom.save(geom_name)

    print(f"\n✅ Сохранено:")
    print(f"   Профиль: {foil_path}")
    print(f"   Область: {domain_path}")
    print(f"   Метаданные: {os.path.join(config.output_dir, f'{geom_name}_metadata.npy')}")

    print("\n" + "="*60)
    print("✅ РАБОТА МОДУЛЯ ЗАВЕРШЕНА")
    print("="*60)
    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    g2d()
