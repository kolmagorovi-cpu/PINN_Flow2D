#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Универсальный модуль геометрии (G2D) для работы с профилями и тройниками.

ВСЕ КООРДИНАТЫ В МЕТРАХ (без нормализации).

Предоставляет классы для загрузки контура профиля или конфигурации тройника,
создания расчётной области, визуализации и сохранения геометрии.

Автоматическая загрузка из geom_config.yaml:
    Модуль автоматически считывает параметры из geom_config.yaml:
    
    Для профилей (airfoil):
    - source_type: 'file' → загрузка из файла (file_path, file_format)
    - source_type: 'inline' → загрузка из foil_geoms.yaml по имени (geom_name)
    
    Для тройников (junction):
    - source_type: 'config' → загрузка из junction_geoms.yaml по имени (geom_name)
    - source_type: 'file' → загрузка из CAD (на перспективу)
    
    Библиотеки геометрии (фиксированные пути):
    - foil_geoms.yaml: C:/Users/BelchikovDA/projects/dev/proj/pinn-solver/PINN_Flow2D/Airfoil PINN/Geometry/foil_geoms.yaml
    - junction_geoms.yaml: C:/Users/BelchikovDA/projects/dev/proj/pinn-solver/PINN_Flow2D/Airfoil PINN/Geometry/junction_geoms.yaml

Поддерживаемые типы геометрии:
    - 'airfoil': аэродинамические профили (крыло)
    - 'junction': плоские тройники (T-образные, Y-образные)

Источники данных для профилей:
    - Файлы: .npy, .csv, .txt, .dat
    - Inline: координаты непосредственно в YAML-библиотеке

Example:
    # Автоматическая загрузка из geom_config.yaml
    >>> geom = load_geometry_from_config()
    >>> geom.create_domain()
    >>> geom.save('my_geometry')
    
    # Загрузка профиля из файла
    >>> config = GeometryConfig(save_format='npy', plot_show=True)
    >>> geom = Geometry2D(config=config, geom_type='airfoil')
    >>> geom.load_from_yaml(geom_name='NACA0012')  # Используется FOIL_GEOMS_PATH
    >>> geom.create_domain(front=0.5, top=0.3)
    >>> geom.plot(title="NACA 0012")
    >>> geom.save('naca0012')
    
    # Загрузка тройника
    >>> geom = Geometry2D(config=config, geom_type='junction')
    >>> geom.load_from_yaml(geom_name='VMFL010')  # Используется JUNCTION_GEOMS_PATH
    >>> geom.create_domain()
    >>> geom.plot(title="T-образный тройник")

Автор: Колмагоров И.Д.
Версия: 4.0 (объединённая)
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
    JunctionParamsDict,
    JunctionMetadataDict,
)

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
DEFAULT_PLOT_DPI = 300

# Пути к YAML-библиотекам геометрии (относительно директории проекта)
FOIL_GEOMS_PATH = os.path.join(PROJECT_ROOT, "Geometry", "foil_geoms.yaml")
JUNCTION_GEOMS_PATH = os.path.join(PROJECT_ROOT, "Geometry", "junction_geoms.yaml")

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
        y_min = np.min(self.foil_coords[:, 1])
        y_trailing = self._get_y_at_x(x_max)

        if self.config.verbose:
            print(f"\n📐 Параметры области (в метрах):")
            print(f"   Длина перед профилем: {length:.3f} м")
            print(f"   Высота над профилем: {height:.3f} м")
            print(f"   Носок профиля: ({x_min:.4f}, {y_min:.4f})")
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
        x_min = np.min(self.foil_coords[:, 0])
        x_max = np.max(self.foil_coords[:, 0])
        y_min = np.min(self.foil_coords[:, 1])
        plt.annotate('Носок', (x_min, y_min), xytext=(5, 5),
                     textcoords='offset points', fontsize=10, color='blue')
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


class JunctionGeometry:
    """
    Класс для работы с геометрией плоского T-образного тройника (бесконечного по оси Z)
    и его расчетной областью.
    Определяется линейными и угловыми размерами трёх прямоугольных каналов,
    сходящихся в начале координат (0,0).
    ВСЕ КООРДИНАТЫ В МЕТРАХ
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.junction_params: Optional[Dict] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.config = config or GeometryConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_junction(self, geom_config_path: str, junctions_config_path: str = None) -> bool:
        """
        Загружает конфигурацию тройника из YAML файлов.
        
        Ожидаемая структура geom_config.yaml:
            geom_type: 'junction'
            geom_name: 'VMFL010'
        
        Ожидаемая структура junctions_config_path:
            junctions:
              - name: 'VMFL010'
                chanel_width: 1
                down_length: 2
                ...
        
        Args:
            geom_config_path: Путь к geom_config.yaml
            junctions_config_path: Путь к junction_geoms.yaml

        Returns:
            True если загрузка успешна
        """
        try:
            with open(geom_config_path, 'r', encoding='utf-8') as geom_config:
                data = yaml.safe_load(geom_config)

            junction_name = data.get('geom_name', 'VMFL010')
            
            # Путь по умолчанию
            if junctions_config_path is None:
                junctions_config_path = os.path.join(PROJECT_ROOT, "Geometry", "junction_geoms.yaml")

            with open(junctions_config_path, 'r', encoding='utf-8') as junction_config:
                data = yaml.safe_load(junction_config)

            junctions = {junction['name']: junction for junction in data['junctions']}
            self.junction_params = junctions.get(junction_name)
            
            if self.junction_params is None:
                logger.error(f"❌ Тройник '{junction_name}' не найден в библиотеке")
                return False

            if self.config.verbose:
                logger.info(f'✅ Конфигурация загружена. Модель: {junction_name}')

            return True

        except Exception as e:
            logger.error(f'❌ Ошибка при импорте конфигурационного файла: {e}')
            return False

    def load_from_yaml(self, yaml_path: str = None, junction_name: str = None) -> bool:
        """
        Загрузка тройника из YAML-библиотеки.
        
        Args:
            yaml_path: Путь к junction_geoms.yaml (по умолчанию используется фиксированный путь)
            junction_name: Имя тройника в библиотеке
            
        Returns:
            True если загрузка успешна
        """
        # Используем фиксированный путь по умолчанию
        if yaml_path is None:
            yaml_path = JUNCTION_GEOMS_PATH
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            junctions = {j['name']: j for j in data.get('junctions', [])}
            
            if junction_name is None:
                # Если имя не указано, берём первый доступный
                if not junctions:
                    logger.error("❌ В библиотеке нет тройников")
                    return False
                junction_name = list(junctions.keys())[0]
            
            self.junction_params = junctions.get(junction_name)
            
            if self.junction_params is None:
                logger.error(f"❌ Тройник '{junction_name}' не найден в {yaml_path}")
                return False
            
            if self.config.verbose:
                logger.info(f'✅ Тройник загружён: {junction_name}')
                logger.info(f'   Ширина канала: {self.junction_params["chanel_width"]} м')
            
            return True
            
        except Exception as e:
            logger.error(f'❌ Ошибка при загрузке тройника: {e}')
            return False

    def create_domain(self, **kwargs) -> Optional[np.ndarray]:
        """
        Создает замкнутый контур расчетной области.

        Returns:
            Массив координат границ области или None в случае ошибки.
        """
        if self.junction_params is None:
            logger.error('❌ Конфигурация тройника не загружена.')
            return None

        # Характерные размеры
        chanel_width = float(self.junction_params['chanel_width'])
        down_length = float(self.junction_params['down_length'])
        side_length = float(self.junction_params['side_length'])
        top_length = float(self.junction_params['top_length'])

        domain_coords = []

        # Точки расчётной области. Порядок обхода: против часовой
        inlet_1 = np.array([
            -chanel_width / 2,
            -(chanel_width / 2 + down_length)
        ])
        domain_coords.append(inlet_1)

        inlet_2 = np.array([
            chanel_width / 2,
            -(chanel_width / 2 + down_length)
        ])
        domain_coords.append(inlet_2)

        wall_1 = np.array([
            chanel_width / 2,
            -chanel_width / 2
        ])
        domain_coords.append(wall_1)

        outlet_side_1 = np.array([
            chanel_width / 2 + side_length,
            -chanel_width / 2
        ])
        domain_coords.append(outlet_side_1)

        outlet_side_2 = np.array([
            chanel_width / 2 + side_length,
            chanel_width / 2
        ])
        domain_coords.append(outlet_side_2)

        wall_2 = np.array([
            chanel_width / 2,
            chanel_width / 2
        ])
        domain_coords.append(wall_2)

        outlet_top_1 = np.array([
            chanel_width / 2,
            chanel_width / 2 + top_length
        ])
        domain_coords.append(outlet_top_1)

        outlet_top_2 = np.array([
            -chanel_width / 2,
            chanel_width / 2 + top_length
        ])
        domain_coords.append(outlet_top_2)

        # Замыкание
        domain_coords.append(domain_coords[0])

        if self.config.verbose:
            logger.info(f'✅ Расчётная область создана.')
            logger.info(f'   Контур включает {len(domain_coords)} точек.')

        self.domain_coords = np.array(domain_coords)
        return self.domain_coords

    def plot(
            self,
            title: str = 'Геометрия расчетной области',
            save_path: Optional[str] = None
        ) -> None:
        """
        Визуализация геометрии расчётной области.
        
        Args:
            title: Заголовок графика
            save_path: Путь для сохранения изображения
        """
        if self.domain_coords is None:
            logger.error("❌ Нет данных для визуализации")
            return

        plt.figure(figsize=(12, 8))

        # Область
        plt.plot(self.domain_coords[:, 0], self.domain_coords[:, 1],
                 'ro-', linewidth=2, markersize=8, label='Граница области', zorder=1)
        plt.scatter(self.domain_coords[:, 0], self.domain_coords[:, 1],
                    c='red', s=100, marker='s', alpha=0.7, zorder=2)

        # Аннотации каналов
        w = float(self.junction_params['chanel_width'])
        L_down = float(self.junction_params['down_length'])
        L_side = float(self.junction_params['side_length'])
        L_top = float(self.junction_params['top_length'])

        plt.annotate('Вход', (0, -(w/2 + L_down/2)), xytext=(-20, 0),
                     textcoords='offset points', fontsize=10, color='green',
                     ha='right', va='center')
        plt.annotate('Выход 1', (w/2 + L_side/2, 0), xytext=(0, 20),
                     textcoords='offset points', fontsize=10, color='blue')
        plt.annotate('Выход 2', (0, w/2 + L_top/2), xytext=(0, -20),
                     textcoords='offset points', fontsize=10, color='blue')

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

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> Optional[str]:
        '''
        Сохранение геометрии (координаты в метрах)

        Args:
            filename: базовое имя файла (без расширения)
            format: формат сохранения ('npy', 'csv', 'txt')

        Returns:
            Строку (путь к файлу расчетной области)
        '''
        fmt = format or self.config.save_format
        base_name = filename or 'junction'

        domain_path = None

        if self.domain_coords is not None:
            domain_path = self._save_coords(self.domain_coords, f'{base_name}_domain', fmt)

            # Сохраняем метаданные
            metadata = {
                'geom_type': 'junction',
                'junction_params': self.junction_params,
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(self.config.output_dir, f'{base_name}_metadata.npy')
            np.save(meta_path, metadata)
            if self.config.verbose:
                logger.info(f'✅ Метаданные сохранены: {meta_path}')

        return domain_path

    def _save_coords(self, coords: np.ndarray, basename: str, fmt: str) -> str:
        '''Сохранение координат в указанном формате'''
        filepath = os.path.join(self.config.output_dir, f"{basename}.{fmt}")

        if fmt == 'npy':
            np.save(filepath, coords)
        elif fmt == 'csv':
            header = 'x,y'
            np.savetxt(filepath, coords, delimiter=',', header=header, comments='')
        else:  # txt или другой текстовый
            np.savetxt(filepath, coords, delimiter='\t', fmt='%.8f')

        if self.config.verbose:
            logger.info(f'✅ Сохранено: {filepath}')
        return filepath

    def print_stats(self) -> None:
        '''Вывод информации о контуре.'''
        if self.junction_params is not None:
            print(f'\n📐 ТРОЙНИК:')
            print(f'   Название: {self.junction_params.get("name", "unknown")}')
            print(f'   Ширина канала: {self.junction_params["chanel_width"]} м')
            print(f'   Длины каналов:')
            print(f'     Вход (down): {self.junction_params["down_length"]} м')
            print(f'     Выход 1 (side): {self.junction_params["side_length"]} м')
            print(f'     Выход 2 (top): {self.junction_params["top_length"]} м')
            print(f'   Углы:')
            print(f'     top2side: {self.junction_params["top2side_angl"]}°')
            print(f'     side2down: {self.junction_params["side2down_angl"]}°')

        if self.domain_coords is not None:
            print(f'\n🗺️  РАСЧЁТНАЯ ОБЛАСТЬ:')
            print(f'   Точек: {len(self.domain_coords)}')
            print(f'   Вершины:')
            for i, (x, y) in enumerate(self.domain_coords):
                print(f'     {i+1}: ({x:.4f}, {y:.4f}) м')


# ============================================================================
# УНИВЕРСАЛЬНЫЙ КЛАСС GEOMETRY2D
# ============================================================================

class Geometry2D:
    """
    Универсальный класс для работы с геометрией профилей и тройников.

    ВСЕ КООРДИНАТЫ В МЕТРАХ.

    Поддерживаемые типы геометрии:
        - 'airfoil': аэродинамические профили
        - 'junction': плоские тройники

    Attributes:
        geom_type: Тип геометрии ('airfoil' или 'junction')
        geometry_coords: Координаты геометрии (профиль или параметры тройника)
        domain_coords: Координаты точек расчётной области
        is_closed: Флаг замкнутости профиля (только для airfoil)
        config: Конфигурация геометрии
        chord: Хорда профиля [м] (только для airfoil)
        junction_params: Параметры тройника (только для junction)
        geom_metadata: Метаданные геометрии из YAML
    """

    def __init__(self, config: Optional[GeometryConfig] = None,
                 geom_type: str = 'airfoil') -> None:
        """
        Инициализирует геометрию.

        Args:
            config: Конфигурация геометрии
            geom_type: Тип геометрии ('airfoil' или 'junction')
        """
        self.geom_type = geom_type
        self.config = config or GeometryConfig()
        
        # Общие атрибуты
        self.geometry_coords: Optional[np.ndarray] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.geom_metadata: Optional[Dict] = None
        
        # Специфичные атрибуты для профиля
        self.is_closed: Optional[bool] = None
        self.chord: float = 1.0
        
        # Специфичные атрибуты для тройника
        self.junction_params: Optional[Dict] = None
        
        # Внутренние объекты
        self._airfoil_geom = AirfoilGeometry(config) if geom_type == 'airfoil' else None
        self._junction_geom = JunctionGeometry(config) if geom_type == 'junction' else None
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.config.verbose:
            logger.info(f"✅ Инициализирована геометрия: {self.geom_type}")

    def load_from_yaml(self, yaml_path: str = None, geom_name: str = None) -> bool:
        """
        Загрузка геометрии из YAML-библиотеки.
        Используется для загрузки из foil_geoms.yaml или junction_geoms.yaml.

        Args:
            yaml_path: Путь к YAML-файлу библиотеки (по умолчанию используется фиксированный путь)
            geom_name: Имя геометрии в библиотеке

        Returns:
            True если загрузка успешна
        """
        # Используем фиксированные пути по умолчанию
        if yaml_path is None:
            if self.geom_type == 'airfoil':
                yaml_path = FOIL_GEOMS_PATH
            elif self.geom_type == 'junction':
                yaml_path = JUNCTION_GEOMS_PATH
            else:
                logger.error(f"❌ Неизвестный тип геометрии: {self.geom_type}")
                return False

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if self.geom_type == 'airfoil':
                key = 'foils'
            elif self.geom_type == 'junction':
                key = 'junctions'
            else:
                logger.error(f"❌ Неизвестный тип геометрии: {self.geom_type}")
                return False

            # Поиск геометрии по имени
            geom_data = None
            for item in data.get(key, []):
                if item['name'] == geom_name:
                    geom_data = item
                    break

            if geom_data is None:
                logger.error(f"❌ Геометрия '{geom_name}' не найдена в {yaml_path}")
                return False

            self.geom_metadata = geom_data

            if self.geom_type == 'airfoil':
                return self._load_airfoil_from_yaml(geom_data)
            elif self.geom_type == 'junction':
                return self._load_junction_from_yaml(geom_data)

            return False

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке геометрии: {e}")
            return False

    def load_from_config(self, config_path: str = None) -> bool:
        """
        Загрузка геометрии напрямую из geom_config.yaml.
        
        Использует параметры airfoil_params или junction_params из конфига:
        - source_type: 'file' или 'inline'
        - file_path: путь к файлу профиля
        - file_format: формат файла
        
        Args:
            config_path: Путь к geom_config.yaml (по умолчанию ищется в PROJECT_ROOT)
            
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
            
            if self.geom_type == 'airfoil':
                return self._load_airfoil_from_config(geom_config)
            elif self.geom_type == 'junction':
                return self._load_junction_from_config(geom_config)
            else:
                logger.error(f"❌ Неизвестный тип геометрии: {self.geom_type}")
                return False
                
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
            file_format = airfoil_params.get('file_format', 'txt')
            
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

    def _load_junction_from_config(self, geom_config: Dict) -> bool:
        """
        Загрузка тройника из параметров geom_config.yaml.
        
        Args:
            geom_config: Словарь конфигурации
            
        Returns:
            True если загрузка успешна
        """
        junction_params = geom_config.get('junction_params', {})
        geom_name = geom_config.get('geom_name')
        source_type = junction_params.get('source_type', 'config')
        
        if source_type == 'config':
            # Загрузка из junction_geoms.yaml по имени
            if geom_name is None:
                logger.error("❌ Не указан geom_name для тройника")
                return False
            
            if self.config.verbose:
                logger.info(f"📂 Загрузка тройника '{geom_name}' из {JUNCTION_GEOMS_PATH}")
            
            return self.load_from_yaml(yaml_path=JUNCTION_GEOMS_PATH, geom_name=geom_name)
        
        elif source_type == 'file':
            logger.warning("⚠️ Загрузка тройника из файла пока не поддерживается")
            return False
        
        else:
            logger.error(f"❌ Неизвестный source_type: {source_type}")
            return False

    def _load_airfoil_from_yaml(self, geom_data: Dict) -> bool:
        """Загрузка профиля из YAML-данных."""
        source_type = geom_data.get('source_type', 'inline')
        
        if source_type == 'file':
            file_path = geom_data.get('file_path')
            file_format = geom_data.get('file_format', 'txt')
            
            if file_path is None:
                logger.error("❌ Не указан file_path для source_type='file'")
                return False
            
            return self._airfoil_geom.load_airfoil(file_path)
        
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

    def _load_junction_from_yaml(self, geom_data: Dict) -> bool:
        """Загрузка тройника из YAML-данных."""
        self._junction_geom.junction_params = geom_data
        self.junction_params = geom_data
        
        if self.config.verbose:
            logger.info(f"✅ Тройник загружён: {geom_data.get('name', 'unknown')}")
            logger.info(f"   Ширина канала: {geom_data['chanel_width']} м")
        
        return True

    def create_domain(self, **kwargs) -> Optional[np.ndarray]:
        """Создание расчётной области."""
        if self.geom_type == 'airfoil':
            self.domain_coords = self._airfoil_geom.create_domain(**kwargs)
            return self.domain_coords
        elif self.geom_type == 'junction':
            self.domain_coords = self._junction_geom.create_domain()
            return self.domain_coords
        return None

    def plot(self, title: str = "Геометрия задачи", save_path: Optional[str] = None) -> None:
        """Визуализация геометрии."""
        if self.geom_type == 'airfoil':
            self._airfoil_geom.plot(title=title, save_path=save_path)
        elif self.geom_type == 'junction':
            self._junction_geom.plot(title=title, save_path=save_path)

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Сохранение геометрии."""
        if self.geom_type == 'airfoil':
            return self._airfoil_geom.save(filename=filename, format=format)
        elif self.geom_type == 'junction':
            domain_path = self._junction_geom.save(filename=filename, format=format)
            return None, domain_path
        return None, None

    def print_stats(self) -> None:
        """Вывод статистики геометрии."""
        if self.geom_type == 'airfoil':
            self._airfoil_geom.print_stats()
        elif self.geom_type == 'junction':
            self._junction_geom.print_stats()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def load_geometry_from_config(config_path: str = None, verbose: bool = True) -> Geometry2D:
    """
    Автоматическая загрузка геометрии из конфигурационного файла.
    
    Args:
        config_path: Путь к geom_config.yaml (по умолчанию ищется в PROJECT_ROOT)
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
    
    geom_type = geom_config.get('geom_type', 'airfoil')
    geom_name = geom_config.get('geom_name')
    
    if geom_name is None:
        raise ValueError("В geom_config.yaml не указан параметр 'geom_name'")
    
    config = GeometryConfig(verbose=verbose)
    geom = Geometry2D(config=config, geom_type=geom_type)
    
    # Загрузка из geom_config.yaml (использует airfoil_params/junction_params)
    if not geom.load_from_config(config_path):
        raise RuntimeError(f"Не удалось загрузить геометрию '{geom_name}'")
    
    if verbose:
        logger.info(f"✅ Геометрия загружена: {geom_type}/{geom_name}")
    
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

    # Инициализация геометрии
    geom = Geometry2D(config=config, geom_type=geom_type)

    if geom_type == 'airfoil':
        # Загрузка из geom_config.yaml (использует airfoil_params)
        if not geom.load_from_config(geom_config_path):
            input("\n❌ Ошибка загрузки. Нажмите Enter...")
            return

        geom.print_stats()

        print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ")
        print("-" * 40)

        # Загрузка параметров области из конфига
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

    elif geom_type == 'junction':
        # Загрузка из geom_config.yaml (использует junction_params)
        if not geom.load_from_config(geom_config_path):
            input("\n❌ Ошибка загрузки. Нажмите Enter...")
            return

        geom.print_stats()

        print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ")
        print("-" * 40)
        geom.create_domain()

        geom.print_stats()

        print("\n📊 Визуализация геометрии...")
        plot_path = os.path.join(config.output_dir, f"geometry_{geom_name}.png")
        geom.plot(save_path=plot_path)

        print("\n💾 Сохранение...")
        _, domain_path = geom.save(f"junction_{geom_name}")

        print(f"\n✅ Сохранено:")
        print(f"   Область: {domain_path}")
        print(f"   Метаданные: {os.path.join(config.output_dir, f'junction_{geom_name}_metadata.npy')}")

    print("\n" + "="*60)
    print("✅ РАБОТА МОДУЛЯ ЗАВЕРШЕНА")
    print("="*60)
    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    g2d()
