#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль геометрии (G2D) для работы с профилями и расчётными областями.

ВСЕ КООРДИНАТЫ В МЕТРАХ (без нормализации).

Предоставляет классы для загрузки контура профиля, создания расчётной области,
визуализации и сохранения геометрии.

Example:
    >>> config = GeometryConfig(save_format='npy', plot_show=True)
    >>> geom = AirfoilGeometry(config)
    >>> geom.load_airfoil('naca0012.dat')
    >>> geom.create_domain(front=0.5, top=0.3)
    >>> geom.plot()
    >>> geom.save('naca0012')

Автор: Колмагоров И.Д.
Версия: 3.260226 (финальная исправленная)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
import logging
from scipy import interpolate

from type_hints import GeometryConfigDict, DomainParamsDict, ClosedDomainParamsDict, GeometryMetadataDict

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
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
    """
    save_format: str = 'npy'
    plot_enabled: bool = True
    plot_show: bool = True
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: str = DEFAULT_OUTPUT_DIR

    def to_dict(self) -> GeometryConfigDict:
        """Преобразует конфигурацию в словарь."""
        return GeometryConfigDict(
            save_format=self.save_format,
            plot_enabled=self.plot_enabled,
            plot_show=self.plot_show,
            plot_dpi=self.plot_dpi,
            output_dir=self.output_dir
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
            filepath: Путь к файлу с координатами (.npy, .csv, .txt)

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
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()  # разбивает по любым пробельным символам
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
            logger.info(f"✅ Замкнутая область создана: {len(self.domain_coords)} вершин")
        else:
            self.domain_coords = self._create_unclosed_domain(kwargs)
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
        x_min = np.min(self.foil_coords[:, 0])          # Носок
        x_max = np.max(self.foil_coords[:, 0])          # Хвостик
        y_min = np.min(self.foil_coords[:, 1])          # Нижняя точка профиля
        y_trailing = self._get_y_at_x(x_max)            # y на хвостике

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

        # 4. Вниз до нижней границы (уровень носка)
        domain_points.append([-length, y_min])

        # 5. Вправо до носка профиля (замыкаем контур)
        domain_points.append([x_min, y_min])

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
        # Сортируем точки по x для интерполяции (это безопасно, так как интерполяция требует монотонности)
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

        # Область (всегда ЗАМКНУТЫЙ контур)
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
                'chord': self.chord,
                'is_closed': self.is_closed,
                'num_foil_points': len(self.foil_coords) if self.foil_coords is not None else 0,
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(self.config.output_dir, f"{base_name}_metadata.npy")
            np.save(meta_path, metadata)
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

        logger.info(f"✅ Сохранено: {filepath}")
        return filepath

    def print_stats(self) -> None:
        """
        Вывод статистики геометрии в консоль.

        Example:
            >>> geom.print_stats()
        """
        print("\n" + "="*50)
        print("СТАТИСТИКА ГЕОМЕТРИИ")
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


def g2d() -> None:
    """
    Главная функция модуля геометрии (интерактивный режим).

    Запускает интерактивный сеанс для загрузки профиля, создания области,
    визуализации и сохранения геометрии.

    Example:
        >>> if __name__ == "__main__":
        ...     g2d()
    """
    print('\033c')
    print("="*60)
    print("МОДУЛЬ ГЕОМЕТРИИ G2D".center(60))
    print("="*60)

    config = GeometryConfig(
        save_format='npy',
        plot_show=True,
        output_dir=DEFAULT_OUTPUT_DIR
    )

    geom = AirfoilGeometry(config)

    print("\n📂 Укажите путь к файлу контура")
    path = input("Путь к файлу: ").strip()

    if not geom.load_airfoil(path):
        input("\n❌ Ошибка загрузки. Нажмите Enter...")
        return

    geom.print_stats()

    print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ")
    print("-" * 40)
    geom.create_domain()

    geom.print_stats()

    print("\n📊 Визуализация геометрии...")
    plot_path = os.path.join(config.output_dir, "geometry.png")
    geom.plot(save_path=plot_path)

    print("\n💾 Сохранение...")
    foil_path, domain_path = geom.save("airfoil")

    print(f"\n✅ Сохранено:")
    print(f"   Профиль: {foil_path}")
    print(f"   Область: {domain_path}")
    print(f"   Метаданные: {os.path.join(config.output_dir, 'airfoil_metadata.npy')}")

    print("\n" + "="*60)
    input("Нажмите Enter для возврата...")


if __name__ == "__main__":
    g2d()