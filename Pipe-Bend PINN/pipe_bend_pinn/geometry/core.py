from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль геометрии Pipe-Bend PINN.

Содержит класс PipeBendGeometry для параметрического построения
расчётной области отвода 90°.

ВСЕ КООРДИНАТЫ В МЕТРАХ.

Контур отвода:
- 8 ключевых вершин, соединённых прямыми и дугами
- Внешняя дуга: radius = bend_radius + pipe_width/2
- Внутренняя дуга: radius = bend_radius - pipe_width/2
- Дуги аппроксимируются N точками (N >= 50)

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
import logging

from pipe_bend_pinn.core.type_hints import GeometryConfigDict, PipeBendParamsDict, GeometryMetadataDict
DEFAULT_OUTPUT_DIR = GEOMETRY_DOMAINS_DIR
DEFAULT_PLOT_DPI = 300

# Фиксированный путь к библиотеке отводов
PIPE_BEND_GEOMS_PATH = GEOMS_LIB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GeometryConfig:
    """
    Конфигурация для геометрических параметров.
    """
    save_format: str = 'npy'
    plot_enabled: bool = True
    plot_show: bool = True
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: str = DEFAULT_OUTPUT_DIR
    verbose: bool = True

    def to_dict(self) -> GeometryConfigDict:
        return dict(
            save_format=self.save_format,
            plot_enabled=self.plot_enabled,
            plot_show=self.plot_show,
            plot_dpi=self.plot_dpi,
            output_dir=self.output_dir,
            verbose=self.verbose
        )


class PipeBendGeometry:
    """
    Класс для работы с геометрией отвода 90°.

    Attributes:
        pipe_bend_params: Параметры отвода
        domain_coords: Координаты точек расчётной области (N, 2)
        config: Конфигурация геометрии
        pipe_width: Ширина канала [м]
        inlet_length: Длина входного участка [м]
        outlet_length: Длина выходного участка [м]
        bend_radius: Радиус изгиба по оси [м]
        bend_angle: Угол изгиба [°]
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.pipe_bend_params: Optional[Dict] = None
        self.domain_coords: Optional[np.ndarray] = None
        self.config = config or GeometryConfig()
        
        self.pipe_width: float = 0.5
        self.inlet_length: float = 2.0
        self.outlet_length: float = 2.0
        self.bend_radius: float = 1.0
        self.bend_angle: float = 90.0
        
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_from_yaml(self, yaml_path: str = None, bend_name: str = None) -> bool:
        """
        Загрузка конфигурации отвода из YAML-библиотеки.

        Args:
            yaml_path: Путь к pipe_bend_geoms.yaml
            bend_name: Имя отвода в библиотеке

        Returns:
            True если загрузка успешна
        """
        if yaml_path is None:
            yaml_path = PIPE_BEND_GEOMS_PATH
        
        if not os.path.exists(yaml_path):
            logger.error(f"Файл библиотеки отводов не найден: {yaml_path}")
            return False

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            bends = {b['name']: b for b in data.get('pipe_bends', [])}
            
            if bend_name is None:
                if not bends:
                    logger.error("В библиотеке нет отводов")
                    return False
                bend_name = list(bends.keys())[0]

            self.pipe_bend_params = bends.get(bend_name)
            if self.pipe_bend_params is None:
                logger.error(f"Отвод '{bend_name}' не найден в {yaml_path}")
                return False

            # Извлекаем параметры
            self.inlet_length = float(self.pipe_bend_params.get('inlet_length', 2.0))
            self.outlet_length = float(self.pipe_bend_params.get('outlet_length', 2.0))
            self.bend_radius = float(self.pipe_bend_params.get('bend_radius', 1.0))
            self.pipe_width = float(self.pipe_bend_params.get('pipe_width', 0.5))
            self.bend_angle = float(self.pipe_bend_params.get('bend_angle', 90.0))

            if self.config.verbose:
                logger.info(f"Отвод загружён: {bend_name}")
                logger.info(f"  Ширина канала: {self.pipe_width} м")
                logger.info(f"  Радиус изгиба: {self.bend_radius} м")
                logger.info(f"  Угол изгиба: {self.bend_angle}°")
                logger.info(f"  Вход: {self.inlet_length} м, Выход: {self.outlet_length} м")
            
            return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке отвода: {e}")
            return False

    def load_from_config(self, config_path: str = None) -> bool:
        """
        Загрузка конфигурации отвода из geom_config.yaml.

        Args:
            config_path: Путь к geom_config.yaml

        Returns:
            True если загрузка успешна
        """
        if config_path is None:
            config_path = GEOM_CONFIG_PATH
        
        if not os.path.exists(config_path):
            logger.error(f"Конфигурационный файл не найден: {config_path}")
            return False

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            pipe_bend_params = data.get('pipe_bend_params', {})
            source_type = pipe_bend_params.get('source_type', 'config')
            geom_name = data.get('geom_name')

            if source_type == 'file':
                logger.warning("Загрузка отвода из файла пока не поддерживается")
                return False
            
            # Загрузка из библиотеки
            if geom_name:
                return self.load_from_yaml(PIPE_BEND_GEOMS_PATH, geom_name)
            else:
                # Прямое задание параметров
                self.inlet_length = float(pipe_bend_params.get('inlet_length', 2.0))
                self.outlet_length = float(pipe_bend_params.get('outlet_length', 2.0))
                self.bend_radius = float(pipe_bend_params.get('bend_radius', 1.0))
                self.pipe_width = float(pipe_bend_params.get('pipe_width', 0.5))
                self.bend_angle = float(pipe_bend_params.get('bend_angle', 90.0))
                
                self.pipe_bend_params = {
                    'name': geom_name or 'custom',
                    'inlet_length': self.inlet_length,
                    'outlet_length': self.outlet_length,
                    'bend_radius': self.bend_radius,
                    'pipe_width': self.pipe_width,
                    'bend_angle': self.bend_angle
                }
                
                if self.config.verbose:
                    logger.info("Параметры отвода загружены из geom_config.yaml")
                return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            return False

    def _generate_arc(self, cx: float, cy: float, radius: float, 
                      start_angle_deg: float, end_angle_deg: float,
                      n_points: int = 50) -> np.ndarray:
        """
        Генерирует точки дуги окружности.

        Args:
            cx, cy: Центр окружности
            radius: Радиус дуги
            start_angle_deg, end_angle_deg: Начальный и конечный угол [°]
            n_points: Количество точек

        Returns:
            Массив точек формы (n_points, 2)
        """
        angles = np.linspace(np.radians(start_angle_deg), 
                            np.radians(end_angle_deg), n_points)
        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)
        return np.column_stack((x, y))

    def create_domain(self, **kwargs) -> Optional[np.ndarray]:
        """
        Создаёт замкнутый контур расчётной области отвода.

        Геометрия 90° отвода направо с внешней и внутренней дугами.

        Returns:
            Массив координат замкнутого контура (N, 2) или None
        """
        R = self.bend_radius
        W = self.pipe_width
        L_in = self.inlet_length
        L_out = self.outlet_length
        
        R_outer = R + W / 2  # Радиус внешней дуги
        R_inner = R - W / 2  # Радиус внутренней дуги
        
        # Проверка: внутренний радиус должен быть > 0
        if R_inner <= 0:
            logger.error(f"Внутренний радиус <= 0! Увеличьте bend_radius или уменьшите pipe_width.")
            logger.error(f"  R={R}, W={W}, R_inner={R_inner}")
            return None

        # Центр дуг находится в точке (R + L_out? нет, просто в начале поворота)
        # Для отвода 90° направо:
        # - Поток входит слева (x отрицательное), идёт по x
        # - Поток выходит вверх (y положительное)
        # - Дуга начинается в точке (0, -R) и идёт до (R, 0)
        
        # Строим контур по часовой стрелке для замкнутого многоугольника
        points = []
        
        # Точка 0: левый нижний угол входа
        points.append([-L_in, -R_outer])
        
        # Точка 1: левый верхний угол входа  
        points.append([-L_in, -R_inner])
        
        # Точка 2: начало внутренней дуги
        points.append([0.0, -R_inner])
        
        # Внутренняя дуга: от (0, -R_inner) вверх до (R_inner, 0)
        # Углы: от 0° (направление вправо) до 90° (направление вверх)
        # В стандартной математике: от -90° до 0°
        inner_arc = self._generate_arc(0, 0, R_inner, -90, 0, n_points=50)
        # Первая точка дуги уже добавлена, пропускаем её
        for pt in inner_arc[1:]:
            points.append(pt.tolist())
        
        # Точка 6: конец внутренней дуги = (R_inner, 0)
        # Точка 7: правый нижний угол выхода
        points.append([R_inner, L_out])
        
        # Точка 8: выход
        points.append([R_outer, L_out])
        
        # Точка 9: правый верхний угол выхода
        points.append([R_outer, 0.0])
        
        # Внешняя дуга: от (R_outer, 0) вниз до (0, -R_outer)
        # Углы: от 0° до -90°
        outer_arc = self._generate_arc(0, 0, R_outer, 0, -90, n_points=50)
        for pt in outer_arc[1:]:
            points.append(pt.tolist())
        
        # Замыкаем контур обратно к точке 0
        # Точка после дуги = (0, -R_outer), возвращаемся к (-L_in, -R_outer)
        points.append([-L_in, -R_outer])

        self.domain_coords = np.array(points)
        
        if self.config.verbose:
            logger.info(f"Контур отвода создан: {len(self.domain_coords)} точек")
            x_min, y_min = np.min(self.domain_coords, axis=0)
            x_max, y_max = np.max(self.domain_coords, axis=0)
            logger.info(f"  X: [{x_min:.3f}, {x_max:.3f}] м")
            logger.info(f"  Y: [{y_min:.3f}, {y_max:.3f}] м")
        
        return self.domain_coords

    def plot(self, title: str = "Геометрия отвода 90°", save_path: Optional[str] = None) -> None:
        """
        Визуализация геометрии отвода.

        Args:
            title: Заголовок графика
            save_path: Путь для сохранения
        """
        if self.domain_coords is None:
            logger.error("Нет данных для визуализации")
            return

        plt.figure(figsize=(12, 10))
        
        # Область
        plt.plot(self.domain_coords[:, 0], self.domain_coords[:, 1],
                 'b-', linewidth=2, label='Контур', zorder=2)
        plt.scatter(self.domain_coords[:, 0], self.domain_coords[:, 1],
                    c='blue', s=20, alpha=0.5, zorder=1)

        # Аннотации
        x_min = np.min(self.domain_coords[:, 0])
        x_max = np.max(self.domain_coords[:, 0])
        y_min = np.min(self.domain_coords[:, 1])
        y_max = np.max(self.domain_coords[:, 1])
        
        # INLET
        inlet_y = (y_min - R_inner) / 2 if hasattr(self, 'pipe_width') else -0.5
        plt.annotate('INLET', (x_min / 2, inlet_y),
                     fontsize=12, color='green', ha='center', weight='bold')
        
        # OUTLET
        plt.annotate('OUTLET', (x_max, y_max / 2),
                     fontsize=12, color='blue', ha='center', weight='bold')
        
        # WALL
        plt.annotate('WALL (внешняя)', (0, y_min * 0.3),
                     fontsize=10, color='red', ha='center')
        plt.annotate('WALL (внутренняя)', (0, y_max * 0.3),
                     fontsize=10, color='red', ha='center')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X, м', fontsize=12)
        plt.ylabel('Y, м', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"График сохранён: {save_path}")

        if self.config.plot_show:
            plt.show()
        else:
            plt.close()

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> Optional[str]:
        """
        Сохранение геометрии.

        Args:
            filename: Базовое имя файла
            format: Формат ('npy', 'csv', 'txt')

        Returns:
            Путь к сохранённому файлу области
        """
        fmt = format or self.config.save_format
        base_name = filename or "pipe_bend"

        domain_path = None
        if self.domain_coords is not None:
            domain_path = self._save_coords(self.domain_coords, f"{base_name}_domain", fmt)

            # Метаданные
            metadata = {
                'geom_type': 'pipe_bend',
                'pipe_bend_params': self.pipe_bend_params or {
                    'inlet_length': self.inlet_length,
                    'outlet_length': self.outlet_length,
                    'bend_radius': self.bend_radius,
                    'pipe_width': self.pipe_width,
                    'bend_angle': self.bend_angle
                },
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(self.config.output_dir, f"{base_name}_metadata.npy")
            np.save(meta_path, metadata)
            if self.config.verbose:
                logger.info(f"Метаданные сохранены: {meta_path}")

        return domain_path

    def _save_coords(self, coords: np.ndarray, basename: str, fmt: str) -> str:
        """Сохранение координат в указанном формате."""
        filepath = os.path.join(self.config.output_dir, f"{basename}.{fmt}")
        if fmt == 'npy':
            np.save(filepath, coords)
        elif fmt == 'csv':
            header = "x,y"
            np.savetxt(filepath, coords, delimiter=',', header=header, comments='')
        else:
            np.savetxt(filepath, coords, delimiter='\t', fmt='%.8f')
        if self.config.verbose:
            logger.info(f"Сохранено: {filepath}")
        return filepath

    def print_stats(self) -> None:
        """Вывод статистики геометрии."""
        print("\n" + "=" * 50)
        print("СТАТИСТИКА ГЕОМЕТРИИ".center(50))
        print("=" * 50)
        
        print(f"\n📐 ОТВОД 90°:")
        print(f"   Ширина канала: {self.pipe_width} м")
        print(f"   Радиус изгиба: {self.bend_radius} м")
        print(f"   R/D: {self.bend_radius / self.pipe_width:.2f}")
        print(f"   Входной участок: {self.inlet_length} м")
        print(f"   Выходной участок: {self.outlet_length} м")
        
        if self.domain_coords is not None:
            print(f"\n🗺️  РАСЧЁТНАЯ ОБЛАСТЬ:")
            print(f"   Точек контура: {len(self.domain_coords)}")
            x_min, y_min = np.min(self.domain_coords, axis=0)
            x_max, y_max = np.max(self.domain_coords, axis=0)
            print(f"   X: [{x_min:.4f}, {x_max:.4f}] м")
            print(f"   Y: [{y_min:.4f}, {y_max:.4f}] м")
        print("=" * 50)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def g2d() -> None:
    """
    Главная функция модуля геометрии.
    Автоматически загружает конфигурацию из geom_config.yaml,
    создаёт геометрию отвода, визуализирует и сохраняет.
    """
    print('\033c')
    print("=" * 60)
    print("МОДУЛЬ ГЕОМЕТРИИ ОТВОДА 90°".center(60))
    print("=" * 60)

    config = GeometryConfig(save_format='npy', plot_show=True,
                           output_dir=DEFAULT_OUTPUT_DIR, verbose=True)

    # Загрузка конфигурации
    geom_config_path = GEOM_CONFIG_PATH
    if not os.path.exists(geom_config_path):
        logger.error(f"Конфигурационный файл не найден: {geom_config_path}")
        input("\nНажмите Enter для выхода...")
        return

    try:
        with open(geom_config_path, 'r', encoding='utf-8') as f:
            geom_config = yaml.safe_load(f)
        
        geom_name = geom_config.get('geom_name', 'PB_R100')
        print(f"\n✅ Конфигурация загружена: {geom_config_path}")
        print(f"   Имя геометрии: {geom_name}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        input("\nНажмите Enter для выхода...")
        return

    # Создание геометрии
    geom = PipeBendGeometry(config=config)
    if not geom.load_from_config(geom_config_path):
        input("\n❌ Ошибка загрузки. Нажмите Enter...")
        return

    geom.print_stats()

    print("\n🔧 СОЗДАНИЕ РАСЧЁТНОЙ ОБЛАСТИ...")
    geom.create_domain()
    geom.print_stats()

    print("\n📊 Визуализация геометрии...")
    plot_path = os.path.join(config.output_dir, f"geometry_pipe_bend.png")
    geom.plot(save_path=plot_path)

    print("\n💾 Сохранение...")
    domain_path = geom.save(geom_name)

    print(f"\n✅ Сохранено:")
    print(f"   Область: {domain_path}")
    print(f"   Метаданные: {os.path.join(config.output_dir, f'{geom_name}_metadata.npy')}")

    print("\n" + "=" * 60)
    print("✅ РАБОТА МОДУЛЯ ЗАВЕРШЕНА")
    print("=" * 60)
    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    g2d()