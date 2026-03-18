#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль геометрии (G2D) для построения контуров расчётных областей
плоских тройников, заданных параметров

ВСЕ КООРДИНАТЫ В МЕТРАХ (без нормализации)
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import interpolate

# Определяем корневую директорию проекта
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Geometry", "Data")
DEFAULT_PLOT_DPI = 300

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GeometryConfig:
    """Конфигурация для геометрических параметров"""
    enable_input: bool = False
    save_format: str = 'npy'          # 'npy', 'csv', 'txt'
    plot_enabled: bool = True
    show_geom: bool = True
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: str = DEFAULT_OUTPUT_DIR


class JunctionGeometry:
    """
    Класс для работы с геометрией плоского T-образного тройника (бесконечного по оси Z)
    и его расчетной областью.
    Определяется линейными и угловыми размерами трёх прямоугольных каналов,
    сходящихся в начале координат (0,0).
    ВСЕ КООРДИНАТЫ В МЕТРАХ
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.junction_params: tuple = None
        self.domain_coords: Optional[np.ndarray] = None
        self.config = config or GeometryConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_junction(self, geom_config_path: str, junctions_config_path: str = 'Airfoil PINN\Geometry\junction_geoms.yaml') -> bool:
        """
        Загружает конфигурацию тройника из YAML файла.
        Ожидаемая структура:
            chanel_width: ...
            down_length: ...
            side_length: ...
            top_length: ...
            down2side_angl: ...
            side2top_angl: ...
        """
        try:
            with open(geom_config_path, 'r', encoding='utf-8') as geom_config:
                data = yaml.safe_load(geom_config)
            
            junction_name = data['geom_name']

            with open(junctions_config_path, 'r', encoding='utf-8') as junction_config:
                data = yaml.safe_load(junction_config)

            junctions = {junction['name']: junction for junction in data['junctions']}
            self.junction_params = junctions[f'{junction_name}']

            logger.info(f'Конфигурационный файл загружен. Модель: {junction_name}')

            return True
        
        except Exception as e:
            logger.error(f'Ошибка при импорте конфигурационного файла: {e}')

        return False

    def create_domain(self, **kwargs) -> None:
        """
        Создает замкнутый контур расчетной области.

        Returns:
            Массив координат границ области или None в случае ошибки.
        """
        if self.junction_params is None:
            logger.error('Конфигурационный файл тройника не загружен.')
            return None

        # Характерные размеры
        chanel_width = float(self.junction_params['chanel_width'])
        down_length = float(self.junction_params['down_length'])
        side_length = float(self.junction_params['side_length'])
        top_length = float(self.junction_params['top_length'])

        # Вычисление углов ответвлений для будущего усложнения функционала
        # down2side_angl = np.deg2rad(float(self.junction_params['down2side_angl']))
        # side2top_angl = np.deg2rad(float(self.junction_params['side2top_angl']))

        domain_coords = []

        # Точки расчетной области. Порядок обхода : против часовой
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

        domain_coords.append(domain_coords[0])

        logger.info(f'Расчетная область создана.')
        logger.info(f'Контур включает {len(domain_coords)} точек.')

        self.domain_coords = np.array(domain_coords)

    def show_domain(
            self,
            show_plot: bool = False,
            title: str = f'Геометрия расчетной области',
            save_path: Optional[str] = None
        ) -> None:
        """
        Визуализация геометрии расчетной области.
        """
        
        plt.figure(figsize=(12, 8))

        # Область (всегда ЗАМКНУТЫЙ контур)
        if self.domain_coords is not None:
            plt.plot(self.domain_coords[:, 0], self.domain_coords[:, 1],
                     'ro-', linewidth=2, markersize=8, label='Граница области', zorder=1)
            plt.scatter(self.domain_coords[:, 0], self.domain_coords[:, 1],
                        c='red', s=100, marker='s', alpha=0.7, zorder=2)

        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X, м', fontsize=12)
        plt.ylabel('Y, м', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save(self, filename: Optional[str] = None, format: Optional[str] = None) -> None:
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
                'num_domain_points': len(self.domain_coords)
            }
            meta_path = os.path.join(self.config.output_dir, f'{base_name}_metadata.npy')
            np.save(meta_path, metadata)
            logger.info(f'Метаданные сохранены: {meta_path}')

    def _save_coords(self, coords: np.ndarray, basename: str, fmt: str) -> None:
        '''Сохранение координат в указанном формате'''
        filepath = os.path.join(self.config.output_dir, f"{basename}.{fmt}")

        if fmt == 'npy':
            np.save(filepath, coords)
        elif fmt == 'csv':
            header = 'x,y'
            np.savetxt(filepath, coords, delimiter=',', header=header, comments='')
        else:  # txt или другой текстовый
            np.savetxt(filepath, coords, delimiter='\t', fmt='%.8f')

        logger.info(f'Сохранено: {filepath}')

    def print_stats(self) -> None:
        '''Вывод информации о контуре.'''

        if self.domain_coords is not None:
            print(f'\n РАСЧЁТНАЯ ОБЛАСТЬ:')
            print(f'   Точек: {len(self.domain_coords)}')
            print(f'   Вершины:')
            for i, (x, y) in enumerate(self.domain_coords):
                print(f'     {i+1}: ({x:.4f}, {y:.4f}) м')


def print_title(title: str, line_width: int = 60):
    '''Печать заголовков исполняемых команд.'''
    print("\n" + "=" * line_width)
    print(title.center(line_width))
    print("=" * line_width)

def g2d() -> None:
    '''Главная функция модуля геометрии с возможностью интерактивного ввода исоходных данных.'''
    
    config = GeometryConfig(
        enable_input=False,
        save_format='npy',
        show_geom=False,
        output_dir=DEFAULT_OUTPUT_DIR
    )

    print('\033c')
    print_title('ЗАПУСК МОДУЛЯ ГЕОМЕТРИЯ G2D')
    geom = JunctionGeometry(config=config)

    if config.enable_input:
        print('\n Укажите путь к конфигурационному файлу тройника.')
        path = input('Путь к файлу geom_config.yaml: ').strip()
        if path == '':
            path = 'Airfoil PINN\geom_config.yaml'
    else:
        path = 'Airfoil PINN\geom_config.yaml'
    
    if not geom.load_junction(path):
        input('\n Ошибка загрузки. Нажмите Enter...')
        return

    print_title('СОЗДАНИЕ КОНТУРА РАСЧЁТНОЙ ОБЛАСТИ')

    geom.create_domain()

    print_title('ПАРАМЕТРЫ КОНТУРА РАСЧЕТНОЙ ОБЛАСТИ')
    geom.print_stats()

    print('\n Визуализация геометрии контура расчетной области...')
    plot_path = os.path.join(config.output_dir, f'geometry_{geom.junction_params['name']}.png')
    geom.show_domain(show_plot=config.show_geom, save_path=plot_path)

    print("\n Сохранение...")
    geom.save(f'junction_{geom.junction_params['name']}')

    print_title('ЗАВЕРШЕНИЕ МОДУЛЯ ГЕОМЕТРИЯ G2D')
    input("Нажмите Enter для возврата...")


if __name__ == "__main__":
    g2d()