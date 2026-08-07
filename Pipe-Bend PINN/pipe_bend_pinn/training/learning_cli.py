from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль обучения PINN для Pipe-Bend PINN.

Обеспечивает интерактивное меню для настройки конфигурации и запуска обучения.

Исправленная версия:
- Абсолютные пути вместо относительных
- Улучшенный поиск файла точек с диагностикой
- Авто-создание конфигурации при нажатии "Старт обучения" без конфигурации

Автор: Pipe-Bend PINN Team
Версия: 1.2
"""

import os
import sys
import logging
from typing import Optional

import torch

from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.app.paths import PROJECT_ROOT, INITIAL_FIELDS_DIR, CONFIGS_DIR
from pipe_bend_pinn.config.manager import ConfigManager
from pipe_bend_pinn.training.solver import PINNSolver

logger = logging.getLogger(__name__)

# Корень проекта (из единой карты путей)
BASE_DIR = str(PROJECT_ROOT)
DEFAULT_POINTS_FILE = os.path.join(INITIAL_FIELDS_DIR, "initial.npy")
CONFIG_DIR = CONFIGS_DIR


def clear_screen() -> None:
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print('\033c')


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(title.center(60))
    print(f"{'='*60}")


def _find_points_file() -> Optional[str]:
    """
    Ищет файл с точками в Initial/Data/.
    Сначала пробует initial.npy, затем любой .npy файл (кроме _metadata).
    Возвращает полный путь или None.
    """
    # 1. Пробуем стандартный файл
    if os.path.exists(DEFAULT_POINTS_FILE):
        return DEFAULT_POINTS_FILE

    # 2. Ищем любой .npy в workspace/initial/fields
    data_dir = INITIAL_FIELDS_DIR
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir)
                 if f.endswith('.npy') and '_metadata' not in f]
        if files:
            return os.path.join(data_dir, files[0])

    # 3. Проверяем, существует ли вообще папка
    if not os.path.exists(data_dir):
        print(f"   📁 Папка {data_dir} не найдена")
    else:
        all_files = os.listdir(data_dir)
        print(f"   📁 Содержимое {data_dir}: {all_files[:10]}...")
    
    return None


def main_menu() -> None:
    """Главное меню модуля обучения PINN."""
    cfg = ConfigManager()
    config_created = False

    while True:
        clear_screen()
        print_header("ОБУЧЕНИЕ PINN")
        print("1. Создать новую конфигурацию")
        print("2. Загрузить конфигурацию")
        print("3. Показать текущую конфигурацию")
        print("4. Старт обучения")
        print("0. Вернуться в главное меню")
        choice = input("Ваш выбор: ").strip()

        if choice == '1':
            create_config(cfg)
            config_created = True
        elif choice == '2':
            load_config(cfg)
            config_created = True
        elif choice == '3':
            cfg.show()
            input("\nНажмите Enter...")
        elif choice == '4':
            # Если конфигурация не создана — создаём автоматически
            if not config_created:
                points_file = _find_points_file()
                if points_file:
                    cfg.config = cfg._default_config()
                    cfg.config['data']['points_file'] = points_file
                    cfg.config['flow_type'] = 'planar'
                    cfg.name = 'pipe_bend_auto'
                    config_created = True
                    print(f"\n✅ Авто-конфигурация: файл точек = {points_file}")
                else:
                    print(f"\n{'='*60}")
                    print("НЕТ ДАННЫХ ДЛЯ ОБУЧЕНИЯ".center(60))
                    print(f"{'='*60}")
                    print("Сначала выполните 'Быструю подготовку' [4]")
                    print("из главного меню, чтобы создать данные.")
                    print(f"{'='*60}")
                    input("\nНажмите Enter...")
                    continue
            start_training(cfg)
        elif choice == '0':
            break
        else:
            print("Неверный ввод.")
            input("Нажмите Enter...")


def create_config(cfg: ConfigManager) -> None:
    """Создание новой конфигурации."""
    default_points = _find_points_file()

    cfg.config = cfg._default_config()
    name = input("Введите имя проекта (по умолчанию 'pipe_bend'): ").strip() or 'pipe_bend'
    cfg.name = name
    cfg.config['project_name'] = name
    cfg.config['data']['points_file'] = default_points or "НЕ УКАЗАН"

    print(f"\nКонфигурация '{name}' создана.")
    print(f"Файл точек: {default_points or '⚠️ НЕ НАЙДЕН! Сначала запустите инициализацию.'}")

    # Настройка количества эпох
    epochs_str = input("Количество эпох [1000]: ").strip()
    if epochs_str:
        cfg.config['training']['epochs'] = int(epochs_str)

    # Настройка batch_size
    bs_str = input("Размер батча [1200]: ").strip()
    if bs_str:
        cfg.config['training']['batch_size'] = int(bs_str)

    # Two-stage обучение
    two_stage = input("Двухэтапное обучение? (y/N): ").strip().lower()
    cfg.config['training']['two_stage'] = two_stage in ('y', 'yes')

    print("✅ Конфигурация создана.")
    input("Нажмите Enter...")


def load_config(cfg: ConfigManager) -> None:
    """Загрузка существующей конфигурации."""
    names = [f.replace('.yaml', '') for f in os.listdir(CONFIG_DIR) if f.endswith('.yaml')]
    if not names:
        print("Нет сохранённых конфигураций.")
        input("Нажмите Enter...")
        return

    print("Доступные конфигурации:")
    for i, n in enumerate(names, 1):
        print(f"{i}. {n}")
    choice = input("Введите номер или имя: ").strip()
    try:
        idx = int(choice) - 1
        name = names[idx]
    except ValueError:
        name = choice

    try:
        cfg.load(name)
        print(f"✅ Конфигурация '{name}' загружена.")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
    input("Нажмите Enter...")


def start_training(cfg: ConfigManager) -> None:
    """Запуск обучения с текущей конфигурацией."""
    points_file = cfg.config['data'].get('points_file')

    # Диагностика
    print(f"\n🔍 Диагностика:")
    print(f"   BASE_DIR = {BASE_DIR}")
    print(f"   points_file из конфига = {points_file}")
    print(f"   Существует? = {os.path.exists(points_file) if isinstance(points_file, str) else 'не строка'}")

    # Если файл не найден — ищем автоматически
    if not isinstance(points_file, str) or not os.path.exists(points_file):
        if points_file:
            print(f"⚠️ Файл не найден: {points_file}")
        
        found = _find_points_file()
        if found:
            points_file = found
            cfg.config['data']['points_file'] = points_file
            print(f"✅ Найден файл точек: {points_file}")
        else:
            print(f"\n{'='*60}")
            print("НЕТ ДАННЫХ ДЛЯ ОБУЧЕНИЯ".center(60))
            print(f"{'='*60}")
            print("Выполните 'Быструю подготовку' [4] из главного меню.")
            print("Или вручную:")
            print("  python test_pipeline.py")
            print(f"{'='*60}")
            input("\nНажмите Enter...")
            return

    # Убедимся, что веса потерь для pipe bend не содержат лишних типов
    bc_weights = cfg.config['loss_weights']['bc_weights']
    for key in ['symmetry', 'axis_symmetry', 'symmetry_grad', 'axis_symmetry_grad']:
        bc_weights.pop(key, None)
    if 'wall' not in bc_weights:
        bc_weights['wall'] = 5000.0

    cfg.config['flow_type'] = 'planar'

    print("\n📋 Текущая конфигурация:")
    cfg.show()
    print("Нажмите Enter для начала обучения или Ctrl+C для отмены...")
    input()

    try:
        solver = PINNSolver(cfg.config)
        solver.train()
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nНажмите Enter для возврата в меню...")