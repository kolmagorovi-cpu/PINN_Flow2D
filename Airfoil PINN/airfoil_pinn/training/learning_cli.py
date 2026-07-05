#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный исполняемый файл проекта PINN Learning.
Обеспечивает иерархическое меню для создания/загрузки/редактирования конфигурации,
настройки процесса обучения и запуска обучения.
Поддерживает пакетный запуск через аргументы командной строки.

Автор: Колмагоров И.Д.
Версия: 3.260226 (с поддержкой пакетного запуска и experiment_dir)
Дополнено: меню настройки диагностики с поддержкой 2D спектра.
"""

import os
import sys
import argparse
import logging

# Импорты модулей проекта
from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.menus.architecture import (
    _configure_activation,
    _configure_coordinate_scaling,
    _configure_hidden_layers,
    _configure_input_params,
    _configure_output_params,
    architecture_menu,
)
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.training.menus.loss import (
    loss_function_menu,
    weights_menu,
)
from airfoil_pinn.training.menus.process import process_settings
from airfoil_pinn.training.menus.results import (
    checkpoint_menu,
    diagnostics_menu,
    output_menu,
)
from airfoil_pinn.training.menus.strategy import training_strategy_menu
from airfoil_pinn.training.runner import (
    DEFAULT_POINTS_FILE,
    run_experiment,
    start_training,
)
from airfoil_pinn.utils import (
    clear_screen,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Константы
CONFIG_DIR = ConfigManager.CONFIG_DIR


def main_menu() -> None:
    """Главное меню модуля обучения PINN (интерактивный режим)."""
    cfg = ConfigManager()
    while True:
        clear_screen()
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ ОБУЧЕНИЯ PINN".center(60))
        print("=" * 60)
        print("1. Создать новую конфигурацию")
        print("2. Загрузить конфигурацию (файл с настройками)")
        print("3. Редактировать конфигурацию")
        print("4. Сохранить конфигурацию")
        print("5. Показать текущую конфигурацию")
        print("6. Настройки процесса обучения")
        print("7. Старт обучения")
        print("0. Выйти")
        choice = input("Ваш выбор: ").strip()

        if choice == '1':
            create_new_config(cfg)
        elif choice == '2':
            load_config(cfg)
        elif choice == '3':
            edit_config(cfg)
        elif choice == '4':
            save_config(cfg)
        elif choice == '5':
            cfg.show()
            input("\nНажмите Enter для продолжения...")
        elif choice == '6':
            process_settings(cfg)
        elif choice == '7':
            start_training(cfg)
        elif choice == '0':
            print("Выход.")
            break
        else:
            print("Неверный ввод, повторите.")
            input("Нажмите Enter...")


def create_new_config(cfg: ConfigManager) -> None:
    """Создание новой конфигурации с выбором разделов."""
    cfg.config = cfg._default_config()
    name = input("Введите имя проекта (по умолчанию 'unnamed'): ").strip() or 'unnamed'
    cfg.name = name
    cfg.config['project_name'] = name
    cfg.config['data']['points_file'] = DEFAULT_POINTS_FILE

    while True:
        clear_screen()
        print_submenu_header(f"СОЗДАНИЕ КОНФИГУРАЦИИ: {cfg.name}")
        print("1. Архитектура сети")
        print("2. Функция потерь")
        print("3. Стратегия обучения")
        print("4. Настройки вывода")
        print("5. Настройки сохранения")
        print("6. Диагностика")
        print("7. Показать текущую конфигурацию")
        print("0. Завершить создание и вернуться в главное меню")
        sub = input("Выберите раздел: ").strip()

        if sub == '1':
            architecture_menu(cfg)
        elif sub == '2':
            loss_function_menu(cfg)
        elif sub == '3':
            training_strategy_menu(cfg)
        elif sub == '4':
            output_menu(cfg)
        elif sub == '5':
            checkpoint_menu(cfg)
        elif sub == '6':
            diagnostics_menu(cfg)
        elif sub == '7':
            cfg.show()
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break
        else:
            print("Неверный ввод.")
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
        print(f"Конфигурация '{name}' загружена.")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
    input("Нажмите Enter...")


def edit_config(cfg: ConfigManager) -> None:
    """Редактирование текущей конфигурации."""
    if cfg.name == 'unnamed' and cfg.config['project_name'] == 'unnamed':
        print("Сначала создайте или загрузите конфигурацию.")
        input("Нажмите Enter...")
        return

    while True:
        clear_screen()
        print_submenu_header("РЕДАКТИРОВАНИЕ КОНФИГУРАЦИИ")
        print("1. Архитектура сети")
        print("2. Функция потерь")
        print("3. Стратегия обучения")
        print("4. Настройки вывода")
        print("5. Настройки сохранения")
        print("6. Диагностика")
        print("7. Показать текущую конфигурацию")
        print("0. Вернуться в главное меню")
        sub = input("Выберите раздел: ").strip()

        if sub == '1':
            architecture_menu(cfg)
        elif sub == '2':
            loss_function_menu(cfg)
        elif sub == '3':
            training_strategy_menu(cfg)
        elif sub == '4':
            output_menu(cfg)
        elif sub == '5':
            checkpoint_menu(cfg)
        elif sub == '6':
            diagnostics_menu(cfg)
        elif sub == '7':
            cfg.show()
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break
        else:
            print("Неверный ввод.")
            input("Нажмите Enter...")


def save_config(cfg: ConfigManager) -> None:
    """Сохранение текущей конфигурации с возможностью переименования."""
    current_name = cfg.name
    prompt = f"Введите имя для сохранения (текущее: '{current_name}', Enter - оставить): "
    new_name = input(prompt).strip()
    if new_name:
        cfg.name = new_name
        cfg.config['project_name'] = new_name
    elif current_name == 'unnamed':
        print("Имя не может быть 'unnamed'. Сохранение отменено.")
        input("Нажмите Enter...")
        return
    cfg.save()
    print(f"Конфигурация сохранена как {cfg.name}.yaml")
    input("Нажмите Enter...")

def main():
    """Основная функция: обрабатывает аргументы командной строки и запускает либо меню, либо эксперимент."""
    parser = argparse.ArgumentParser(description="PINN Learning - главный модуль")
    parser.add_argument("--config", type=str, help="Путь к файлу конфигурации для запуска эксперимента (без меню)")
    parser.add_argument("--experiment_dir", type=str, help="Директория для сохранения результатов эксперимента")
    args = parser.parse_args()

    if args.config:
        # Пакетный режим: запускаем эксперимент
        if not args.experiment_dir:
            logger.error("При использовании --config необходимо указать --experiment_dir")
            sys.exit(1)
        run_experiment(args.config, args.experiment_dir)
    else:
        # Интерактивный режим
        main_menu()


if __name__ == "__main__":
    main()
