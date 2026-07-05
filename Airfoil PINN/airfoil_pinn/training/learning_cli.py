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

import torch

# Импорты модулей проекта
from airfoil_pinn.config.manager import ConfigManager, yes_no
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

def output_menu(cfg: ConfigManager) -> None:
    """Меню настройки вывода."""
    while True:
        clear_screen()
        print_submenu_header("Настройки вывода")
        out = cfg.config['output']
        print("1. Только консоль / графическое окно")
        print("2. Частота обновления графиков (эпох)")
        print("3. Показать настройки вывода")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            out['console_only'] = yes_no("Только консоль (иначе графические окна)", out['console_only'])
            if not out['console_only']:
                gw = out['graphical_window']
                gw['loss'] = yes_no("Показывать общую потерю (Loss)", gw['loss'])
                gw['pde'] = yes_no("Показывать PDE (каждая невязка)", gw['pde'])
                gw['bc'] = yes_no("Показывать BC (каждая невязка)", gw['bc'])
        elif sub == '2':
            out['plot_frequency'] = int(input(f"Частота обновления графиков (эпох) [{out['plot_frequency']}]: ") or out['plot_frequency'])
        elif sub == '3':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ВЫВОДА ---")
            print(f"Только консоль: {out['console_only']}")
            if not out['console_only']:
                print(f"  Loss: {out['graphical_window']['loss']}")
                print(f"  PDE: {out['graphical_window']['pde']}")
                print(f"  BC: {out['graphical_window']['bc']}")
            print(f"Частота обновления графиков: {out['plot_frequency']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def checkpoint_menu(cfg: ConfigManager) -> None:
    """Меню настройки сохранения."""
    while True:
        clear_screen()
        print_submenu_header("Настройки сохранения")
        ck = cfg.config['checkpoint']
        print("1. Период сохранения контрольных точек")
        print("2. Имя для сохранения контрольных точек")
        print("3. Имя нейронной сети при сохранении")
        print("4. Показать настройки сохранения")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            ck['interval'] = int(input(f"Период сохранения чекпоинтов (0 - выкл) [{ck['interval']}]: ") or ck['interval'])
        elif sub == '2':
            ck['name_pattern'] = input(f"Шаблон имени чекпоинта (используйте {{project_name}}, {{timestamp}}) [{ck['name_pattern']}]: ").strip() or ck['name_pattern']
        elif sub == '3':
            ck['model_name_pattern'] = input(f"Шаблон имени модели [{ck['model_name_pattern']}]: ").strip() or ck['model_name_pattern']
        elif sub == '4':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ СОХРАНЕНИЯ ---")
            print(f"Период чекпоинтов: {ck['interval']}")
            print(f"Шаблон чекпоинта: {ck['name_pattern']}")
            print(f"Шаблон модели: {ck['model_name_pattern']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def diagnostics_menu(cfg: ConfigManager) -> None:
    """Меню настройки диагностики."""
    while True:
        clear_screen()
        print_submenu_header("Настройки диагностики")
        diag = cfg.config.get('diagnostics', {})
        if not diag:
            diag = cfg.config['diagnostics'] = {}
        print("1. Интервал диагностики (эпох)")
        print("2. Построение карты невязки")
        print("3. Построение карты полной ошибки RAR")
        print("4. Построение профилей скорости")
        print("5. Сечения для профилей (через запятую)")
        print("6. Построение спектра (1D)")
        print("7. Координата X для спектра (1D)")
        print("8. Построение 2D спектра")
        print("9. Разрешение 2D спектра")
        print("10. Вычисление раздельных градиентов")
        print("11. Вычисление NTK")
        print("12. Сохранение истории RAR")
        print("13. Интервал inlet/outlet метрик")
        print("14. Показать текущие настройки")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            diag['interval'] = int(input(f"Интервал (эпох) [{diag.get('interval', 50)}]: ") or diag.get('interval', 50))
        elif sub == '2':
            diag['plot_residual_map'] = yes_no("Строить карту невязки", diag.get('plot_residual_map', True))
        elif sub == '3':
            diag['plot_rar_error_map'] = yes_no("Строить карту полной ошибки RAR", diag.get('plot_rar_error_map', True))
        elif sub == '4':
            diag['plot_profiles'] = yes_no("Строить профили скорости", diag.get('plot_profiles', True))
        elif sub == '5':
            current = diag.get('profile_sections', [0.2, 0.5, 0.8])
            sections_str = input(f"Сечения (x, м) через запятую [{', '.join(map(str, current))}]: ").strip()
            if sections_str:
                try:
                    diag['profile_sections'] = [float(x.strip()) for x in sections_str.split(',')]
                except ValueError:
                    print("Неверный формат, оставлено текущее значение.")
        elif sub == '6':
            diag['plot_spectrum'] = yes_no("Строить 1D спектр", diag.get('plot_spectrum', True))
        elif sub == '7':
            diag['spectrum_section_x'] = float(input(f"X для 1D спектра (м) [{diag.get('spectrum_section_x', 0.5)}]: ") or diag.get('spectrum_section_x', 0.5))
        elif sub == '8':
            diag['plot_2d_spectrum'] = yes_no("Строить 2D спектр", diag.get('plot_2d_spectrum', False))
        elif sub == '9':
            diag['spectrum_2d_resolution'] = int(input(f"Разрешение 2D спектра [{diag.get('spectrum_2d_resolution', 256)}]: ") or diag.get('spectrum_2d_resolution', 256))
        elif sub == '10':
            diag['compute_grad_cosine'] = yes_no("Вычислять раздельные градиенты", diag.get('compute_grad_cosine', False))
        elif sub == '11':
            diag['compute_ntk_condition'] = yes_no("Вычислять NTK", diag.get('compute_ntk_condition', False))
        elif sub == '12':
            diag['save_rar_history'] = yes_no("Сохранять историю RAR", diag.get('save_rar_history', True))
        elif sub == '13':
            diag['mass_metrics_interval'] = int(
                input(f"Интервал inlet/outlet метрик [{diag.get('mass_metrics_interval', 1)}]: ")
                or diag.get('mass_metrics_interval', 1)
            )
            diag['mass_metrics_interval'] = max(1, diag['mass_metrics_interval'])
        elif sub == '14':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ДИАГНОСТИКИ ---")
            print(f"Интервал: {diag.get('interval', 50)}")
            print(f"Карта невязки: {diag.get('plot_residual_map', True)}")
            print(f"Карта полной ошибки RAR: {diag.get('plot_rar_error_map', True)}")
            print(f"Профили: {diag.get('plot_profiles', True)}")
            print(f"Сечения: {diag.get('profile_sections', [0.2, 0.5, 0.8])}")
            print(f"1D спектр: {diag.get('plot_spectrum', True)}")
            print(f"X 1D спектра: {diag.get('spectrum_section_x', 0.5)}")
            print(f"2D спектр: {diag.get('plot_2d_spectrum', False)}")
            print(f"Разрешение 2D спектра: {diag.get('spectrum_2d_resolution', 256)}")
            print(f"Раздельные градиенты: {diag.get('compute_grad_cosine', False)}")
            print(f"NTK: {diag.get('compute_ntk_condition', False)}")
            print(f"История RAR: {diag.get('save_rar_history', True)}")
            print(f"Интервал inlet/outlet метрик: {diag.get('mass_metrics_interval', 1)}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def process_settings(cfg: ConfigManager) -> None:
    """Меню настроек процесса обучения."""
    while True:
        clear_screen()
        print_submenu_header("Настройки процесса обучения")
        proc = cfg.config['process']
        print("1. Загрузить PINN и её конфигурацию для дообучения")
        print("2. Выбор GPU/CPU и их количества")
        print("3. Выбор типа данных")
        print("4. Показать настройки")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            if yes_no("Загрузить PINN и её конфигурацию для дообучения", proc['load_model_path'] is not None):
                default_path = os.path.join(cfg.config.get('checkpoint_dir', 'Checkpoints'), 'pinn_interrupted.pt')
                path = input(f"Путь к файлу модели (.pt) [Enter для '{default_path}']: ").strip()
                if not path:
                    path = default_path
                if os.path.exists(path):
                    proc['load_model_path'] = path
                    base = os.path.splitext(os.path.basename(path))[0]
                    conf_path = os.path.join(CONFIG_DIR, base + '.yaml')
                    if os.path.exists(conf_path):
                        cfg.load(base)
                        print(f"Конфигурация {base} загружена.")
                    else:
                        print("Конфигурация не найдена, будет использована текущая.")
                else:
                    print("Файл не найден.")
                    proc['load_model_path'] = None
            else:
                proc['load_model_path'] = None
        elif sub == '2':
            proc['device'] = input(f"Устройство (auto/cuda/cpu) [{proc['device']}]: ").strip() or proc['device']
            if torch.cuda.is_available() and proc['device'] != 'cpu':
                proc['num_gpus'] = int(input(f"Количество GPU [{proc['num_gpus']}]: ") or proc['num_gpus'])
            else:
                proc['num_gpus'] = 0
        elif sub == '3':
            proc['dtype'] = input(f"Тип данных (float32/float64/float16) [{proc['dtype']}]: ").strip() or proc['dtype']
            if proc['dtype'] == 'float32':
                proc['use_amp'] = yes_no("Использовать AMP", proc['use_amp'])
        elif sub == '4':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ПРОЦЕССА ---")
            print(f"Загруженная модель: {proc['load_model_path']}")
            print(f"Устройство: {proc['device']}, GPU: {proc['num_gpus']}")
            print(f"Тип данных: {proc['dtype']}, AMP: {proc['use_amp']}")
            print(f"Seed: {proc.get('seed', 42)}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


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
