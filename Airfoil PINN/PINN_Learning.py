# Исправленный файл: PINN_Learning.py

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
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Импорты модулей проекта
from config_manager import ConfigManager, yes_no
from solver import PINNSolver
from checkpoint import CheckpointManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Константы
DEFAULT_POINTS_FILE = os.path.join("Initial", "Data", "initial.npy")
PICTURES_DIR = "Pictures"
CONFIG_DIR = ConfigManager.CONFIG_DIR


def clear_screen() -> None:
    """Очищает экран с помощью ANSI escape-последовательности."""
    print('\033c', end='')


def print_submenu_header(title: str) -> None:
    """Выводит заголовок подменю."""
    print(f"\n{title} {{")


def print_submenu_footer() -> None:
    """Выводит закрывающую скобку подменю."""
    print("}")


def visualize_network(config: dict, filename: str) -> None:
    """
    Строит схему нейронной сети в виде столбцов и сохраняет в PNG.
    """
    input_params = config.get('input_params', {})
    arch = config.get('architecture', {})
    hidden_layers = arch.get('hidden_layers', [50, 50, 50, 50])
    activation = arch.get('activation', 'tanh')
    output_dim = arch.get('output_dim', 3)

    # Формируем подписи для входного слоя
    input_labels = ["x", "y", "sdf"]
    if input_params.get('inlet_velocity', False):
        input_labels.extend(["u_in", "v_in"])
    if input_params.get('viscosity', False):
        input_labels.append("μ")
    if input_params.get('density', False):
        input_labels.append("ρ")
    # one-hot границ всегда присутствует, если есть типы границ,
    # но для схемы добавим как один обобщённый нейрон
    input_labels.append("boundary (one‑hot)")

    input_dim = len(input_labels)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(hidden_layers) * 2), 6))
    ax.set_xlim(0, len(hidden_layers) + 2.5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    color_input = '#FFDDC1'
    color_hidden = '#C1E1FF'
    color_output = '#D1FFD1'

    box_width = 0.8
    max_neurons = max(hidden_layers) if hidden_layers else 1
    scale = 8.0 / max(max_neurons, 1)

    # Входной слой
    x_in = 0.5
    height_in = input_dim * scale
    rect_in = patches.Rectangle((x_in - box_width / 2, 1), box_width, height_in,
                                 linewidth=2, edgecolor='black', facecolor=color_input)
    ax.add_patch(rect_in)
    ax.text(x_in, 1 + height_in / 2, f"Input\n{input_dim} neurons",
            ha='center', va='center', fontsize=10, weight='bold')
    y_text = 1 + height_in + 0.2
    ax.text(x_in, y_text, "\n".join(input_labels[:5]), ha='center', va='bottom', fontsize=7)
    if len(input_labels) > 5:
        ax.text(x_in, y_text - 0.3, "...", ha='center', va='bottom', fontsize=7)

    # Скрытые слои
    for i, neurons in enumerate(hidden_layers):
        x = 1.5 + i
        height = neurons * scale
        rect = patches.Rectangle((x - box_width / 2, 1), box_width, height,
                                 linewidth=2, edgecolor='black', facecolor=color_hidden)
        ax.add_patch(rect)
        # Добавляем информацию об активации
        ax.text(x, 1 + height / 2, f"Layer {i + 1}\n{neurons}\n{activation}",
                ha='center', va='center', fontsize=9)

    # Выходной слой
    x_out = 1.5 + len(hidden_layers)
    height_out = output_dim * scale
    rect_out = patches.Rectangle((x_out - box_width / 2, 1), box_width, height_out,
                                 linewidth=2, edgecolor='black', facecolor=color_output)
    ax.add_patch(rect_out)
    outputs = ["u", "v", "p"] if output_dim == 3 else ["u", "v", "p", "ρ"]
    ax.text(x_out, 1 + height_out / 2, f"Output\n" + ", ".join(outputs),
            ha='center', va='center', fontsize=9)

    plt.title(f"Схема сети: {config.get('project_name', 'unnamed')}", fontsize=14)

    os.makedirs(PICTURES_DIR, exist_ok=True)
    path = os.path.join(PICTURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Схема сети сохранена: {path}")


def run_experiment(config_path: str, experiment_dir: str) -> None:
    """
    Запускает обучение с заданной конфигурацией и сохраняет результаты в experiment_dir.
    Используется для пакетного запуска.

    Args:
        config_path: путь к файлу конфигурации (.yaml)
        experiment_dir: директория для сохранения результатов
    """
    cfg = ConfigManager()
    try:
        # Загружаем конфигурацию напрямую из файла
        cfg.load_from_file(config_path)
    except Exception as e:
        logger.error(f"Не удалось загрузить конфигурацию {config_path}: {e}")
        sys.exit(1)

    # Проверяем наличие файла с точками
    points_file = cfg.config['data'].get('points_file')
    if not points_file or not os.path.exists(points_file):
        logger.error(f"Файл с точками не найден: {points_file}")
        sys.exit(1)

    # Создаём solver с указанием директории эксперимента
    solver = PINNSolver(cfg.config, experiment_dir=experiment_dir)
    solver.train()


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


def architecture_menu(cfg: ConfigManager) -> None:
    """Меню настройки архитектуры сети."""
    while True:
        clear_screen()
        print_submenu_header("Архитектура сети")
        arch = cfg.config['architecture']
        print("1. Выбор входных параметров нейронной сети")
        print("2. Настройка скрытых слоёв")
        print(f"3. Функция активации (текущая: {arch.get('activation', 'tanh')})")
        print("4. Настройка выходных параметров")
        print("5. Визуализация сети (схема)")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            clear_screen()
            print_submenu_header("Входные параметры")
            inp = cfg.config['input_params']
            inp['inlet_velocity'] = yes_no("Использовать компоненты скорости на входе", inp['inlet_velocity'])
            inp['viscosity'] = yes_no("Использовать вязкость жидкости", inp['viscosity'])
            inp['density'] = yes_no("Использовать плотность жидкости", inp['density'])
            if inp['viscosity'] or inp['density']:
                fluids = [f.replace('.json', '') for f in os.listdir(ConfigManager.FLUIDS_DIR) if f.endswith('.json')]
                if fluids:
                    print("Доступные жидкости:")
                    for i, fl in enumerate(fluids, 1):
                        print(f"{i}. {fl}")
                    ch = input("Выберите номер или имя файла: ").strip()
                    try:
                        idx = int(ch) - 1
                        inp['fluid_file'] = fluids[idx]
                    except ValueError:
                        inp['fluid_file'] = ch
                else:
                    print("В папке Fluids нет JSON-файлов. Поместите файлы туда.")
                    inp['fluid_file'] = None
            print_submenu_footer()
            input("Нажмите Enter...")
        elif sub == '2':
            layers_str = input(f"Количество нейронов в скрытых слоях через запятую (текущее: {arch['hidden_layers']}): ").strip()
            if layers_str:
                arch['hidden_layers'] = [int(x) for x in layers_str.split(',')]
        elif sub == '3':
            print("Доступные функции активации:")
            activations = ['tanh', 'relu', 'sigmoid', 'leaky_relu', 'elu', 'swish', 'siren']
            for i, act in enumerate(activations, 1):
                print(f"{i}. {act}")
            choice = input("Выберите номер: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(activations):
                    arch['activation'] = activations[idx]
                else:
                    print("Неверный номер.")
            except ValueError:
                print("Введите число.")
            input("Нажмите Enter...")
        elif sub == '4':
            print("Выходные параметры: 1 - только u,v,p (несжимаемая); 2 - добавить плотность (сжимаемая)")
            out_choice = input("Выберите (1/2): ").strip()
            if out_choice == '2':
                arch['output_dim'] = 4
            else:
                arch['output_dim'] = 3
            bias_u = input(f"Начальное смещение для U (текущее: {arch['output_bias'][0]}): ").strip()
            bias_v = input(f"Начальное смещение для V (текущее: {arch['output_bias'][1]}): ").strip()
            bias_p = input(f"Начальное смещение для P (текущее: {arch['output_bias'][2]}): ").strip()
            arch['output_bias'] = (
                float(bias_u) if bias_u else arch['output_bias'][0],
                float(bias_v) if bias_v else arch['output_bias'][1],
                float(bias_p) if bias_p else arch['output_bias'][2],
            )
        elif sub == '5':
            filename = f"{cfg.name}_network.png"
            visualize_network(cfg.config, filename)
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def loss_function_menu(cfg: ConfigManager) -> None:
    """Меню настройки функции потерь."""
    while True:
        clear_screen()
        print_submenu_header("Настройка функции потерь")
        print("1. Выбор системы ДУ")
        print("2. Выбор типа ГУ (жёсткие/мягкие)")
        print("3. Настройка весов сети")
        print("4. Вывод текущих настроек функции потерь")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            print("Система уравнений: 1 - вязкая несжимаемая, 2 - Эйлер (идеальная)")
            sys_choice = input("Выберите (1/2): ").strip()
            cfg.config['pde']['system'] = 'incompressible_navier_stokes' if sys_choice == '1' else 'euler'
        elif sub == '2':
            bc_type = input("Тип граничных условий (soft/hard, по умолч. soft): ").strip().lower()
            cfg.config['boundary_conditions']['type'] = bc_type if bc_type in ['hard'] else 'soft'
        elif sub == '3':
            weights_menu(cfg)
        elif sub == '4':
            lw = cfg.config['loss_weights']
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ФУНКЦИИ ПОТЕРЬ ---")
            print(f"Система ДУ: {cfg.config['pde']['system']}")
            print(f"Тип ГУ: {cfg.config['boundary_conditions']['type']}")
            print("Веса PDE:")
            for k, v in lw['pde_weights'].items():
                print(f"  {k}: {v:.2e}")
            print("Веса BC:")
            for k, v in lw['bc_weights'].items():
                print(f"  {k}: {v:.2e}")
            print(f"Адаптивные BC: {lw['adaptive_bc']}")
            print(f"SA-PINN: {lw['use_sa_weights']}")
            print(f"SA-PINN для BC: {lw['sa_apply_to_bc']}")
            print(f"Интегральная потеря: {lw['integral_condition']}, вес: {lw['integral_weight']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def weights_menu(cfg: ConfigManager) -> None:
    """Меню настройки весов функции потерь."""
    while True:
        clear_screen()
        print_submenu_header("Настройка весов сети")
        lw = cfg.config['loss_weights']
        print("1. Ручная настройка весов функции потерь")
        print(f"2. Адаптивные веса для BC (текущ: {lw['adaptive_bc']})")
        print(f"3. SA-PINN (текущ: {lw['use_sa_weights']})")
        print(f"4. SA-PINN для BC (текущ: {lw['sa_apply_to_bc']})")
        print(f"5. Интегральная потеря (текущ: {lw['integral_condition']}, вес: {lw['integral_weight']})")
        print("6. Вывод текущих настроек")
        print("0. Назад")
        ch = input("Выберите: ").strip()

        if ch == '1':
            for k in lw['pde_weights']:
                val = input(f"Вес PDE {k} [{lw['pde_weights'][k]}]: ").strip()
                if val:
                    lw['pde_weights'][k] = float(val)
            for k in lw['bc_weights']:
                val = input(f"Вес BC {k} [{lw['bc_weights'][k]}]: ").strip()
                if val:
                    lw['bc_weights'][k] = float(val)
        elif ch == '2':
            lw['adaptive_bc'] = not lw['adaptive_bc']
        elif ch == '3':
            lw['use_sa_weights'] = not lw['use_sa_weights']
        elif ch == '4':
            lw['sa_apply_to_bc'] = not lw['sa_apply_to_bc']
        elif ch == '5':
            lw['integral_condition'] = not lw['integral_condition']
            if lw['integral_condition']:
                w = input(f"Вес интегральной потери [{lw['integral_weight']}]: ").strip()
                if w:
                    lw['integral_weight'] = float(w)
        elif ch == '6':
            print("\n--- ТЕКУЩИЕ ВЕСА ---")
            print("PDE weights:", lw['pde_weights'])
            print("BC weights:", lw['bc_weights'])
            print(f"adaptive_bc: {lw['adaptive_bc']}")
            print(f"use_sa_weights: {lw['use_sa_weights']}")
            print(f"sa_apply_to_bc: {lw['sa_apply_to_bc']}")
            print(f"integral_condition: {lw['integral_condition']}, weight: {lw['integral_weight']}")
            input("Нажмите Enter...")
        elif ch == '0':
            print_submenu_footer()
            break


def training_strategy_menu(cfg: ConfigManager) -> None:
    """Меню настройки стратегии обучения."""
    while True:
        clear_screen()
        print_submenu_header("Настройка стратегии обучения")
        tr = cfg.config['training']
        print("1. Двухэтапное обучение")
        print("2. Изменяющийся LR")
        print("3. Период сброса оптимизатора")
        print("4. Отсечение градиентов")
        print("5. RAR")
        print("6. Переключение Adam → L-BFGS")
        print("7. Размер батча и градиентное накопление")
        print("8. Количество эпох")
        print("9. Вывод текущей стратегии")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            tr['two_stage'] = yes_no("Двухэтапное обучение", tr['two_stage'])
            if tr['two_stage']:
                tr['stage1_epochs'] = int(input(f"Эпох только BC [{tr['stage1_epochs']}]: ") or tr['stage1_epochs'])
                tr['stage2_pde_weight_start'] = float(input(f"Начальный вес PDE [{tr['stage2_pde_weight_start']}]: ") or tr['stage2_pde_weight_start'])
                tr['stage2_pde_weight_end'] = float(input(f"Конечный вес PDE [{tr['stage2_pde_weight_end']}]: ") or tr['stage2_pde_weight_end'])
                tr['stage2_ramp_epochs'] = int(input(f"Эпох нарастания [{tr['stage2_ramp_epochs']}]: ") or tr['stage2_ramp_epochs'])
        elif sub == '2':
            print("Тип изменения LR: 0 - постоянный, 1 - периодический, 2 - по детектору")
            lr_type = input("Выберите (0/1/2): ").strip()
            if lr_type == '1':
                tr['lr_schedule'] = 'cyclic'
                tr['lr_cyclic_period'] = int(input(f"Период (эпох) [{tr['lr_cyclic_period']}]: ") or tr['lr_cyclic_period'])
                tr['lr_cyclic_amplitude'] = float(input(f"Амплитуда [{tr['lr_cyclic_amplitude']}]: ") or tr['lr_cyclic_amplitude'])
            elif lr_type == '2':
                tr['lr_schedule'] = 'detector'
                tr['lr_detector_factor'] = float(input(f"Коэффициент увеличения [{tr['lr_detector_factor']}]: ") or tr['lr_detector_factor'])
                tr['lr_detector_patience'] = int(input(f"Терпение (эпох) [{tr['lr_detector_patience']}]: ") or tr['lr_detector_patience'])
                tr['lr_detector_recovery'] = int(input(f"Эпох восстановления [{tr['lr_detector_recovery']}]: ") or tr['lr_detector_recovery'])
            else:
                tr['lr_schedule'] = 'constant'
        elif sub == '3':
            tr['reset_interval'] = int(input(f"Интервал сброса оптимизатора (0 - выкл) [{tr['reset_interval']}]: ") or tr['reset_interval'])
            if tr['reset_interval'] > 0:
                tr['reset_noise'] = float(input(f"Амплитуда шума [{tr['reset_noise']}]: ") or tr['reset_noise'])
        elif sub == '4':
            clip = input(f"Градиентное отсечение (Enter - выкл, текущ: {tr['grad_clip']}): ").strip()
            tr['grad_clip'] = float(clip) if clip else None
        elif sub == '5':
            tr['rar_enable'] = yes_no("RAR", tr['rar_enable'])
            if tr['rar_enable']:
                tr['rar_interval'] = int(input(f"Интервал RAR [{tr['rar_interval']}]: ") or tr['rar_interval'])
                tr['rar_num_points'] = int(input(f"Количество точек [{tr['rar_num_points']}]: ") or tr['rar_num_points'])
        elif sub == '6':
            tr['switch_to_lbfgs_epoch'] = int(input(f"Переключение на L-BFGS (эпоха, 0 - выкл) [{tr['switch_to_lbfgs_epoch']}]: ") or tr['switch_to_lbfgs_epoch'])
            if tr['switch_to_lbfgs_epoch'] > 0:
                tr['lbfgs_max_iter'] = int(input(f"max_iter L-BFGS [{tr['lbfgs_max_iter']}]: ") or tr['lbfgs_max_iter'])
                tr['lbfgs_history_size'] = int(input(f"history_size [{tr['lbfgs_history_size']}]: ") or tr['lbfgs_history_size'])
        elif sub == '7':
            tr['batch_size'] = int(input(f"Размер батча [{tr['batch_size']}]: ") or tr['batch_size'])
            tr['gradient_accumulation_steps'] = int(input(f"Шаги градиентного накопления [{tr['gradient_accumulation_steps']}]: ") or tr['gradient_accumulation_steps'])
        elif sub == '8':
            tr['epochs'] = int(input(f"Количество эпох [{tr['epochs']}]: ") or tr['epochs'])
        elif sub == '9':
            print("\n--- ТЕКУЩАЯ СТРАТЕГИЯ ОБУЧЕНИЯ ---")
            print(f"Двухэтапное обучение: {tr['two_stage']}")
            if tr['two_stage']:
                print(f"  Stage1 эпох: {tr['stage1_epochs']}")
                print(f"  Stage2 вес PDE от {tr['stage2_pde_weight_start']} до {tr['stage2_pde_weight_end']} за {tr['stage2_ramp_epochs']} эпох")
            print(f"LR schedule: {tr['lr_schedule']}")
            print(f"Интервал сброса: {tr['reset_interval']} (шум {tr['reset_noise']})")
            print(f"Градиентное отсечение: {tr['grad_clip']}")
            print(f"RAR: {tr['rar_enable']}")
            print(f"Переключение на L-BFGS: эпоха {tr['switch_to_lbfgs_epoch']}")
            print(f"Батч: {tr['batch_size']}, накопление: {tr['gradient_accumulation_steps']}")
            print(f"Эпох: {tr['epochs']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


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
        print("3. Построение профилей скорости")
        print("4. Сечения для профилей (через запятую)")
        print("5. Построение спектра (1D)")
        print("6. Координата X для спектра (1D)")
        print("7. Построение 2D спектра")
        print("8. Разрешение 2D спектра")
        print("9. Вычисление раздельных градиентов")
        print("10. Вычисление NTK")
        print("11. Сохранение истории RAR")
        print("12. Показать текущие настройки")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            diag['interval'] = int(input(f"Интервал (эпох) [{diag.get('interval', 50)}]: ") or diag.get('interval', 50))
        elif sub == '2':
            diag['plot_residual_map'] = yes_no("Строить карту невязки", diag.get('plot_residual_map', True))
        elif sub == '3':
            diag['plot_profiles'] = yes_no("Строить профили скорости", diag.get('plot_profiles', True))
        elif sub == '4':
            current = diag.get('profile_sections', [0.2, 0.5, 0.8])
            sections_str = input(f"Сечения (x, м) через запятую [{', '.join(map(str, current))}]: ").strip()
            if sections_str:
                try:
                    diag['profile_sections'] = [float(x.strip()) for x in sections_str.split(',')]
                except ValueError:
                    print("Неверный формат, оставлено текущее значение.")
        elif sub == '5':
            diag['plot_spectrum'] = yes_no("Строить 1D спектр", diag.get('plot_spectrum', True))
        elif sub == '6':
            diag['spectrum_section_x'] = float(input(f"X для 1D спектра (м) [{diag.get('spectrum_section_x', 0.5)}]: ") or diag.get('spectrum_section_x', 0.5))
        elif sub == '7':
            diag['plot_2d_spectrum'] = yes_no("Строить 2D спектр", diag.get('plot_2d_spectrum', False))
        elif sub == '8':
            diag['spectrum_2d_resolution'] = int(input(f"Разрешение 2D спектра [{diag.get('spectrum_2d_resolution', 256)}]: ") or diag.get('spectrum_2d_resolution', 256))
        elif sub == '9':
            diag['compute_grad_cosine'] = yes_no("Вычислять раздельные градиенты", diag.get('compute_grad_cosine', False))
        elif sub == '10':
            diag['compute_ntk_condition'] = yes_no("Вычислять NTK", diag.get('compute_ntk_condition', False))
        elif sub == '11':
            diag['save_rar_history'] = yes_no("Сохранять историю RAR", diag.get('save_rar_history', True))
        elif sub == '12':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ДИАГНОСТИКИ ---")
            print(f"Интервал: {diag.get('interval', 50)}")
            print(f"Карта невязки: {diag.get('plot_residual_map', True)}")
            print(f"Профили: {diag.get('plot_profiles', True)}")
            print(f"Сечения: {diag.get('profile_sections', [0.2, 0.5, 0.8])}")
            print(f"1D спектр: {diag.get('plot_spectrum', True)}")
            print(f"X 1D спектра: {diag.get('spectrum_section_x', 0.5)}")
            print(f"2D спектр: {diag.get('plot_2d_spectrum', False)}")
            print(f"Разрешение 2D спектра: {diag.get('spectrum_2d_resolution', 256)}")
            print(f"Раздельные градиенты: {diag.get('compute_grad_cosine', False)}")
            print(f"NTK: {diag.get('compute_ntk_condition', False)}")
            print(f"История RAR: {diag.get('save_rar_history', True)}")
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


def start_training(cfg: ConfigManager, experiment_dir: Optional[str] = None) -> None:
    """
    Запуск обучения с текущей конфигурацией.

    Args:
        cfg: объект ConfigManager с загруженной конфигурацией
        experiment_dir: если указан, результаты сохраняются в эту директорию
    """
    if not cfg.config['data'].get('points_file'):
        cfg.config['data']['points_file'] = DEFAULT_POINTS_FILE
        print(f"Используется путь к точкам по умолчанию: {DEFAULT_POINTS_FILE}")

    if not os.path.exists(cfg.config['data']['points_file']):
        print(f"Файл с точками не найден: {cfg.config['data']['points_file']}")
        alt = input("Введите путь к файлу вручную (или Enter для отмены): ").strip()
        if alt and os.path.exists(alt):
            cfg.config['data']['points_file'] = alt
        else:
            print("Файл не найден. Обучение невозможно.")
            input("Нажмите Enter...")
            return

    # Убедимся, что есть раздел boundary_conditions (для совместимости)
    if 'boundary_conditions' not in cfg.config:
        cfg.config['boundary_conditions'] = {'type': 'soft'}
        logger.warning("В конфигурации отсутствует раздел 'boundary_conditions', установлен 'soft' по умолчанию.")

    if experiment_dir is None:
        # Интерактивный режим: показываем конфигурацию и запрашиваем подтверждение
        clear_screen()
        print("=" * 60)
        print("ЗАПУСК ОБУЧЕНИЯ".center(60))
        print("=" * 60)
        cfg.show()
        print("\nНажмите Enter для начала обучения или Ctrl+C для отмены...")
        input()

    # Создаём solver (передаём experiment_dir, если есть)
    solver = PINNSolver(cfg.config, experiment_dir=experiment_dir)
    solver.train()


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