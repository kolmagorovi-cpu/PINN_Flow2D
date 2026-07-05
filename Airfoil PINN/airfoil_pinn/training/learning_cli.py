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
from airfoil_pinn.config.manager import ConfigManager, warn_activation, yes_no
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.training.network_diagram import visualize_network
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


# =============================================================================
# Вспомогательные функции для настройки архитектуры
# =============================================================================

def _configure_input_params(cfg: ConfigManager) -> None:
    """Настройка входных параметров нейронной сети."""
    clear_screen()
    print_submenu_header("Входные параметры")
    inp = cfg.config['input_params']

    inp['coordinates'] = yes_no("Использовать координаты x,y", inp.get('coordinates', True))
    inp['sdf'] = yes_no("Использовать SDF", inp.get('sdf', True))
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


def _configure_hidden_layers(cfg: ConfigManager) -> None:
    """Настройка скрытых слоёв."""
    arch = cfg.config['architecture']
    layers_str = input(f"Количество нейронов в скрытых слоях через запятую (текущее: {arch['hidden_layers']}): ").strip()
    if layers_str:
        arch['hidden_layers'] = [int(x) for x in layers_str.split(',')]


def _configure_activation(cfg: ConfigManager) -> None:
    """Выбор функции активации."""
    arch = cfg.config['architecture']

    print("Доступные функции активации:")
    activations = ['tanh', 'relu', 'sigmoid', 'leaky_relu', 'elu', 'swish', 'siren']
    for i, act in enumerate(activations, 1):
        print(f"{i}. {act}")

    choice = input("Выберите номер: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(activations):
            selected_activation = activations[idx]
            x_min = arch.get('coordinate_scaling', {}).get('input_min', -6.0)
            x_max = arch.get('coordinate_scaling', {}).get('input_max', 6.0)
            enabled, scale_factor = warn_activation(selected_activation, x_min, x_max)

            arch['activation'] = selected_activation
            arch['coordinate_scaling']['enabled'] = enabled
            arch['coordinate_scaling']['scale_factor'] = scale_factor
            arch['coordinate_scaling']['input_min'] = x_min
            arch['coordinate_scaling']['input_max'] = x_max
        else:
            print("Неверный номер.")
    except ValueError:
        print("Введите число.")
    input("Нажмите Enter...")


def _configure_coordinate_scaling(cfg: ConfigManager) -> None:
    """Меню настройки масштабирования координат."""
    arch = cfg.config['architecture']
    scaling = arch.get('coordinate_scaling', {})

    while True:
        clear_screen()
        print_submenu_header("Масштабирование координат")

        print(f"Включено: {scaling.get('enabled', False)}")
        print(f"scale_factor: {scaling.get('scale_factor', 1.0):.4f}")
        print(f"Диапазон координат: [{scaling.get('input_min', -6.0):.2f}, {scaling.get('input_max', 6.0):.2f}]")
        print("\n1. Включить/выключить масштабирование")
        print("2. Изменить scale_factor")
        print("3. Изменить диапазон координат (input_min, input_max)")
        print("0. Выход")

        scale_sub = input("Выберите пункт: ").strip()

        if scale_sub == '1':
            scaling['enabled'] = not scaling.get('enabled', False)
            print(f"✅ Масштабирование {'включено' if scaling['enabled'] else 'выключено'}")
        elif scale_sub == '2':
            try:
                new_scale = input(f"Введите scale_factor (текущее: {scaling.get('scale_factor', 1.0):.4f}): ").strip()
                scaling['scale_factor'] = float(new_scale)
                print(f"✅ scale_factor установлен: {scaling['scale_factor']:.4f}")
            except ValueError:
                print("⚠️ Введите корректное число!")
        elif scale_sub == '3':
            try:
                new_min = input(f"Введите input_min (текущее: {scaling.get('input_min', -6.0):.2f}): ").strip()
                new_max = input(f"Введите input_max (текущее: {scaling.get('input_max', 6.0):.2f}): ").strip()
                scaling['input_min'] = float(new_min)
                scaling['input_max'] = float(new_max)
                print(f"✅ Диапазон установлен: [{scaling['input_min']:.2f}, {scaling['input_max']:.2f}]")
            except ValueError:
                print("⚠️ Введите корректные числа!")
        elif scale_sub == '0':
            break
        input("Нажмите Enter...")


def _configure_output_params(cfg: ConfigManager) -> None:
    """Настройка выходных параметров."""
    arch = cfg.config['architecture']

    print("Выходные параметры: 1 - только u,v,p (несжимаемая); 2 - добавить плотность (сжимаемая)")
    out_choice = input("Выберите (1/2): ").strip()
    arch['output_dim'] = 4 if out_choice == '2' else 3

    bias_u = input(f"Начальное смещение для U (текущее: {arch['output_bias'][0]}): ").strip()
    bias_v = input(f"Начальное смещение для V (текущее: {arch['output_bias'][1]}): ").strip()
    bias_p = input(f"Начальное смещение для P (текущее: {arch['output_bias'][2]}): ").strip()

    arch['output_bias'] = (
        float(bias_u) if bias_u else arch['output_bias'][0],
        float(bias_v) if bias_v else arch['output_bias'][1],
        float(bias_p) if bias_p else arch['output_bias'][2],
    )


# =============================================================================


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
    arch = cfg.config['architecture']

    # Отображаем меню
    clear_screen()
    print_submenu_header("Архитектура сети")
    print("1. Выбор входных параметров нейронной сети")
    print("2. Настройка скрытых слоёв")
    print(f"3. Функция активации (текущая: {arch.get('activation', 'tanh')})")
    print(f"4. Масштабирование координат (включено: {arch.get('coordinate_scaling', {}).get('enabled', False)}, scale_factor: {arch.get('coordinate_scaling', {}).get('scale_factor', 1.0):.4f})")
    print("5. Настройка выходных параметров")
    print("6. Визуализация сети (схема)")
    print("0. Выход")
    sub = input("Выберите пункт: ").strip()

    if sub == '0':
        print_submenu_footer()
        return

    # Обработка выбора
    if sub == '1':
        _configure_input_params(cfg)
    elif sub == '2':
        _configure_hidden_layers(cfg)
    elif sub == '3':
        _configure_activation(cfg)
    elif sub == '4':
        _configure_coordinate_scaling(cfg)
    elif sub == '5':
        _configure_output_params(cfg)
    elif sub == '6':
        filename = f"{cfg.name}_network.png"
        visualize_network(cfg.config, filename)
        input("Нажмите Enter...")


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
            fvm_cfg = lw.get('fvm', {})
            fvm_on = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
            fvm_w = fvm_cfg.get('weight', 1.0) if isinstance(fvm_cfg, dict) else float(fvm_cfg)
            fvm_bi = fvm_cfg.get('batch_interval', 1) if isinstance(fvm_cfg, dict) else 1
            print(f"FVM Mass flow: {fvm_on}, вес: {fvm_w}, batch_interval: {fvm_bi}")
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
        fvm_cfg = lw.setdefault('fvm', {'enabled': False, 'weight': 1.0, 'batch_interval': 1})
        if not isinstance(fvm_cfg, dict):
            fvm_cfg = {'enabled': bool(fvm_cfg), 'weight': float(fvm_cfg) if float(fvm_cfg) > 0 else 1.0, 'batch_interval': 1}
            lw['fvm'] = fvm_cfg
        fvm_cfg.setdefault('batch_interval', 1)
        fvm_enabled = bool(fvm_cfg.get('enabled', False))
        fvm_weight = fvm_cfg.get('weight', 1.0)
        fvm_bi = fvm_cfg.get('batch_interval', 1)
        print(f"5. Интегральная потеря (текущ: {lw['integral_condition']}, вес: {lw['integral_weight']})")
        print(f"6. FVM Mass flow (включено: {fvm_enabled}, вес: {fvm_weight}, каждые {fvm_bi} батч.)")
        print("7. Вывод текущих настроек")
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
            fvm_cfg['enabled'] = not fvm_cfg.get('enabled', False)
            if fvm_cfg['enabled']:
                w = input(f"Вес FVM Mass flow [{fvm_cfg.get('weight', 1.0)}]: ").strip()
                if w:
                    try:
                        fvm_cfg['weight'] = float(w)
                    except ValueError:
                        print("Неверное значение, оставлено текущее.")
                bi = input(f"FVM раз в K батчей (1=каждый батч) [{fvm_cfg.get('batch_interval', 1)}]: ").strip()
                if bi:
                    try:
                        val = int(bi)
                        fvm_cfg['batch_interval'] = max(1, val)
                    except ValueError:
                        print("Неверное значение, оставлено текущее.")
            print(f"FVM Mass flow: {'включено' if fvm_cfg['enabled'] else 'выключено'}, "
                  f"вес: {fvm_cfg.get('weight', 1.0)}, каждые {fvm_cfg.get('batch_interval', 1)} батч.")
        elif ch == '7':
            fvm_cfg = lw.get('fvm', {})
            fvm_on = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
            fvm_w = fvm_cfg.get('weight', 1.0) if isinstance(fvm_cfg, dict) else float(fvm_cfg)
            fvm_bi = fvm_cfg.get('batch_interval', 1) if isinstance(fvm_cfg, dict) else 1
            print("\n--- ТЕКУЩИЕ ВЕСА ---")
            print("PDE weights:", lw['pde_weights'])
            print("BC weights:", lw['bc_weights'])
            print(f"adaptive_bc: {lw['adaptive_bc']}")
            print(f"use_sa_weights: {lw['use_sa_weights']}")
            print(f"sa_apply_to_bc: {lw['sa_apply_to_bc']}")
            print(f"integral_condition: {lw['integral_condition']}, weight: {lw['integral_weight']}")
            print(f"fvm_mass_flow: {fvm_on}, weight: {fvm_w}, batch_interval: {fvm_bi}")
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
        print("9. Early stopping")
        print("10. Вывод текущей стратегии")
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
                tr.setdefault('rar_mode', 'adaptive')
                tr.setdefault('rar_score_mode', 'total_loss')
                tr.setdefault('rar_error_quantile', 0.75)
                tr.setdefault('rar_jitter_scale', 0.85)
                tr.setdefault('rar_min_distance_factor', 0.35)
                tr.setdefault('rar_rebuild_fvm', True)
                tr.setdefault('rar_fvm_rebuild_every', 1)
                tr.setdefault('rar_fvm_min_new_points', 1)
                tr.setdefault('rar_save_mesh_volume_plot', True)
                tr.setdefault('rar_mesh_plot_dpi', 150)
                tr['rar_interval'] = int(input(f"Интервал RAR [{tr['rar_interval']}]: ") or tr['rar_interval'])
                tr['rar_num_points'] = int(input(f"Количество точек [{tr['rar_num_points']}]: ") or tr['rar_num_points'])
                mode = input(f"Режим RAR (adaptive/legacy) [{tr['rar_mode']}]: ").strip().lower()
                if mode in ('adaptive', 'legacy'):
                    tr['rar_mode'] = mode
                if tr['rar_mode'] == 'adaptive':
                    score_mode = input(
                        f"Score для выбора RAR-точек (total_loss/pde) [{tr['rar_score_mode']}]: "
                    ).strip().lower()
                    if score_mode in ('total_loss', 'pde'):
                        tr['rar_score_mode'] = score_mode
                    tr['rar_error_quantile'] = float(input(
                        f"Квантиль ошибки для источников [{tr['rar_error_quantile']}]: "
                    ) or tr['rar_error_quantile'])
                    tr['rar_jitter_scale'] = float(input(
                        f"Радиус генерации в локальных шагах [{tr['rar_jitter_scale']}]: "
                    ) or tr['rar_jitter_scale'])
                    tr['rar_min_distance_factor'] = float(input(
                        f"Мин. расстояние до соседей в локальных шагах [{tr['rar_min_distance_factor']}]: "
                    ) or tr['rar_min_distance_factor'])
                    flow_rar = tr.setdefault('flow_rar', {})
                    flow_rar.setdefault('enabled', True)
                    flow_rar.setdefault('num_points', None)
                    flow_rar.setdefault('exclude_sdf_delta', None)
                    boundary_rar = tr.setdefault('boundary_rar', {})
                    boundary_rar.setdefault('enabled', False)
                    boundary_rar.setdefault('num_points', None)
                    boundary_rar.setdefault('fraction', 0.4)
                    boundary_rar.setdefault('delta_mode', 'estimated')
                    boundary_rar.setdefault('delta_max', 5.0)
                    boundary_rar.setdefault(
                        'eta_levels',
                        [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    )
                    boundary_rar.setdefault('contour_samples', 400)
                    boundary_rar.setdefault('alpha', 1.5)
                    boundary_rar.setdefault('uniform_contour_weight', 0.5)
                    boundary_rar.setdefault('min_distance_factor', 0.02)
                    boundary_rar.setdefault('score_batch_size', 2048)
                    boundary_rar.setdefault(
                        'error_weights',
                        {'cont': 1.0, 'mom_x': 3.0, 'mom_y': 3.0, 'mom_r': 3.0, 'mom_z': 3.0},
                    )

                    flow_rar['enabled'] = yes_no("Flow RAR для свободного потока", flow_rar.get('enabled', True))
                    excl_current = flow_rar.get('exclude_sdf_delta', None)
                    excl = input(
                        f"Исключать из Flow RAR зону sdf/delta < value (Enter - авто/выкл, текущ: {excl_current}): "
                    ).strip()
                    if excl:
                        flow_rar['exclude_sdf_delta'] = float(excl)

                    boundary_rar['enabled'] = yes_no(
                        "Boundary-Layer RAR для пограничного слоя",
                        boundary_rar.get('enabled', False),
                    )
                    if boundary_rar['enabled']:
                        n_boundary = input(
                            f"Boundary RAR точек (Enter - fraction от rar_num_points, текущ: {boundary_rar.get('num_points')}): "
                        ).strip()
                        boundary_rar['num_points'] = int(n_boundary) if n_boundary else None
                        if boundary_rar['num_points'] is None:
                            boundary_rar['fraction'] = float(input(
                                f"Boundary RAR fraction [{boundary_rar.get('fraction', 0.4)}]: "
                            ) or boundary_rar.get('fraction', 0.4))
                        boundary_rar['delta_max'] = float(input(
                            f"Максимальная высота Boundary RAR sdf/delta [{boundary_rar.get('delta_max', 5.0)}]: "
                        ) or boundary_rar.get('delta_max', 5.0))
                        boundary_rar['contour_samples'] = int(input(
                            f"Сэмплов по контуру профиля [{boundary_rar.get('contour_samples', 400)}]: "
                        ) or boundary_rar.get('contour_samples', 400))
                        boundary_rar['alpha'] = float(input(
                            f"Степень вероятности по ошибке alpha [{boundary_rar.get('alpha', 1.5)}]: "
                        ) or boundary_rar.get('alpha', 1.5))
                        boundary_rar['uniform_contour_weight'] = float(input(
                            "Доля равномерности по контуру "
                            f"[{boundary_rar.get('uniform_contour_weight', 0.5)}]: "
                        ) or boundary_rar.get('uniform_contour_weight', 0.5))
                        boundary_rar['min_distance_factor'] = float(input(
                            "Мин. расстояние Boundary RAR в долях delta "
                            f"[{boundary_rar.get('min_distance_factor', 0.02)}]: "
                        ) or boundary_rar.get('min_distance_factor', 0.02))
                    tr['rar_rebuild_fvm'] = yes_no("Перестраивать FVM после adaptive RAR", tr['rar_rebuild_fvm'])
                    if tr['rar_rebuild_fvm']:
                        tr['rar_fvm_rebuild_every'] = int(input(
                            f"FVM rebuild раз в N срабатываний RAR [{tr['rar_fvm_rebuild_every']}]: "
                        ) or tr['rar_fvm_rebuild_every'])
                        tr['rar_fvm_min_new_points'] = int(input(
                            f"Мин. новых точек для FVM rebuild [{tr['rar_fvm_min_new_points']}]: "
                        ) or tr['rar_fvm_min_new_points'])
                    tr['rar_save_mesh_volume_plot'] = yes_no(
                        "Сохранять график CV-объёмов сетки после RAR",
                        tr['rar_save_mesh_volume_plot'],
                    )
                    if tr['rar_save_mesh_volume_plot']:
                        tr['rar_mesh_plot_dpi'] = int(input(
                            f"DPI графика CV-объёмов [{tr['rar_mesh_plot_dpi']}]: "
                        ) or tr['rar_mesh_plot_dpi'])
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
            es = tr.setdefault('early_stopping', {
                'enabled': False,
                'patience': 100,
                'min_delta': 1e-6,
                'restore_best': True,
            })
            es['enabled'] = yes_no("Early stopping", es.get('enabled', False))
            if es['enabled']:
                es['patience'] = int(input(f"Patience [{es.get('patience', 100)}]: ") or es.get('patience', 100))
                es['min_delta'] = float(input(f"Min delta [{es.get('min_delta', 1e-6)}]: ") or es.get('min_delta', 1e-6))
                es['restore_best'] = yes_no("Восстановить лучшую модель", es.get('restore_best', True))
        elif sub == '10':
            print("\n--- ТЕКУЩАЯ СТРАТЕГИЯ ОБУЧЕНИЯ ---")
            print(f"Двухэтапное обучение: {tr['two_stage']}")
            if tr['two_stage']:
                print(f"  Stage1 эпох: {tr['stage1_epochs']}")
                print(f"  Stage2 вес PDE от {tr['stage2_pde_weight_start']} до {tr['stage2_pde_weight_end']} за {tr['stage2_ramp_epochs']} эпох")
            print(f"LR schedule: {tr['lr_schedule']}")
            print(f"Интервал сброса: {tr['reset_interval']} (шум {tr['reset_noise']})")
            print(f"Градиентное отсечение: {tr['grad_clip']}")
            print(f"RAR: {tr['rar_enable']}")
            if tr['rar_enable']:
                print(f"  режим: {tr.get('rar_mode', 'adaptive')}, "
                      f"интервал: {tr['rar_interval']}, точек: {tr['rar_num_points']}")
                if tr.get('rar_mode', 'adaptive') == 'adaptive':
                    print(f"  score: {tr.get('rar_score_mode', 'total_loss')}, "
                          f"error_quantile: {tr.get('rar_error_quantile', 0.75)}, "
                          f"FVM rebuild: {tr.get('rar_rebuild_fvm', True)}")
                    print(f"  mesh CV plot: {tr.get('rar_save_mesh_volume_plot', True)}, "
                          f"dpi={tr.get('rar_mesh_plot_dpi', 150)}")
                    flow_rar = tr.get('flow_rar', {}) or {}
                    boundary_rar = tr.get('boundary_rar', {}) or {}
                    print(
                        "  Flow RAR: "
                        f"enabled={flow_rar.get('enabled', True)}, "
                        f"num_points={flow_rar.get('num_points', None)}, "
                        f"exclude_sdf_delta={flow_rar.get('exclude_sdf_delta', None)}"
                    )
                    print(
                        "  Boundary RAR: "
                        f"enabled={boundary_rar.get('enabled', False)}, "
                        f"num_points={boundary_rar.get('num_points', None)}, "
                        f"fraction={boundary_rar.get('fraction', 0.4)}, "
                        f"delta_max={boundary_rar.get('delta_max', 5.0)}, "
                        f"min_distance_factor={boundary_rar.get('min_distance_factor', 0.02)}"
                    )
            print(f"Переключение на L-BFGS: эпоха {tr['switch_to_lbfgs_epoch']}")
            print(f"Батч: {tr['batch_size']}, накопление: {tr['gradient_accumulation_steps']}")
            print(f"Эпох: {tr['epochs']}")
            es = tr.get('early_stopping', {})
            print(f"Early stopping: {es.get('enabled', False)} "
                  f"(patience={es.get('patience', 100)}, min_delta={es.get('min_delta', 1e-6)}, "
                  f"restore_best={es.get('restore_best', True)})")
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
