#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню настройки архитектуры PINN.

Модуль содержит интерактивные действия, которые меняют раздел ``architecture``
и связанные с ним входные параметры конфигурации. ``learning_cli`` только
вызывает ``architecture_menu`` и не хранит детали этого подменю.
"""

import os

from airfoil_pinn.config.manager import ConfigManager, warn_activation, yes_no
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.training.network_diagram import visualize_network
from airfoil_pinn.utils import clear_screen

__all__ = [
    "_configure_activation",
    "_configure_coordinate_scaling",
    "_configure_hidden_layers",
    "_configure_input_params",
    "_configure_output_params",
    "architecture_menu",
]


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
            print(f"В папке {ConfigManager.FLUIDS_DIR} нет JSON-файлов. Поместите файлы туда.")
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
