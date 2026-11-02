#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню процесса обучения PINN.

Этот экран управляет техническими параметрами запуска: загрузкой
модели для дообучения, выбором устройства, числа GPU/CPU, dtype и AMP.
Основной CLI получает только готовую функцию ``process_settings``.
"""

import os

import torch

from airfoil_pinn.app.paths import PATHS, project_relative, resolve_project_path
from airfoil_pinn.config.manager import ConfigManager, yes_no
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.utils import clear_screen

CONFIG_DIR = ConfigManager.CONFIG_DIR

__all__ = ["process_settings"]


def _parse_thread_count(value: str) -> int | str:
    """Parse an interactive ``auto``/positive-integer thread setting."""
    if value.lower() == 'auto':
        return 'auto'
    parsed = int(value)
    if parsed <= 0:
        raise ValueError
    return parsed


def process_settings(cfg: ConfigManager) -> None:
    """Меню настроек процесса обучения."""
    while True:
        clear_screen()
        print_submenu_header("Настройки процесса обучения")
        proc = cfg.config['process']
        print("1. Загрузить PINN и её конфигурацию для дообучения")
        print("2. Выбор GPU/CPU и количества потоков")
        print("3. Выбор типа данных")
        print("4. Показать настройки")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            if yes_no("Загрузить PINN и её конфигурацию для дообучения", proc['load_model_path'] is not None):
                default_path = os.path.join(
                    cfg.config.get('checkpoint_dir', project_relative(PATHS.experiments_dir)),
                    'pinn_interrupted.pt',
                )
                path = input(f"Путь к файлу модели (.pt) [Enter для '{default_path}']: ").strip()
                if not path:
                    path = default_path
                resolved_path = resolve_project_path(path)
                if resolved_path.exists():
                    proc['load_model_path'] = str(resolved_path)
                    base = os.path.splitext(os.path.basename(resolved_path))[0]
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
                current = proc.get('cpu_threads', 'auto')
                value = input(
                    "Количество потоков CPU (auto или целое >= 1) "
                    f"[{current}]: "
                ).strip()
                if value:
                    try:
                        proc['cpu_threads'] = _parse_thread_count(value)
                    except ValueError:
                        print("Введите 'auto' или целое число не меньше 1.")
                        input("Нажмите Enter...")
        elif sub == '3':
            proc['dtype'] = input(f"Тип данных (float32/float64/float16) [{proc['dtype']}]: ").strip() or proc['dtype']
            if proc['dtype'] == 'float32':
                proc['use_amp'] = yes_no("Использовать AMP", proc['use_amp'])
        elif sub == '4':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ПРОЦЕССА ---")
            print(f"Загруженная модель: {proc['load_model_path']}")
            print(f"Устройство: {proc['device']}, GPU: {proc['num_gpus']}")
            print(
                "Потоки CPU: "
                f"intra-op={proc.get('cpu_threads', 'auto')}, "
                f"inter-op={proc.get('cpu_interop_threads', 1)}"
            )
            print(f"Тип данных: {proc['dtype']}, AMP: {proc['use_amp']}")
            print(f"Seed: {proc.get('seed', 42)}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break
