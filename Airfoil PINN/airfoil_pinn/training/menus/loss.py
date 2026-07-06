#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню настройки функции потерь и весов PINN.

Модуль отделяет интерактивную настройку ``loss_weights`` от общего
``learning_cli``. Так главный CLI остается короче, а логику весов проще
тестировать и развивать отдельно от остальных разделов обучения.
"""

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.utils import clear_screen


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
            cfg.config['pde']['system'] = (
                'incompressible_navier_stokes'
                if sys_choice == '1'
                else 'euler'
            )
        elif sub == '2':
            bc_type = input(
                "Тип граничных условий (soft/hard, по умолч. soft): "
            ).strip().lower()
            cfg.config['boundary_conditions']['type'] = (
                bc_type if bc_type in ['hard'] else 'soft'
            )
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
            print(
                f"Интегральная потеря: {lw['integral_condition']}, "
                f"вес: {lw['integral_weight']}"
            )
            fvm_cfg = lw.get('fvm', {})
            fvm_on = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
            if isinstance(fvm_cfg, dict):
                fvm_w = fvm_cfg.get('weight', 1.0)
                fvm_bi = fvm_cfg.get('batch_interval', 1)
            else:
                fvm_w = float(fvm_cfg)
                fvm_bi = 1
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
        fvm_cfg = lw.setdefault(
            'fvm',
            {
                'enabled': False,
                'weight': 1.0,
                'batch_interval': 1,
                'reduction': 'volume_weighted',
            },
        )
        if not isinstance(fvm_cfg, dict):
            fvm_cfg = {
                'enabled': bool(fvm_cfg),
                'weight': float(fvm_cfg) if float(fvm_cfg) > 0 else 1.0,
                'batch_interval': 1,
                'reduction': 'legacy',
            }
            lw['fvm'] = fvm_cfg
        fvm_cfg.setdefault('batch_interval', 1)
        fvm_cfg.setdefault('reduction', 'volume_weighted')
        fvm_enabled = bool(fvm_cfg.get('enabled', False))
        fvm_weight = fvm_cfg.get('weight', 1.0)
        fvm_bi = fvm_cfg.get('batch_interval', 1)
        print(
            "5. Интегральная потеря "
            f"(текущ: {lw['integral_condition']}, вес: {lw['integral_weight']})"
        )
        print(
            "6. FVM Mass flow "
            f"(включено: {fvm_enabled}, вес: {fvm_weight}, "
            f"каждые {fvm_bi} батч.)"
        )
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
                w = input(
                    f"Вес интегральной потери [{lw['integral_weight']}]: "
                ).strip()
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
                bi = input(
                    "FVM раз в K батчей (1=каждый батч) "
                    f"[{fvm_cfg.get('batch_interval', 1)}]: "
                ).strip()
                if bi:
                    try:
                        val = int(bi)
                        fvm_cfg['batch_interval'] = max(1, val)
                    except ValueError:
                        print("Неверное значение, оставлено текущее.")
            fvm_status = 'включено' if fvm_cfg['enabled'] else 'выключено'
            print(
                f"FVM Mass flow: {fvm_status}, "
                f"вес: {fvm_cfg.get('weight', 1.0)}, "
                f"каждые {fvm_cfg.get('batch_interval', 1)} батч."
            )
        elif ch == '7':
            fvm_cfg = lw.get('fvm', {})
            fvm_on = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
            if isinstance(fvm_cfg, dict):
                fvm_w = fvm_cfg.get('weight', 1.0)
                fvm_bi = fvm_cfg.get('batch_interval', 1)
            else:
                fvm_w = float(fvm_cfg)
                fvm_bi = 1
            print("\n--- ТЕКУЩИЕ ВЕСА ---")
            print("PDE weights:", lw['pde_weights'])
            print("BC weights:", lw['bc_weights'])
            print(f"adaptive_bc: {lw['adaptive_bc']}")
            print(f"use_sa_weights: {lw['use_sa_weights']}")
            print(f"sa_apply_to_bc: {lw['sa_apply_to_bc']}")
            print(
                f"integral_condition: {lw['integral_condition']}, "
                f"weight: {lw['integral_weight']}"
            )
            print(f"fvm_mass_flow: {fvm_on}, weight: {fvm_w}, batch_interval: {fvm_bi}")
            input("Нажмите Enter...")
        elif ch == '0':
            print_submenu_footer()
            break
