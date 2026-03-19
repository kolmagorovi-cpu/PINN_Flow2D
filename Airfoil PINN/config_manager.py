# Исправленный файл: config_manager.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Управление конфигурацией для проекта PINN Learning.
Содержит класс ConfigManager для хранения, загрузки и сохранения настроек проекта.
Добавлен раздел diagnostics с параметрами диагностики.

Автор: Колмагоров И.Д.
Версия: 3.260226 (с поддержкой диагностики и 2D спектра)
"""

import os
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Хранит все настройки проекта, загружает/сохраняет в YAML-файлы.
    Предоставляет методы для интерактивного редактирования через меню.
    """
    CONFIG_DIR = "Configs"
    FLUIDS_DIR = "Fluids"

    def __init__(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        os.makedirs(self.FLUIDS_DIR, exist_ok=True)
        self.config = self._default_config()
        self.name = "unnamed"

    def _default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию с вложенной структурой, включая диагностику."""
        return {
            'project_name': 'unnamed',
            'input_params': {
                'coordinates': True,
                'sdf': True,
                'inlet_velocity': False,
                'viscosity': False,
                'density': False,
                'fluid_file': None,
            },
            'architecture': {
                'hidden_layers': [50, 50, 50, 50],
                'activation': 'tanh',
                'output_dim': 3,
                'output_bias': [0.1, 0.0, 0.0],
            },
            'pde': {
                'system': 'incompressible_navier_stokes',
                'flow_type': '2d',
            },
            'boundary_conditions': {
                'type': 'soft',
            },
            'loss_weights': {
                'pde_weights': {'cont': 1.0, 'mom_x': 1.0, 'mom_y': 1.0},
                'bc_weights': {
                    'inlet': 1.0,
                    'outlet': 1.0,
                    'wall': 1.0,
                    'symmetry': 1.0,
                    'axis_symmetry': 1.0,
                    'symmetry_grad': 1.0,
                    'axis_symmetry_grad': 1.0,
                },
                'adaptive_bc': False,
                'use_sa_weights': False,
                'sa_apply_to_bc': False,
                'integral_condition': False,
                'integral_weight': 0.0,
            },
            'training': {
                'two_stage': False,
                'stage1_epochs': 100,
                'stage2_pde_weight_start': 1.0,
                'stage2_pde_weight_end': 100000.0,
                'stage2_ramp_epochs': 500,
                'lr_schedule': 'constant',
                'lr_cyclic_period': 100,
                'lr_cyclic_amplitude': 0.5,
                'lr_detector_factor': 2.0,
                'lr_detector_patience': 10,
                'lr_detector_recovery': 50,
                'reset_interval': 0,
                'reset_noise': 0.01,
                'grad_clip': None,
                'rar_enable': False,
                'rar_interval': 100,
                'rar_num_points': 200,
                'switch_to_lbfgs_epoch': 0,
                'lbfgs_max_iter': 20,
                'lbfgs_history_size': 100,
                'batch_size': 10,
                'gradient_accumulation_steps': 1,
                'epochs': 1000,
                # Early stopping
                'early_stopping': {
                    'enabled': False,
                    'patience': 100,
                    'min_delta': 1e-6,
                    'restore_best': True,
                },
            },
            'output': {
                'console_only': True,
                'graphical_window': {
                    'loss': True,
                    'pde': True,
                    'bc': False,
                },
                'plot_frequency': 10,
            },
            'checkpoint': {
                'interval': 500,
                'name_pattern': '{project_name}_{timestamp}',
                'model_name_pattern': '{project_name}_{timestamp}',
            },
            'process': {
                'load_model_path': None,
                'device': 'auto',
                'num_gpus': 1,
                'dtype': 'float32',
                'use_amp': False,
                'seed': 42,
            },
            'data': {
                'points_file': None,
            },
            'reference': {
                'path': None,
            },
            'diagnostics': {
                'interval': 50,
                'plot_residual_map': True,
                'plot_profiles': True,
                'profile_sections': [0.2, 0.5, 0.8],
                'plot_spectrum': True,
                'spectrum_section_x': 0.5,
                'plot_2d_spectrum': False,
                'spectrum_2d_resolution': 256,
                'compute_grad_cosine': False,
                'compute_ntk_condition': False,
                'save_rar_history': True,
            },
            'enable_key_listener': True,
            'checkpoint_dir': 'Checkpoints',
        }

    def load(self, name: str) -> None:
        """
        Загружает конфигурацию из YAML-файла в папке CONFIG_DIR.

        Args:
            name: имя конфигурации (без расширения .yaml)

        Raises:
            FileNotFoundError: если файл не найден.
            yaml.YAMLError: при ошибке парсинга YAML.
        """
        path = os.path.join(self.CONFIG_DIR, name + '.yaml')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Конфигурация {name} не найдена")

        self.load_from_file(path)
        self.name = name
        logger.info(f"Конфигурация '{name}' загружена из {path}")

    def load_from_file(self, filepath: str) -> None:
        """
        Загружает конфигурацию из произвольного YAML-файла и обновляет текущую.

        Args:
            filepath: полный путь к файлу

        Raises:
            FileNotFoundError: если файл не найден.
            yaml.YAMLError: при ошибке парсинга YAML.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл конфигурации не найден: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
        except yaml.constructor.ConstructorError as e:
            logger.warning(f"Ошибка safe_load для {filepath}: {e}. Пробуем FullLoader (может быть небезопасно)")
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            logger.error(f"Не удалось загрузить YAML из {filepath}: {e}")
            raise

        if loaded is None:
            loaded = {}

        # Рекурсивно обновляем словарь
        self._deep_update(self.config, loaded)

        # Преобразуем списки в кортежи там, где это необходимо (например, output_bias)
        if 'architecture' in self.config and 'output_bias' in self.config['architecture']:
            if isinstance(self.config['architecture']['output_bias'], list):
                self.config['architecture']['output_bias'] = tuple(self.config['architecture']['output_bias'])

    def save(self, name: Optional[str] = None) -> None:
        """
        Сохраняет текущую конфигурацию в YAML-файл.

        Args:
            name: имя конфигурации (если не указано, используется текущее self.name)
        """
        if name:
            self.name = name
        # Перед сохранением преобразуем кортежи в списки
        config_copy = self._prepare_for_save(self.config)
        path = os.path.join(self.CONFIG_DIR, self.name + '.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Конфигурация сохранена: {path}")

    def _prepare_for_save(self, d: Any) -> Any:
        """Рекурсивно преобразует кортежи в списки для корректной сериализации YAML."""
        if isinstance(d, dict):
            return {k: self._prepare_for_save(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._prepare_for_save(v) for v in d]
        elif isinstance(d, tuple):
            return list(d)
        else:
            return d

    def show(self) -> None:
        """Подробный вывод текущей конфигурации в столбик (читаемый формат)."""
        print("\n" + "=" * 70)
        print(f"КОНФИГУРАЦИЯ: {self.config.get('project_name', 'unnamed')}".center(70))
        print("=" * 70)

        data = self.config['data']
        print(f"Файл точек: {data.get('points_file', 'не указан')}")
        print()

        print("ВХОДНЫЕ ПАРАМЕТРЫ:")
        inp = self.config['input_params']
        print(f"  inlet_velocity  : {inp['inlet_velocity']}")
        print(f"  viscosity       : {inp['viscosity']}")
        print(f"  density         : {inp['density']}")
        print(f"  fluid_file      : {inp.get('fluid_file', '-')}")
        print()

        print("АРХИТЕКТУРА:")
        arch = self.config['architecture']
        print(f"  hidden_layers   : {arch['hidden_layers']}")
        print(f"  activation      : {arch.get('activation', 'tanh')}")
        print(f"  output_dim      : {arch['output_dim']}")
        print(f"  output_bias     : {arch['output_bias']}")
        print()

        print("ВЕСА PDE:")
        lw = self.config['loss_weights']
        for k, v in lw['pde_weights'].items():
            print(f"  {k:<16}: {v}")
        print()

        print("ВЕСА BC:")
        for k, v in lw['bc_weights'].items():
            print(f"  {k:<20}: {v}")
        print()

        print("АДАПТИВНЫЕ НАСТРОЙКИ:")
        print(f"  adaptive_bc     : {lw['adaptive_bc']}")
        print(f"  use_sa_weights  : {lw['use_sa_weights']}")
        print(f"  sa_apply_to_bc  : {lw['sa_apply_to_bc']}")
        print(f"  integral        : {lw['integral_condition']} (weight={lw['integral_weight']})")
        print()

        tr = self.config['training']
        print("ОБУЧЕНИЕ:")
        print(f"  two_stage               : {tr['two_stage']}")
        if tr['two_stage']:
            print(f"    stage1_epochs         : {tr['stage1_epochs']}")
            print(f"    stage2_pde_weight_start: {tr['stage2_pde_weight_start']}")
            print(f"    stage2_pde_weight_end  : {tr['stage2_pde_weight_end']}")
            print(f"    stage2_ramp_epochs     : {tr['stage2_ramp_epochs']}")
        print(f"  lr_schedule             : {tr['lr_schedule']}")
        print(f"  reset_interval          : {tr['reset_interval']} (noise={tr['reset_noise']})")
        print(f"  grad_clip               : {tr['grad_clip']}")
        print(f"  rar_enable              : {tr['rar_enable']}")
        if tr['rar_enable']:
            print(f"    rar_interval          : {tr['rar_interval']}")
            print(f"    rar_num_points        : {tr['rar_num_points']}")
        print(f"  switch_to_lbfgs_epoch   : {tr['switch_to_lbfgs_epoch']}")
        if tr['switch_to_lbfgs_epoch'] > 0:
            print(f"    lbfgs_max_iter        : {tr['lbfgs_max_iter']}")
            print(f"    lbfgs_history_size    : {tr['lbfgs_history_size']}")
        print(f"  batch_size              : {tr['batch_size']}")
        print(f"  gradient_accumulation_steps: {tr['gradient_accumulation_steps']}")
        print(f"  epochs                  : {tr['epochs']}")
        print()

        out = self.config['output']
        print("ВЫВОД:")
        print(f"  console_only   : {out['console_only']}")
        if not out['console_only']:
            print(f"    loss         : {out['graphical_window']['loss']}")
            print(f"    pde          : {out['graphical_window']['pde']}")
            print(f"    bc           : {out['graphical_window']['bc']}")
        print(f"  plot_frequency : {out['plot_frequency']}")
        print()

        ck = self.config['checkpoint']
        print("СОХРАНЕНИЕ:")
        print(f"  interval       : {ck['interval']}")
        print(f"  name_pattern   : {ck['name_pattern']}")
        print(f"  model_name_pattern: {ck['model_name_pattern']}")
        print()

        proc = self.config['process']
        print("ПРОЦЕСС:")
        print(f"  device         : {proc['device']}")
        print(f"  num_gpus       : {proc['num_gpus']}")
        print(f"  dtype          : {proc['dtype']}")
        print(f"  use_amp        : {proc['use_amp']}")
        print(f"  seed           : {proc.get('seed', 42)}")
        print()

        ref = self.config.get('reference', {})
        print("ЭТАЛОННОЕ РЕШЕНИЕ:")
        print(f"  path           : {ref.get('path', 'не указан')}")
        print()

        diag = self.config.get('diagnostics', {})
        print("ДИАГНОСТИКА:")
        print(f"  interval                 : {diag.get('interval', 50)}")
        print(f"  plot_residual_map        : {diag.get('plot_residual_map', True)}")
        print(f"  plot_profiles            : {diag.get('plot_profiles', True)}")
        print(f"  profile_sections         : {diag.get('profile_sections', [0.2, 0.5, 0.8])}")
        print(f"  plot_spectrum (1D)       : {diag.get('plot_spectrum', True)}")
        print(f"  spectrum_section_x       : {diag.get('spectrum_section_x', 0.5)}")
        print(f"  plot_2d_spectrum         : {diag.get('plot_2d_spectrum', False)}")
        print(f"  spectrum_2d_resolution   : {diag.get('spectrum_2d_resolution', 256)}")
        print(f"  compute_grad_cosine      : {diag.get('compute_grad_cosine', False)}")
        print(f"  compute_ntk_condition    : {diag.get('compute_ntk_condition', False)}")
        print(f"  save_rar_history         : {diag.get('save_rar_history', True)}")
        print("=" * 70)

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Рекурсивно обновляет словарь target значениями из source.
        Сохраняет ключи, отсутствующие в source.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def reset_to_defaults(self) -> None:
        """Сбрасывает конфигурацию к значениям по умолчанию."""
        self.config = self._default_config()
        self.name = "unnamed"
        logger.info("Конфигурация сброшена к значениям по умолчанию.")


# Вспомогательная функция для интерактивных меню
def yes_no(prompt: str, default: bool = False) -> bool:
    """
    Запрашивает у пользователя ответ yes/no.

    Args:
        prompt: текст вопроса
        default: значение по умолчанию (True для Y, False для N)

    Returns:
        bool: True если пользователь ввёл y/yes, иначе False
    """
    suffix = " (Y/n): " if default else " (y/N): "
    ans = input(prompt + suffix).strip().lower()
    if default:
        return ans not in ['n', 'no']
    else:
        return ans in ['y', 'yes']