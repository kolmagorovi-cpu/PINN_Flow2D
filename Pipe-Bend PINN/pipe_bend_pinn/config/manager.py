#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Управление конфигурацией для Pipe-Bend PINN.

Поддерживает:
- Главный конфиг config.yaml в workspace/configs/ или в корне проекта
- Вложенные ссылки (geometry_config, fluids_config, bc_config)
- Авто-загрузку при инициализации
- Обратную совместимость со старыми JSON/YAML файлами

Автор: Адаптировано из Airfoil PINN
Версия: 2.1
"""

import os
import logging
from typing import Dict, Any, Optional

import yaml

from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT,
    PATHS,
    CONFIGS_DIR,
    LEGACY_CONFIGS_DIR,
)

logger = logging.getLogger(__name__)


def _resolve_path(base_dir: str, path: str) -> str:
    """Разрешает относительный путь относительно base_dir."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _load_yaml(filepath: str) -> Dict[str, Any]:
    """Загружает YAML-файл, возвращает словарь."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class ConfigManager:
    """
    Менеджер конфигурации.

    - По умолчанию ищет config.yaml в workspace/configs/ или в корне проекта
    - Загружает вложенные конфиги из geometry_config, fluids_config, boundary_conditions_config
    - Может работать в legacy-режиме (Configs/*.yaml)
    """

    CONFIG_DIR = CONFIGS_DIR
    MAIN_CONFIG = "config.yaml"

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Args:
            config_path: Путь к config.yaml. Если None — ищется автоматически.
        """
        # Определяем корень проекта
        self.BASE_DIR = str(PROJECT_ROOT)
        self.name: str = "unnamed"
        self.main_config_path: Optional[str] = None
        self.loaded_sub_configs: Dict[str, str] = {}  # какие подконфиги загружены

        # Загружаем конфиг
        if config_path:
            self.main_config_path = config_path
            self.config = self._load_main_config(config_path)
        else:
            # Ищем config.yaml по умолчанию: workspace/configs, затем корень проекта
            default_paths = [
                os.path.join(CONFIGS_DIR, self.MAIN_CONFIG),
                os.path.join(self.BASE_DIR, self.MAIN_CONFIG),
            ]
            found_path = None
            for p in default_paths:
                if os.path.exists(p):
                    found_path = p
                    break

            if found_path is not None:
                self.main_config_path = found_path
                self.config = self._load_main_config(found_path)
            else:
                self.config = self._default_config()

    @staticmethod
    def _find_base_dir() -> str:
        """Определяет корень проекта по расположению этого файла."""
        return str(PROJECT_ROOT)

    def _load_main_config(self, filepath: str) -> Dict[str, Any]:
        """Загружает главный config.yaml и мержит вложенные."""
        base_dir = os.path.dirname(os.path.abspath(filepath))
        config = self._default_config()

        # Загружаем главный YAML
        main = _load_yaml(filepath)
        self.name = main.get('project_name', 'unnamed')

        # Мержим все секции из главного конфига
        self._deep_merge(config, main)

        # Загружаем вложенные подконфиги
        for key, sub_path_key in [
            ('geometry_config', 'geometry_config'),
            ('fluids_config', 'fluids_config'),
            ('boundary_conditions_config', 'boundary_conditions_config'),
        ]:
            sub_path = main.get(sub_path_key)
            if sub_path and isinstance(sub_path, str):
                resolved = _resolve_path(base_dir, sub_path)
                if os.path.exists(resolved):
                    try:
                        sub_data = _load_yaml(resolved)
                        self._deep_merge(config, {
                            '_' + key: sub_data,  # сохраняем в служебном поле
                            '_loaded_sub': sub_path,
                        })
                        self.loaded_sub_configs[key] = resolved
                        logger.info(f"Загружен подконфиг {key}: {resolved}")
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить {key} ({resolved}): {e}")

        # Поле с путём к файлу точек: относительный путь резолвим от корня проекта
        raw_points = config.get('data', {}).get('points_file')
        if raw_points is not None and isinstance(raw_points, str):
            if not os.path.isabs(raw_points):
                config['data']['points_file'] = os.path.normpath(
                    os.path.join(str(PROJECT_ROOT), raw_points)
                )
        else:
            initial = os.path.join(PATHS.initial_fields_dir, "initial.npy")
            if os.path.exists(initial):
                config['data']['points_file'] = initial

        return config

    def _default_config(self) -> Dict[str, Any]:
        return {
            'project_name': 'unnamed',
            'flow_type': 'planar',
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
                'coordinate_scaling': {
                    'enabled': False,
                    'scale_factor': 1.0,
                    'input_min': -6.0,
                    'input_max': 6.0,
                },
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
                    'inlet': 5000.0, 'outlet': 5000.0, 'wall': 5000.0,
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
                    'loss': True, 'pde': True, 'bc': False,
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
            'checkpoint_dir': 'workspace/experiments',
        }

    def load(self, name: str) -> None:
        """Загружает конфигурацию из workspace/configs/ (или legacy Configs/)."""
        # Сначала пробуем workspace/configs, затем старый Configs/
        candidates = [
            os.path.join(CONFIGS_DIR, name + '.yaml'),
            os.path.join(str(LEGACY_CONFIGS_DIR), name + '.yaml'),
        ]
        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break

        if path is None:
            raise FileNotFoundError(f"Конфигурация {name} не найдена в {CONFIGS_DIR} или Configs/")
        self.load_from_file(path)
        self.name = name
        logger.info(f"Конфигурация '{name}' загружена из {path}")

    def load_from_file(self, filepath: str) -> None:
        """Загружает из произвольного YAML-файла."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл конфигурации не найден: {filepath}")
        loaded = _load_yaml(filepath)
        self._deep_merge(self.config, loaded)

    def save(self, name: Optional[str] = None) -> None:
        """Сохраняет конфигурацию в workspace/configs/."""
        if name:
            self.name = name
        config_copy = self._prepare_for_save(self.config)
        path = os.path.join(CONFIGS_DIR, self.name + '.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Конфигурация сохранена: {path}")

    def _prepare_for_save(self, d: Any) -> Any:
        if isinstance(d, dict):
            return {k: self._prepare_for_save(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._prepare_for_save(v) for v in d]
        elif isinstance(d, tuple):
            return list(d)
        return d

    def show(self) -> None:
        """Вывод текущей конфигурации."""
        print("\n" + "=" * 70)
        print(f"КОНФИГУРАЦИЯ: {self.config.get('project_name', 'unnamed')}".center(70))
        print("=" * 70)
        print(f"Главный конфиг: {self.main_config_path or 'не указан'}")
        if self.loaded_sub_configs:
            print("Подключённые подконфиги:")
            for key, path in self.loaded_sub_configs.items():
                print(f"  {key}: {path}")
        print(f"Тип течения: {self.config.get('flow_type', 'planar')}")
        print(f"Файл точек: {self.config['data'].get('points_file', 'не указан')}")
        print()
        print("АРХИТЕКТУРА:")
        arch = self.config['architecture']
        print(f"  Слои: {arch['hidden_layers']}")
        print(f"  Активация: {arch.get('activation', 'tanh')}")
        print()
        print("ВЕСА:")
        lw = self.config['loss_weights']
        for k, v in lw['pde_weights'].items():
            print(f"  PDE {k}: {v}")
        for k, v in lw['bc_weights'].items():
            print(f"  BC {k}: {v}")
        print()
        tr = self.config['training']
        print(f"Эпох: {tr['epochs']}, Батч: {tr['batch_size']}, LR: {tr.get('lr', 1e-3)}")
        print(f"Two-stage: {tr['two_stage']}, RAR: {tr['rar_enable']}")
        print(f"Device: {self.config['process']['device']}")
        print("=" * 70)

    @staticmethod
    def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Рекурсивно обновляет target значениями из source."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigManager._deep_merge(target[key], value)
            else:
                target[key] = value

    def reset_to_defaults(self) -> None:
        self.config = self._default_config()
        self.name = "unnamed"
        logger.info("Конфигурация сброшена к значениям по умолчанию.")