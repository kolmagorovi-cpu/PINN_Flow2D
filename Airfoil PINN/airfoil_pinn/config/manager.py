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

from airfoil_pinn.app.paths import PATHS, project_relative

logger = logging.getLogger(__name__)

__all__ = ["ConfigManager", "yes_no", "warn_activation"]


class ConfigManager:
    """
    Хранит все настройки проекта, загружает/сохраняет в YAML-файлы.
    Предоставляет методы для интерактивного редактирования через меню.
    """
    CONFIG_DIR = str(PATHS.configs_dir)
    FLUIDS_DIR = str(PATHS.fluids_dir)

    def __init__(self) -> None:
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        os.makedirs(self.FLUIDS_DIR, exist_ok=True)
        self.config: Dict[str, Any] = self._default_config()
        self.name: str = "unnamed"

    def _default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию с вложенной структурой, включая диагностику."""
        return {
            'project_name': 'unnamed',
            'input_params': {
                'coordinates': True,
                'sdf': True,
                'boundary_types': True,
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
                # Масштабирование координат для функции активации
                'coordinate_scaling': {
                    'enabled': False,
                    'scale_factor': 1.0,  # Будет вычислено автоматически
                    'input_min': -6.0,
                    'input_max': 6.0,
                },
            },
            'pde': {
                'system': 'incompressible_navier_stokes',
                'flow_type': '2d',
                'pde_on_boundary': False,
            },
            'boundary_conditions': {
                'type': 'soft',
            },
            'loss_weights': {
                'pde_weights': {'cont': 1.0, 'mom_x': 1.0, 'mom_y': 1.0, 'mom_r': 1.0, 'mom_z': 1.0},
                'bc_weights': {
                    'inlet': 1.0,
                    'outlet': 1.0,
                    'outlet_grad': 1.0,
                    'farfield': 1.0,
                    'slip': 1.0,
                    'slip_tangent_grad': 1.0,
                    'slip_velocity': 0.1,
                    'velocity_dirichlet': 1.0,
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
                'fvm': {
                    'enabled': False,
                    'weight': 1.0,
                    'batch_interval': 1,
                    'reduction': 'volume_weighted',
                },
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
                'rar_mode': 'adaptive',
                'rar_score_mode': 'total_loss',
                'rar_interval': 100,
                'rar_num_points': 200,
                'rar_error_quantile': 0.75,
                'rar_residual_power': 1.0,
                'rar_candidate_multiplier': 12,
                'rar_jitter_scale': 0.85,
                'rar_min_distance_factor': 0.35,
                'rar_min_boundary_distance_factor': 0.05,
                'flow_rar': {
                    'enabled': True,
                    'num_points': None,
                    'exclude_sdf_delta': None,
                },
                'boundary_rar': {
                    'enabled': False,
                    'num_points': None,
                    'fraction': 0.4,
                    'delta_mode': 'estimated',
                    'delta_max': 5.0,
                    'eta_levels': [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    'contour_samples': 400,
                    'alpha': 1.5,
                    'uniform_contour_weight': 0.5,
                    'min_distance_factor': 0.02,
                    'score_batch_size': 2048,
                    'error_weights': {
                        'cont': 1.0,
                        'mom_x': 3.0,
                        'mom_y': 3.0,
                        'mom_r': 3.0,
                        'mom_z': 3.0,
                    },
                },
                'rar_rebuild_fvm': True,
                'rar_fvm_rebuild_every': 1,
                'rar_fvm_min_new_points': 1,
                'rar_save_mesh_volume_plot': True,
                'rar_mesh_plot_dpi': 150,
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
                'save_flow_images': True,
                'save_flow_npy': False,
                'flow_plot_physical': False,
                'flow_plot_subdir': 'flow',
            },
            'process': {
                'load_model_path': None,
                'device': 'auto',
                'num_gpus': 1,
                'dtype': 'float32',
                'use_amp': False,
                'torch_compile': False,
                'torch_compile_mode': None,
                'torch_compile_fullgraph': False,
                'torch_compile_dynamic': None,
                'seed': 42,
            },
            'data': {
                'points_file': project_relative(PATHS.initial_fields_dir / 'initial.npy'),
                'mesh_file': project_relative(PATHS.sampling_mesh_dir / 'mesh2d.npz'),
                'num_workers': 0,
                'pin_memory': None,
                'persistent_workers': None,
                'prefetch_factor': 2,
            },
            'reference': {
                'path': None,
            },
            'diagnostics': {
                'interval': 50,
                'plot_residual_map': True,
                'plot_rar_error_map': True,
                'plot_profiles': True,
                'profile_sections': [0.2, 0.5, 0.8],
                'plot_spectrum': True,
                'spectrum_section_x': 0.5,
                'plot_2d_spectrum': False,
                'spectrum_2d_resolution': 256,
                'compute_grad_cosine': False,
                'compute_ntk_condition': False,
                'strict_autograd': True,
                'save_rar_history': True,
                'mass_metrics_interval': 1,
                'boundary_layer': {
                    'enabled': False,
                    'interval': 50,
                    'sections': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
                    'sides': ['upper', 'lower'],
                    'profile_step_fraction': 0.01,
                    'first_layer_height_fraction': 1.0e-5,
                    'layer_count': 80,
                    'growth_ratio': 1.12,
                    'max_normal_distance_fraction': 0.08,
                    'edge_threshold': 0.99,
                    'fit_points': 8,
                    'x_min_c': 0.02,
                    'x_max_c': 0.98,
                    'max_surface_points_per_side': 120,
                    'unresolved_threshold': 0.8,
                    'eval_batch_size': 8192,
                    'save_profiles_npz': True,
                    'save_metrics_csv': True,
                    'plot_profiles': True,
                    'plot_cp': True,
                    'plot_cf': True,
                    'plot_delta99': True,
                    'plot_deficit': True,
                },
            },
            'enable_key_listener': True,
            'checkpoint_dir': project_relative(PATHS.experiments_dir),
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
        """
        Рекурсивно преобразует кортежи в списки для корректной сериализации YAML.

        Args:
            d: Данные для преобразования

        Returns:
            Данные с кортежами, преобразованными в списки
        """
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
        print(f"FVM-сетка : {data.get('mesh_file', 'не указана')}")
        print(
            f"DataLoader: workers={data.get('num_workers', 0)}, "
            f"pin_memory={data.get('pin_memory', None)}, "
            f"prefetch={data.get('prefetch_factor', 2)}"
        )
        print()

        print("ВХОДНЫЕ ПАРАМЕТРЫ:")
        inp = self.config['input_params']
        print(f"  coordinates     : {inp.get('coordinates', True)}")
        print(f"  sdf             : {inp.get('sdf', True)}")
        print(f"  boundary_types  : {inp.get('boundary_types', True)}")
        print(f"  inlet_velocity  : {inp.get('inlet_velocity', False)}")
        print(f"  viscosity       : {inp.get('viscosity', False)}")
        print(f"  density         : {inp.get('density', False)}")
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
        fvm_cfg = lw.get('fvm', {})
        fvm_enabled = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
        fvm_weight = fvm_cfg.get('weight', 1.0) if isinstance(fvm_cfg, dict) else float(fvm_cfg)
        fvm_bi = fvm_cfg.get('batch_interval', 1) if isinstance(fvm_cfg, dict) else 1
        fvm_red = fvm_cfg.get('reduction', 'legacy') if isinstance(fvm_cfg, dict) else 'legacy'
        print(
            f"  fvm_mass_flow   : {fvm_enabled} "
            f"(weight={fvm_weight}, batch_interval={fvm_bi}, reduction={fvm_red})"
        )
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
            print(f"    rar_mode              : {tr.get('rar_mode', 'adaptive')}")
            print(f"    rar_score_mode        : {tr.get('rar_score_mode', 'total_loss')}")
            if tr.get('rar_mode', 'adaptive') == 'adaptive':
                print(f"    rar_error_quantile    : {tr.get('rar_error_quantile', 0.75)}")
                print(f"    rar_rebuild_fvm       : {tr.get('rar_rebuild_fvm', True)}")
                print(f"    rar_fvm_rebuild_every : {tr.get('rar_fvm_rebuild_every', 1)}")
                print(f"    rar_save_mesh_volume_plot: {tr.get('rar_save_mesh_volume_plot', True)}")
                flow_rar = tr.get('flow_rar', {}) or {}
                boundary_rar = tr.get('boundary_rar', {}) or {}
                print(f"    flow_rar.enabled      : {flow_rar.get('enabled', True)}")
                print(f"    flow_rar.exclude_sdf_delta: {flow_rar.get('exclude_sdf_delta', None)}")
                print(f"    boundary_rar.enabled  : {boundary_rar.get('enabled', False)}")
                if boundary_rar.get('enabled', False):
                    print(f"    boundary_rar.num_points: {boundary_rar.get('num_points', None)}")
                    print(f"    boundary_rar.fraction : {boundary_rar.get('fraction', 0.4)}")
                    print(f"    boundary_rar.delta_max: {boundary_rar.get('delta_max', 5.0)}")
                    print(f"    boundary_rar.contour_samples: {boundary_rar.get('contour_samples', 400)}")
                    print(f"    boundary_rar.min_distance_factor: {boundary_rar.get('min_distance_factor', 0.02)}")
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
        print(f"  save_flow_images: {ck.get('save_flow_images', True)}")
        print(f"  save_flow_npy  : {ck.get('save_flow_npy', False)}")
        print(f"  flow_plot_physical: {ck.get('flow_plot_physical', False)}")
        print(f"  flow_plot_subdir: {ck.get('flow_plot_subdir', 'flow')}")
        print()

        proc = self.config['process']
        print("ПРОЦЕСС:")
        print(f"  device         : {proc['device']}")
        print(f"  num_gpus       : {proc['num_gpus']}")
        print(f"  dtype          : {proc['dtype']}")
        print(f"  use_amp        : {proc['use_amp']}")
        print(f"  torch_compile  : {proc.get('torch_compile', False)}")
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
        print(f"  plot_rar_error_map       : {diag.get('plot_rar_error_map', True)}")
        print(f"  plot_profiles            : {diag.get('plot_profiles', True)}")
        print(f"  profile_sections         : {diag.get('profile_sections', [0.2, 0.5, 0.8])}")
        print(f"  plot_spectrum (1D)       : {diag.get('plot_spectrum', True)}")
        print(f"  spectrum_section_x       : {diag.get('spectrum_section_x', 0.5)}")
        print(f"  plot_2d_spectrum         : {diag.get('plot_2d_spectrum', False)}")
        print(f"  spectrum_2d_resolution   : {diag.get('spectrum_2d_resolution', 256)}")
        print(f"  compute_grad_cosine      : {diag.get('compute_grad_cosine', False)}")
        print(f"  compute_ntk_condition    : {diag.get('compute_ntk_condition', False)}")
        print(f"  strict_autograd          : {diag.get('strict_autograd', True)}")
        print(f"  save_rar_history         : {diag.get('save_rar_history', True)}")
        print(f"  mass_metrics_interval    : {diag.get('mass_metrics_interval', 1)}")
        print("=" * 70)

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Рекурсивно обновляет словарь target значениями из source.
        Сохраняет ключи, отсутствующие в source.

        Args:
            target: Целевой словарь для обновления
            source: Источник значений для обновления
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
    suffix: str = " (Y/n): " if default else " (y/N): "
    ans: str = input(prompt + suffix).strip().lower()
    if default:
        return ans not in ['n', 'no']
    else:
        return ans in ['y', 'yes']


# ==============================================================================
# ПРЕДУПРЕЖДЕНИЯ О ФУНКЦИЯХ АКТИВАЦИИ
# ==============================================================================

ACTIVATION_WARNINGS = {
    'tanh': {
        'level': 'critical',
        'message': 'Требуется масштабирование координат к [-2, 2]. Без масштабирования градиенты затухнут на 95%+.',
        'recommendation': 'Автоматическое масштабирование: x_scaled = x * scale_factor, где scale_factor = 2.0 / max(|x|)'
    },
    'sigmoid': {
        'level': 'critical',
        'message': 'Требуется масштабирование координат к [-2, 2]. Максимальный градиент всего 0.25, на краях затухает на 99%+.',
        'recommendation': 'Автоматическое масштабирование: x_scaled = x * scale_factor, где scale_factor = 2.0 / max(|x|)'
    },
    'relu': {
        'level': 'critical',
        'message': 'Требуется масштабирование к [0, ∞). Отрицательные координаты убьют 50% нейронов (градиент = 0).',
        'recommendation': 'Автоматическое масштабирование: x_scaled = (x - min(x)) / (max(x) - min(x)) * 6.0'
    },
    'leaky_relu': {
        'level': 'warning',
        'message': 'Отрицательные значения имеют градиент 0.01 (в 100 раз меньше положительного).',
        'recommendation': 'Рекомендуется масштабирование к [-1, 5] для баланса.'
    },
    'elu': {
        'level': 'warning',
        'message': 'Отрицательные значения имеют затухающие градиенты (e^x). При x=-6 градиент ≈ 0.0025.',
        'recommendation': 'Рекомендуется масштабирование к [-2, 5].'
    },
    'swish': {
        'level': 'warning',
        'message': 'Отрицательные значения имеют затухающие градиенты. При x=-6 градиент ≈ 0.0025.',
        'recommendation': 'Рекомендуется масштабирование к [-2, 5].'
    },
    'silu': {
        'level': 'warning',
        'message': 'Отрицательные значения имеют затухающие градиенты. При x=-6 градиент ≈ 0.0025.',
        'recommendation': 'Рекомендуется масштабирование к [-2, 5].'
    },
    'siren': {
        'level': 'info',
        'message': 'Масштабирование не требуется. SIREN (sin) работает с любым диапазоном — градиенты периодические.',
        'recommendation': 'Специальная инициализация уже настроена в models.py'
    },
}


def warn_activation(activation_name: str, x_min: float = -6.0, x_max: float = 6.0) -> tuple[bool, float]:
    """
    Выводит предупреждение о выбранной функции активации.

    Args:
        activation_name: название функции активации
        x_min: минимальное значение координат
        x_max: максимальное значение координат

    Returns:
        tuple[bool, float]: (пользователь согласен, scale_factor)
    """
    activation_lower: str = activation_name.lower()
    warning: Optional[Dict[str, str]] = ACTIVATION_WARNINGS.get(activation_lower)

    if not warning:
        print(f"\n⚪ Неизвестная функция активации '{activation_name}'. Масштабирование не настроено.")
        return True, 1.0

    level_icons: Dict[str, str] = {'critical': '🔴', 'warning': '🟡', 'info': '✅'}
    icon: str = level_icons.get(warning['level'], '⚪')

    print(f"\n{'='*70}")
    print(f"{icon} ПРЕДУПРЕЖДЕНИЕ: Функция активации '{activation_name}'")
    print(f"{'='*70}")
    print(f"   Диапазон координат: [{x_min:.2f}, {x_max:.2f}]")
    print(f"   {warning['message']}")
    print(f"   💡 Рекомендация: {warning['recommendation']}")

    # Вычисляем рекомендуемый scale_factor
    scale_factor: float
    if activation_lower in ['tanh', 'sigmoid']:
        # Масштабирование к [-2, 2]
        max_abs: float = max(abs(x_min), abs(x_max))
        scale_factor = 2.0 / max_abs if max_abs > 0 else 1.0
    elif activation_lower == 'relu':
        # Масштабирование к [0, 6]
        range_x: float = x_max - x_min
        scale_factor = 6.0 / range_x if range_x > 0 else 1.0
    elif activation_lower in ['leaky_relu', 'elu', 'swish', 'silu']:
        # Масштабирование к [-2, 5]
        max_abs = max(abs(x_min), abs(x_max))
        scale_factor = 5.0 / max_abs if max_abs > 0 else 1.0
    else:  # siren
        scale_factor = 1.0

    if warning['level'] == 'info':
        print(f"\n   ✅ Масштабирование не требуется (scale_factor = {scale_factor:.3f})")
        input("   Нажмите Enter для продолжения...")
        return True, scale_factor

    print(f"\n   Рекомендуемый scale_factor: {scale_factor:.4f}")
    print(f"   {'='*70}")
    print(f"\n   Выберите режим масштабирования:")
    print(f"   1. Применить рекомендуемый scale_factor = {scale_factor:.4f}")
    print(f"   2. Ввести свой scale_factor")
    print(f"   3. Отключить масштабирование (не рекомендуется!)")

    mode_choice: str = input("\n   Ваш выбор (1/2/3): ").strip()

    if mode_choice == '1':
        print(f"   ✅ Масштабирование включено (scale_factor = {scale_factor:.4f})")
        return True, scale_factor
    elif mode_choice == '2':
        while True:
            try:
                custom_scale: str = input(f"   Введите scale_factor (рекомендуется {scale_factor:.4f}): ").strip()
                custom_scale_float: float = float(custom_scale)
                if custom_scale_float <= 0:
                    print("   ⚠️ scale_factor должен быть положительным числом!")
                    continue
                print(f"   ✅ Масштабирование включено (scale_factor = {custom_scale_float:.4f})")
                return True, custom_scale_float
            except ValueError:
                print("   ⚠️ Введите корректное число!")
    else:  # mode_choice == '3'
        print(f"   ⚠️ Масштабирование отключено. Возможны проблемы с обучением!")
        return False, 1.0

    return True, scale_factor
