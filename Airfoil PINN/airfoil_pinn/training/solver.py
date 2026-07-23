#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной решатель PINN (дирижёр) для проекта PINN Learning.
Координирует работу компонентов: LossComputer, Trainer, MetricsCollector, InteractiveHandler.
Добавлена диагностическая логика: сбор карт невязки, профилей, 2D спектра, градиентов и т.д.

Исправленная версия:
- Диагностика использует предсказания модели вместо начальных полей
- Добавлен 2D спектр и его настройка в конфигурации
- Исправлен вызов scheduler.step()
- Уменьшена частота torch.cuda.empty_cache()
- Для карты невязки используется отдельное поле PDE-невязки; RAR использует
  собственный score (обычно local total loss)
- Оптимизирована работа RAR с SA-весами
- В 2D спектре исключены граничные точки (отступ 5% от краёв) для устранения артефактов
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np

from airfoil_pinn.app.paths import PATHS, project_relative, resolve_project_path
from airfoil_pinn.config.validation import (
    validate_runtime_contract,
    validate_static_config,
)
from airfoil_pinn.utils import check_memory_usage, set_seed
from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.training.models import PINNIncompressible, SelfAdaptiveWeights
from airfoil_pinn.training.checkpoint import CheckpointManager
from airfoil_pinn.training.contract import (
    build_training_contract,
    validate_checkpoint_contract,
)
from airfoil_pinn.training.key_listener import KeyListener
from airfoil_pinn.training.visualizer import Visualizer
from airfoil_pinn.training.boundary_layer_diagnostics import run_boundary_layer_diagnostics
from airfoil_pinn.training.reference import (
    compute_l2_errors,
    interpolate_reference,
    load_reference,
)

# Компоненты тренировочного процесса.
from airfoil_pinn.training.components.optimizers import LocalMinimaDetector, CyclicLR
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.components.metrics import MetricsCollector
from airfoil_pinn.training.components.interactive import InteractiveHandler

logger = logging.getLogger(__name__)

__all__ = ["PINNSolver"]


class PINNSolver:
    """
    Основной класс-дирижёр, управляющий процессом обучения PINN.
    Инициализирует все компоненты и запускает цикл обучения.
    """

    def __init__(self, config: Dict, experiment_dir: Optional[str] = None):
        """
        Args:
            config: словарь конфигурации
            experiment_dir: директория для сохранения результатов эксперимента
                           (если None, создаётся уникальная папка в workspace/experiments)
        """
        validate_static_config(config)
        self.config = config
        self.config_name = config.get('project_name', 'unnamed')
        self.experiment_dir = str(resolve_project_path(experiment_dir)) if experiment_dir else None

        # Timestamp фиксируется один раз — используется во всех _get_*_dir()
        self._timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Установка seed
        seed = config.get('process', {}).get('seed', 42)
        set_seed(seed)

        # Устройство и тип данных
        self.device = self._init_device(config['process'])
        self.dtype = self._init_dtype(config['process']['dtype'])
        requested_amp = bool(config['process'].get('use_amp', False))
        self.use_amp = requested_amp and self.device.type == 'cuda' and self.dtype == torch.float32
        if requested_amp and not self.use_amp:
            logger.warning("AMP отключён: требуется CUDA и dtype=float32.")

        # Параметры мониторинга памяти
        self.memory_threshold_gb = config.get('process', {}).get('memory_threshold_gb', 0.5)

        # Загрузка данных
        data_cfg = config.get('data', {})
        pin_memory = data_cfg.get('pin_memory')
        if pin_memory is None:
            pin_memory = self.device.type == 'cuda'
        self.data_handler = DataHandler(
            points_file=data_cfg['points_file'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            mesh_file=data_cfg.get('mesh_file'),
            num_workers=data_cfg.get('num_workers', 0),
            pin_memory=pin_memory,
            persistent_workers=data_cfg.get('persistent_workers'),
            prefetch_factor=data_cfg.get('prefetch_factor'),
        )

        # Константы PDE (нормированные)
        self.rho_norm = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.mu_norm = torch.tensor(1.0 / self.data_handler.Re, device=self.device, dtype=self.dtype)
        self._validate_physics_metadata()

        # Загрузка эталонного решения (если указано)
        self.reference_data = None
        ref_path = config.get('reference', {}).get('path')
        if ref_path and os.path.exists(ref_path):
            try:
                self.reference_data = load_reference(ref_path)
                logger.info(f"Эталонное решение загружено из {ref_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить эталонное решение: {e}")

        # Создание LossComputer (требует границы области)
        self.x_min = self.data_handler.dataset.x.min().item()
        self.x_max = self.data_handler.dataset.x.max().item()
        self.y_min = self.data_handler.dataset.y.min().item()
        self.y_max = self.data_handler.dataset.y.max().item()

        self.loss_computer = LossComputer(
            config=config,
            data_handler=self.data_handler,
            rho_norm=self.rho_norm,
            mu_norm=self.mu_norm,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            device=self.device,
            dtype=self.dtype
        )
        validate_runtime_contract(
            config,
            self.data_handler,
            self.loss_computer.feature_assembler,
            sdf_geometry_fingerprint=(
                self.loss_computer.sdf_geometry_fingerprint
            ),
        )
        self.training_contract = build_training_contract(
            config,
            self.data_handler,
            self.loss_computer,
        )
        logger.info(
            "Training contract: %s",
            self.training_contract.fingerprint()[:12],
        )

        # FVM-геометрия используется не только для FVM-loss, но и для
        # корректного интегрального баланса массы по граничным граням.
        self._setup_fvm_residuals()

        # SA-веса (если используются)
        self.sa_weights = None
        if config['loss_weights']['use_sa_weights']:
            n_eqns = 3 if not config['loss_weights']['sa_apply_to_bc'] else 3 + len(self.data_handler.boundary_type_names)
            self.sa_weights = SelfAdaptiveWeights(self.data_handler.n_points, n_eqns, init_val=1.0).to(self.device, self.dtype)

        # Модель
        self.model = self._build_model()
        self.model = self.model.to(self.device, self.dtype)
        if config['process'].get('num_gpus', 1) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(config['process']['num_gpus'])))
        self.model = self._maybe_compile_model(self.model)

        # Оптимизатор и планировщик
        self.optimizer = self._init_optimizer(optim_type='adam')
        self.scheduler = self._init_scheduler()

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Стартовая эпоха — будет перезаписана при загрузке чекпоинта
        self.start_epoch = 0

        # Диагностическая конфигурация
        self.diagnostics_config = config.get('diagnostics', {})
        self.diagnostics_interval = self.diagnostics_config.get('interval', 50)
        self.profile_sections = self.diagnostics_config.get('profile_sections', [0.2, 0.5, 0.8])
        self.compute_grad_cosine = self.diagnostics_config.get('compute_grad_cosine', False)
        # Параметр ограничения по y для профилей (доля от y_max)
        self.profile_y_max_ratio = self.diagnostics_config.get('profile_y_max_ratio', 0.95)
        # Включение двумерного спектра
        self.plot_2d_spectrum = self.diagnostics_config.get('plot_2d_spectrum', False)
        self.plot_1d_spectrum = self.diagnostics_config.get('plot_spectrum', True)
        self.compute_ntk_condition = self.diagnostics_config.get('compute_ntk_condition', False)
        self.ntk_sample_size = self.diagnostics_config.get('ntk_sample_size', 64)
        self.mass_metrics_interval = self.diagnostics_config.get('mass_metrics_interval', 1)
        # Разрешение сетки для 2D спектра
        self.spectrum_2d_resolution = self.diagnostics_config.get('spectrum_2d_resolution', 256)

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_computer=self.loss_computer,
            data_handler=self.data_handler,
            config=config['training'],
            device=self.device,
            dtype=self.dtype,
            sa_weights=self.sa_weights,
            scaler=self.scaler,
            using_lbfgs=False,
            diagnostics_config=self.diagnostics_config,
            use_amp=self.use_amp,
            plots_dir=self._get_plots_dir(),
        )

        # MetricsCollector
        self.metrics = MetricsCollector(
            model=self.model,
            data_handler=self.data_handler,
            loss_computer=self.loss_computer,
            config=config,
            device=self.device,
            dtype=self.dtype,
            logs_dir=self._get_logs_dir(),
            config_name=self.config_name,
            reference_data=self.reference_data
        )

        # Загрузка чекпоинта — после MetricsCollector, т.к. восстанавливает history
        self._load_checkpoint_if_needed()

        # Визуализация
        out_cfg = config['output']
        plot_freq = out_cfg['plot_frequency'] if not out_cfg['console_only'] else None
        self.visualizer = Visualizer(plot_freq, config_name=self.config_name, config=config)
        if plot_freq:
            self.visualizer.create_live_plot()

        # Клавиатурный слушатель
        self.key_listener = None
        if config.get('enable_key_listener', True):
            self.key_listener = KeyListener()
            self.key_listener.start()

        # CheckpointManager
        self.checkpoint_manager = CheckpointManager(self._get_checkpoint_dir())

        # InteractiveHandler
        self.interactive = InteractiveHandler(
            trainer=self.trainer,
            metrics_collector=self.metrics,
            visualizer=self.visualizer,
            checkpoint_manager=self.checkpoint_manager,
            key_listener=self.key_listener,
            loss_computer=self.loss_computer,
            data_handler=self.data_handler,
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            config=config,
            sa_weights=self.sa_weights,
            plots_dir=self._get_plots_dir()
        )

        # Параметры обучения
        self.two_stage = config['training']['two_stage']
        self.stage1_epochs = config['training']['stage1_epochs']
        self.stage2_pde_weight_start = config['training']['stage2_pde_weight_start']
        self.stage2_pde_weight_end = config['training']['stage2_pde_weight_end']
        self.stage2_ramp_epochs = config['training']['stage2_ramp_epochs']
        self.use_lbfgs = config['training']['switch_to_lbfgs_epoch'] > 0
        self.adam_epochs = config['training']['switch_to_lbfgs_epoch']

        # Early stopping
        early_cfg = config['training'].get('early_stopping', {})
        self.early_stopping_enabled = bool(early_cfg.get('enabled', False))
        self.early_stopping_patience = int(early_cfg.get('patience', 100))
        self.early_stopping_min_delta = float(early_cfg.get('min_delta', 1e-6))
        self.early_stopping_restore_best = bool(early_cfg.get('restore_best', True))
        self._early_best_loss = float('inf')
        self._early_best_epoch = 0
        self._early_bad_epochs = 0
        self._early_best_model_state = None
        self._early_best_sa_state = None

        # Детектор локальных минимумов
        self.minima_detector = None
        if config['training']['lr_schedule'] == 'detector':
            self.minima_detector = LocalMinimaDetector(
                window=20,
                threshold=1e-4,
                patience=config['training']['lr_detector_patience']
            )
            self.lr_boost_factor = config['training']['lr_detector_factor']
            self.lr_recovery_epochs = config['training']['lr_detector_recovery']

        # Время начала
        self.start_time = None

        # Состояние L-BFGS
        self.using_lbfgs = False

    # ------------------------------------------------------------------
    # Вспомогательные методы инициализации
    # ------------------------------------------------------------------
    def _init_device(self, proc: Dict) -> torch.device:
        dev = proc['device']
        if dev == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif dev == 'cuda':
            if not torch.cuda.is_available():
                import logging as _log
                _log.getLogger(__name__).warning(
                    "CUDA запрошена, но недоступна (torch без CUDA или нет GPU). "
                    "Переключение на CPU."
                )
                return torch.device('cpu')
            return torch.device('cuda:0')
        else:
            return torch.device(dev)

    def _init_dtype(self, dtype_str: str) -> torch.dtype:
        return {'float32': torch.float32, 'float64': torch.float64, 'float16': torch.float16}[dtype_str]

    def _build_model(self) -> PINNIncompressible:
        arch = self.config['architecture']
        # Определяем фактическую размерность входа
        sample_batch = next(iter(self.data_handler.get_dataloader()))
        idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = sample_batch
        case_ids = getattr(sample_batch, 'case_id', None)
        x = x.to(self.device)
        y = y.to(self.device)
        sdf = sdf.to(self.device)
        if bt.shape[1] > 0:
            bt = bt.to(self.device)
        else:
            bt = None
        input_tensor = self.loss_computer.build_input_tensor(
            x,
            y,
            sdf,
            bt,
            case_ids=case_ids,
        )
        actual_input_dim = input_tensor.shape[1]
        if actual_input_dim != self.loss_computer.feature_manifest.input_dim:
            raise RuntimeError(
                "Sample input width disagrees with the feature manifest: "
                f"{actual_input_dim} != "
                f"{self.loss_computer.feature_manifest.input_dim}"
            )

        # Получаем параметры масштабирования координат
        coord_scaling = arch.get('coordinate_scaling', {'enabled': False, 'scale_factor': 1.0}).copy()
        coord_scaling['coord_dim'] = 2 if self.config.get('input_params', {}).get('coordinates', True) else 0

        return PINNIncompressible(
            input_dim=actual_input_dim,
            layers=arch['hidden_layers'],
            activation=arch.get('activation', 'tanh'),
            output_dim=arch['output_dim'],
            output_bias=arch['output_bias'],
            coordinate_scaling=coord_scaling
        )

    def _maybe_compile_model(self, model: nn.Module) -> nn.Module:
        """Optionally compiles the model with torch.compile.

        PINN training differentiates outputs with respect to inputs, so the
        feature is opt-in and conservative by default.
        """
        proc = self.config.get('process', {})
        if not bool(proc.get('torch_compile', False)):
            return model

        if not hasattr(torch, 'compile'):
            logger.warning("torch_compile=True, но torch.compile недоступен в этой версии PyTorch.")
            return model

        if isinstance(model, nn.DataParallel):
            logger.warning("torch_compile=True пропущен для nn.DataParallel; используйте num_gpus=1.")
            return model

        compile_kwargs = {
            'fullgraph': bool(proc.get('torch_compile_fullgraph', False)),
            'dynamic': proc.get('torch_compile_dynamic'),
        }
        mode = proc.get('torch_compile_mode')
        if mode not in (None, '', 'default'):
            compile_kwargs['mode'] = mode

        try:
            compiled_model = torch.compile(model, **compile_kwargs)
        except Exception as exc:
            logger.warning("Не удалось включить torch.compile: %s", exc)
            return model

        logger.info(
            "Модель скомпилирована через torch.compile "
            "(mode=%s, fullgraph=%s, dynamic=%s)",
            mode or 'default',
            compile_kwargs['fullgraph'],
            compile_kwargs['dynamic'],
        )
        return compiled_model

    @staticmethod
    def _unwrap_model_for_state(model: nn.Module) -> nn.Module:
        """Returns the underlying module for stable state_dict operations."""
        if isinstance(model, nn.DataParallel):
            model = model.module
        return getattr(model, '_orig_mod', model)

    @staticmethod
    def _strip_state_prefix(
        state_dict: Dict[str, torch.Tensor],
        prefix: str
    ) -> Dict[str, torch.Tensor]:
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict

    def _validate_physics_metadata(self) -> None:
        """Проверяет, что пользовательские input_params не маскируют старые metadata."""
        inp = self.config.get('input_params', {})
        configured_fluid = inp.get('fluid_file')
        metadata_fluid = getattr(self.data_handler, 'fluid_name', None)

        if configured_fluid and metadata_fluid and str(configured_fluid) != str(metadata_fluid):
            logger.warning(
                "fluid_file='%s' в настройках отличается от fluid_name='%s' в initial_metadata.npy. "
                "PDE использует rho/mu/Re из metadata: rho=%.6g, mu=%.6g, Re=%.6g.",
                configured_fluid,
                metadata_fluid,
                self.data_handler.rho,
                self.data_handler.mu,
                self.data_handler.Re,
            )
        elif inp.get('density') or inp.get('viscosity'):
            logger.info(
                "Физические параметры для входов/PDE взяты из metadata: fluid=%s, rho=%.6g, mu=%.6g, Re=%.6g",
                metadata_fluid or "unknown",
                self.data_handler.rho,
                self.data_handler.mu,
                self.data_handler.Re,
            )

    def _setup_fvm_residuals(self) -> None:
        """Инициализирует FVMResiduals, если они нужны для loss или mass integral."""
        lw = self.config.get('loss_weights', {})
        fvm_cfg = lw.get('fvm', {})
        fvm_enabled = isinstance(fvm_cfg, dict) and bool(fvm_cfg.get('enabled', False))
        integral_enabled = bool(lw.get('integral_condition', False))
        need_fvm_geometry = fvm_enabled or integral_enabled or self.data_handler.mesh_data is not None

        if (
            fvm_enabled
            and self.loss_computer.flow_type == "axisymmetric"
        ):
            raise ValueError(
                "Axisymmetric FVM loss is not supported by the current "
                "planar dual-cell operator. Disable loss_weights.fvm or use "
                "the pointwise axisymmetric PDE/integral constraint; silently "
                "using planar face fluxes would violate conservation."
            )

        if not need_fvm_geometry:
            self.loss_computer.fvm = None
            return

        if self.data_handler.mesh_data is None:
            if fvm_enabled:
                logger.warning(
                    "FVM Mass flow включён (fvm.enabled=True), но mesh_file не задан — "
                    "FVM-потеря не будет вычисляться"
                )
            if integral_enabled:
                logger.warning(
                    "Integral condition включён, но FVM-сетки нет — будет использован "
                    "fallback по boundary-точкам."
                )
            self.loss_computer.fvm = None
            return

        try:
            from airfoil_pinn.physics.fvm import BC_CODES, FVMResiduals
            fvm = FVMResiduals.from_mesh_data(self.data_handler.mesh_data)
            fvm.to(self.device, dtype=self.dtype)
            fvm.nondimensionalize_geometry(self.data_handler.chord)
            inlet_u = getattr(self.data_handler, 'inlet_u_norm', 1.0)
            inlet_v = getattr(self.data_handler, 'inlet_v_norm', 0.0)
            velocity_bc = torch.tensor([inlet_u, inlet_v], dtype=self.dtype, device=self.device)
            fvm.set_inlet_velocity(velocity_bc)
            code = BC_CODES.get('velocity_dirichlet')
            if code is not None and bool((fvm.bnd_type == code).any()):
                fvm.set_bc_velocity('velocity_dirichlet', velocity_bc)
            self.loss_computer.fvm = fvm
            logger.info("FVMResiduals подключён: %d узлов", fvm.N)
        except Exception as e:
            self.loss_computer.fvm = None
            logger.warning("Не удалось инициализировать FVMResiduals: %s", e)

    def _init_optimizer(self, optim_type='adam') -> optim.Optimizer:
        opt_params = self.config.get('optimizer_params', {'lr': 1e-3})
        lr = opt_params.get('lr', 1e-3)
        weight_decay = float(opt_params.get('weight_decay', 0.0))
        params = [
            {
                'params': self.model.parameters(),
                'weight_decay': weight_decay,
            }
        ]
        if self.sa_weights is not None and optim_type == 'adam':
            for parameter in self.sa_weights.parameters():
                parameter.requires_grad_(True)
            params.append(
                {
                    'params': self.sa_weights.parameters(),
                    'lr': self.config.get('sa_lr', 0.001),
                    'weight_decay': 0.0,
                }
            )
        if optim_type == 'adam':
            return optim.Adam(params, lr=lr)
        else:  # LBFGS
            all_params = []
            for group in params:
                all_params.extend(group['params'])
            return optim.LBFGS(all_params, lr=lr,
                               max_iter=self.config['training']['lbfgs_max_iter'],
                               history_size=self.config['training']['lbfgs_history_size'],
                               line_search_fn='strong_wolfe')

    def _ensure_sa_weights_shape(self, sa_state: Dict[str, torch.Tensor]) -> None:
        """Пересоздаёт SA-веса, если чекпоинт сохранён после изменения размера датасета."""
        if self.sa_weights is None or 'log_weights' not in sa_state:
            return

        target_shape = tuple(sa_state['log_weights'].shape)
        current_shape = tuple(self.sa_weights.log_weights.shape)
        if current_shape == target_shape:
            return

        n_points, n_eqns = target_shape
        self.sa_weights = SelfAdaptiveWeights(n_points, n_eqns, init_val=1.0).to(self.device, self.dtype)
        self.optimizer = self._init_optimizer(optim_type='adam')
        self.scheduler = self._init_scheduler()
        self.trainer.sa_weights = self.sa_weights
        self.trainer.optimizer = self.optimizer
        logger.info(
            "SA-веса пересозданы под чекпоинт: %s -> %s",
            current_shape,
            target_shape,
        )

    def _init_scheduler(self):
        lr_schedule = self.config['training']['lr_schedule']
        if lr_schedule == 'cyclic':
            base_lr = self.optimizer.param_groups[0]['lr']
            max_lr = base_lr * (1 + self.config['training']['lr_cyclic_amplitude'])
            return CyclicLR(self.optimizer, base_lr, max_lr, self.config['training']['lr_cyclic_period'])
        elif lr_schedule == 'detector':
            return None
        elif lr_schedule == 'constant':
            return None
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                         factor=0.5, patience=50)

    def _load_checkpoint_if_needed(self):
        load_path = self.config['process'].get('load_model_path')
        if not load_path:
            return
        load_path = str(resolve_project_path(load_path))
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Requested checkpoint does not exist: {load_path}"
            )
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            checkpoint_cfg = self.config.get("checkpoint", {})
            contract_metadata = checkpoint.get("training_contract")
            if contract_metadata is None:
                if not bool(
                    checkpoint_cfg.get(
                        "allow_legacy_without_contract",
                        True,
                    )
                ):
                    raise RuntimeError(
                        "Legacy checkpoint has no training contract and "
                        "checkpoint.allow_legacy_without_contract=False"
                    )
                logger.warning(
                    "Legacy checkpoint has no training contract; only tensor "
                    "shape compatibility can be checked."
                )
            else:
                try:
                    validate_checkpoint_contract(
                        contract_metadata,
                        self.training_contract,
                    )
                except Exception:
                    if bool(checkpoint_cfg.get("strict_contract", True)):
                        raise
                    logger.warning(
                        "Checkpoint contract mismatch ignored because "
                        "checkpoint.strict_contract=False.",
                        exc_info=True,
                    )
            # Сначала восстанавливаем данные/RAR-сетку, чтобы размеры SA-весов и
            # optimizer state соответствовали фактическому датасету.
            if 'data_handler_state' in checkpoint:
                self.data_handler.load_state(checkpoint['data_handler_state'])
                self.x_min = self.data_handler.dataset.x.min().item()
                self.x_max = self.data_handler.dataset.x.max().item()
                self.y_min = self.data_handler.dataset.y.min().item()
                self.y_max = self.data_handler.dataset.y.max().item()
                self.loss_computer.x_min = self.x_min
                self.loss_computer.x_max = self.x_max
                self.loss_computer.y_min = self.y_min
                self.loss_computer.y_max = self.y_max
                self._setup_fvm_residuals()
                self.trainer._fvm_tensor_cache = None
            elif self.trainer.rar_enable:
                logger.warning(
                    "Чекпоинт не содержит data_handler_state. Если он был сохранён после RAR, "
                    "обучение продолжится на исходном датасете."
                )

            # Загрузка модели (адаптация ключей для DataParallel)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                state_dict = self._strip_state_prefix(state_dict, 'module.')
                state_dict = self._strip_state_prefix(state_dict, '_orig_mod.')
                self._unwrap_model_for_state(self.model).load_state_dict(state_dict)
                logger.info("Модель загружена из чекпоинта")

            # Загрузка SA-весов
            if 'sa_weights_state_dict' in checkpoint and self.sa_weights:
                self._ensure_sa_weights_shape(checkpoint['sa_weights_state_dict'])
                self.sa_weights.load_state_dict(checkpoint['sa_weights_state_dict'])
                self.trainer.sa_weights = self.sa_weights

            if 'trainer_state' in checkpoint:
                self.trainer.load_state(checkpoint['trainer_state'])

            # Optimizer загружается после возможного resize SA-весов.
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить состояние оптимизатора: {e}")
            # Загрузка scheduler
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить scheduler: {e}")
            # Загрузка истории и эпохи
            if 'history' in checkpoint:
                self.metrics.history = checkpoint['history']
            if 'epoch' in checkpoint and checkpoint['epoch'] is not None:
                self.start_epoch = checkpoint['epoch']
            logger.info(f"Чекпоинт загружен, продолжение с эпохи {self.start_epoch}")
        except Exception as e:
            logger.error("Ошибка загрузки чекпоинта: %s", e, exc_info=True)
            raise RuntimeError(
                f"Checkpoint resume failed for {load_path}: {e}"
            ) from e

    def _get_run_dir(self) -> str:
        """Возвращает корневую директорию текущего запуска (единый timestamp)."""
        base_dir = self.config.get('checkpoint_dir', project_relative(PATHS.experiments_dir))
        return os.path.join(str(resolve_project_path(base_dir)), f"{self.config_name}_{self._timestamp}")

    def _get_checkpoint_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "checkpoints")
        return os.path.join(self._get_run_dir(), "checkpoints")

    def _get_logs_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "logs")
        return os.path.join(self._get_run_dir(), "logs")

    def _get_plots_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "plots")
        return os.path.join(self._get_run_dir(), "plots")

    def _save_checkpoint(self, identifier: Union[int, str]) -> None:
        filename = f"pinn_{identifier}.pt"
        self.checkpoint_manager.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler if not self.using_lbfgs else None,
            loss_weights=self.loss_computer.loss_weights,
            history=self.metrics.history,
            sa_weights=self.sa_weights,
            epoch=identifier if isinstance(identifier, int) else None,
            config=self.config,
            extra_state={
                'training_contract': self.training_contract.to_metadata(),
                'data_handler_state': self.data_handler.get_state(),
                'trainer_state': self.trainer.get_state(),
                'using_lbfgs': self.using_lbfgs,
            }
        )

    def _save_checkpoint_flow_images(self, identifier: Union[int, str]) -> None:
        """Saves predicted flow-field maps next to periodic checkpoints."""
        checkpoint_cfg = self.config.get('checkpoint', {})
        if not checkpoint_cfg.get('save_flow_images', True):
            return

        try:
            pred_norm = self._get_predictions_norm()
            plots_subdir = str(checkpoint_cfg.get('flow_plot_subdir', 'flow') or 'flow')
            savedir = os.path.join(self._get_plots_dir(), plots_subdir)

            def as_column_tensor(name: str) -> torch.Tensor:
                return torch.from_numpy(pred_norm[name]).reshape(-1, 1)

            dataset = self.data_handler.dataset
            boundary_types = getattr(dataset, 'boundary_types', None)
            self.visualizer.plot_final_fields_from_tensors(
                as_column_tensor('u_norm'),
                as_column_tensor('v_norm'),
                as_column_tensor('p_norm'),
                as_column_tensor('x_norm'),
                as_column_tensor('y_norm'),
                self.data_handler,
                savedir,
                identifier,
                boundary_types=boundary_types,
                boundary_type_names=self.data_handler.boundary_type_names,
                save_npy=bool(checkpoint_cfg.get('save_flow_npy', False)),
                plot_physical=bool(checkpoint_cfg.get('flow_plot_physical', False)),
                config=self.config,
            )
            logger.info("Картины потока для чекпоинта %s сохранены в %s", identifier, savedir)
        except Exception as exc:
            logger.warning(
                "Не удалось сохранить картины потока для чекпоинта %s: %s",
                identifier,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Диагностическая процедура
    # ------------------------------------------------------------------
    def _run_diagnostics(self, epoch: int):
        """Сбор и сохранение диагностических данных в заданную эпоху."""
        logger.info(f"Запуск диагностики на эпохе {epoch}")
        plots_dir = self._get_plots_dir()
        need_predictions = (
            self.diagnostics_config.get('plot_profiles', True)
            or self.plot_1d_spectrum
            or self.plot_2d_spectrum
        )
        pred_norm = self._get_predictions_norm() if need_predictions else None

        # 1. Карта невязки PDE + карта полной ошибки, используемой RAR
        if self.diagnostics_config.get('plot_residual_map', True):
            x_np = self.data_handler.dataset.x.cpu().numpy().flatten() * self.data_handler.chord
            y_np = self.data_handler.dataset.y.cpu().numpy().flatten() * self.data_handler.chord
            residual_src = getattr(self.trainer, 'point_pde_residual', self.trainer.point_residual)
            residual = residual_src.numpy()
            if getattr(self.data_handler, 'mesh_domain_coords', None) is None:
                self.data_handler.load_mesh_geometry_from_geom_config()
            clip_polygon = getattr(self.data_handler, 'mesh_domain_coords', None)
            airfoil_polyline = getattr(self.data_handler, 'mesh_airfoil_coords', None)
            outline_polylines = [airfoil_polyline] if airfoil_polyline is not None else None
            self.visualizer.plot_residual_map(
                x_np, y_np, residual, plots_dir, epoch,
                clip_polygon=clip_polygon,
                mask_polygon=airfoil_polyline,
                outline_polylines=outline_polylines,
            )
            if self.diagnostics_config.get('plot_rar_error_map', True):
                rar_error = self.trainer.point_residual.numpy()
                self.visualizer.plot_rar_error_map(
                    x_np, y_np, rar_error, plots_dir, epoch,
                    clip_polygon=clip_polygon,
                    mask_polygon=airfoil_polyline,
                    outline_polylines=outline_polylines,
                )

        # 2. Профили скорости в нескольких сечениях с фильтрацией по y
        if self.diagnostics_config.get('plot_profiles', True) and pred_norm is not None:
            u_pred_norm = pred_norm['u_norm']  # одномерный массив numpy
            profiles = {}
            for x_sec in self.profile_sections:
                x_norm = x_sec  # profile_sections — уже нормированные доли хорды (0..1)
                # Передаём предсказанные значения u_norm в interpolate_profile
                y_grid, u_interp_norm = self.data_handler.interpolate_profile(
                    x_norm,
                    u_vals=torch.from_numpy(u_pred_norm).to(dtype=self.dtype),
                    num_points=200,
                    y_max_ratio=self.profile_y_max_ratio
                )
                if np.all(np.isnan(u_interp_norm)):
                    logger.warning(f"Сечение x={x_sec} пропущено: нет данных для интерполяции.")
                    continue
                y_phys = y_grid * self.data_handler.chord
                u_phys = u_interp_norm * self.data_handler.U_inf
                profiles[x_sec] = (y_phys, u_phys)
            self.visualizer.plot_velocity_profiles_multiple(profiles, plots_dir, epoch)

        # 3. Одномерный спектр в заданном сечении
        if self.plot_1d_spectrum and pred_norm is not None:
            x_sec = self.diagnostics_config.get('spectrum_section_x', 0.5)
            y_grid, u_interp_norm = self.data_handler.interpolate_profile(
                x_sec,
                u_vals=torch.from_numpy(pred_norm['u_norm']).to(dtype=self.dtype),
                num_points=256,
                y_max_ratio=self.profile_y_max_ratio
            )
            if not np.all(np.isnan(u_interp_norm)):
                self.visualizer.plot_1d_spectrum(
                    y_grid * self.data_handler.chord,
                    u_interp_norm * self.data_handler.U_inf,
                    plots_dir,
                    epoch,
                    x_section=x_sec * self.data_handler.chord
                )
            else:
                logger.warning(f"1D спектр пропущен: нет данных в сечении x={x_sec}.")

        # 4. Двумерный спектр (если включено) – с фильтрацией граничных точек
        if self.plot_2d_spectrum and pred_norm is not None:
            x_np = pred_norm['x_norm'] * self.data_handler.chord
            y_np = pred_norm['y_norm'] * self.data_handler.chord
            u_phys = pred_norm['u_norm'] * self.data_handler.U_inf

            # Исключаем граничные точки (отступ 5% от каждой границы)
            eps_x = 0.05 * (x_np.max() - x_np.min())
            eps_y = 0.05 * (y_np.max() - y_np.min())
            mask = (x_np > x_np.min() + eps_x) & (x_np < x_np.max() - eps_x) & \
                   (y_np > y_np.min() + eps_y) & (y_np < y_np.max() - eps_y)

            x_filtered = x_np[mask]
            y_filtered = y_np[mask]
            u_filtered = u_phys[mask]

            self.visualizer.plot_2d_spectrum(
                x_filtered, y_filtered, u_filtered, plots_dir, epoch,
                nx=self.spectrum_2d_resolution, ny=self.spectrum_2d_resolution
            )

        # 4. Раздельные градиенты (если включено)
        if self.compute_grad_cosine:
            # Берём один батч из даталоадера
            sample_batch = next(iter(self.data_handler.get_dataloader()))
            grad_metrics = self.trainer.compute_grad_norms_separately(sample_batch)
            if 'grad_norm_pde' in self.metrics.history and len(self.metrics.history['grad_norm_pde']) > 0:
                self.visualizer.plot_gradient_norms_separate(
                    self.metrics.history['grad_norm_pde'],
                    self.metrics.history['grad_norm_bc'],
                    self.metrics.history['grad_cosine'],
                    plots_dir, epoch
                )

        # 5. Гистограмма RAR
        if self.diagnostics_config.get('save_rar_history', True) and hasattr(self.trainer, 'rar_history') and self.trainer.rar_history:
            self.visualizer.plot_rar_histogram(self.trainer.rar_history, self.data_handler.boundary_type_names, plots_dir, epoch)

        # 6. Диагностика пограничного слоя по нормальным лучам от профиля.
        # Этот блок ничего не добавляет в loss: он только сохраняет профили,
        # Cp/Cf и численные признаки разрешённости пристенного слоя.
        bl_result = run_boundary_layer_diagnostics(
            model=self.model,
            loss_computer=self.loss_computer,
            data_handler=self.data_handler,
            device=self.device,
            dtype=self.dtype,
            savedir=plots_dir,
            epoch=epoch,
            config=self.diagnostics_config.get('boundary_layer', {}),
        )
        if bl_result is not None:
            s = bl_result.summary
            self.metrics.save_log(
                "BL diagnostics: "
                f"profiles={int(s.get('profile_count', 0))}, "
                f"unresolved={int(s.get('unresolved_profiles', 0))}, "
                f"u_first/Uedge max={s.get('u_first_over_Uedge_max', float('nan')):.3e}, "
                f"delta99 mean={s.get('delta99_mean', float('nan')):.3e}, "
                f"Cf rms={s.get('Cf_rms', float('nan')):.3e}"
            )

    def _get_predictions_norm(self) -> Dict[str, np.ndarray]:
        """
        Возвращает предсказания модели на всём датасете в нормализованных единицах.
        Возвращает словарь с ключами 'x_norm', 'y_norm', 'u_norm', 'v_norm', 'p_norm'.
        """
        was_training = self.model.training
        self.model.eval()
        dataloader = self.data_handler.get_eval_dataloader()
        all_u = []
        all_v = []
        all_p = []
        all_x = []
        all_y = []

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                case_ids = getattr(batch, 'case_id', None)
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = sdf.to(self.device, dtype=self.dtype)
                if bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None

                input_tensor = self.loss_computer.build_input_tensor(
                    x,
                    y,
                    sdf,
                    bt,
                    case_ids=case_ids,
                )
                out = self.model(input_tensor)
                u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

                # Применяем hard constraints, если нужно
                if self.loss_computer.bc_type == 'hard':
                    mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
                    mask_v = mask_v.to(self.device, dtype=torch.bool).view(-1, 1)
                    mask_p = mask_p.to(self.device, dtype=torch.bool).view(-1, 1)
                    u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    v_bc = v_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    p_bc = p_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                    u, v, p = self.loss_computer.apply_hard_constraints(
                        u_raw, v_raw, p_raw, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc
                    )
                else:
                    u, v, p = u_raw, v_raw, p_raw
                u, v = self.loss_computer.apply_velocity_dirichlet_constraint(
                    u,
                    v,
                    bt,
                    case_ids=case_ids,
                )

                all_u.append(u.cpu().numpy().flatten())
                all_v.append(v.cpu().numpy().flatten())
                all_p.append(p.cpu().numpy().flatten())
                all_x.append(x.cpu().numpy().flatten())
                all_y.append(y.cpu().numpy().flatten())

        if was_training:
            self.model.train()
        return {
            'x_norm': np.concatenate(all_x),
            'y_norm': np.concatenate(all_y),
            'u_norm': np.concatenate(all_u),
            'v_norm': np.concatenate(all_v),
            'p_norm': np.concatenate(all_p)
        }

    # ------------------------------------------------------------------
    # Основной цикл обучения
    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None):
        if epochs is None:
            epochs = self.config['training']['epochs']

        logger.info("🚀 Начало обучения...")
        sys.stdout.flush()

        self.start_time = time.time()
        epoch = self.start_epoch  # на случай Ctrl+C до старта цикла

        try:
            for epoch in range(self.start_epoch + 1, epochs + 1):
                start_time = time.time()

                # Мониторинг памяти (каждые 10 эпох)
                if epoch % 10 == 0:
                    free_mem = check_memory_usage(self.device, return_free=True)
                    if free_mem is not None and free_mem < self.memory_threshold_gb:
                        logger.warning(f"⚠️ Мало памяти ({free_mem:.2f} ГБ). Прерываю обучение.")
                        self._save_checkpoint('interrupted')
                        break

                # Множитель PDE для двухэтапного обучения
                pde_mult = self._get_pde_multiplier(epoch)

                # Переключение на L-BFGS
                if self.use_lbfgs and not self.using_lbfgs and epoch > self.adam_epochs:
                    self._switch_to_lbfgs()

                # Шаг обучения
                if self.using_lbfgs:
                    loss, pde_raw, bc_comp, extra_grad, integral_loss, wall_speed, pde_max, pde_rate, grad_norm = \
                        self.trainer.train_epoch_lbfgs()
                else:
                    loss, pde_raw, bc_comp, extra_grad, integral_loss, wall_speed, pde_max, pde_rate, grad_norm = \
                        self.trainer.train_epoch_adam(pde_mult)

                # Сбор дополнительных метрик для диагностики (если нужно)
                extra_diag = {'fvm_loss': self.trainer._last_fvm_loss}
                sample_batch = None
                if self.sa_weights is not None:
                    with torch.no_grad():
                        extra_diag['sa_weights_mean'] = self.sa_weights().mean().item()
                if self.compute_grad_cosine and not self.using_lbfgs and epoch % self.diagnostics_interval == 0:
                    # Берём один батч
                    sample_batch = next(iter(self.data_handler.get_dataloader()))
                    grad_metrics = self.trainer.compute_grad_norms_separately(sample_batch)
                    extra_diag['grad_norm_pde'] = grad_metrics['pde_norm']
                    extra_diag['grad_norm_bc'] = grad_metrics['bc_norm']
                    extra_diag['grad_cosine'] = grad_metrics['cosine']
                if self.compute_ntk_condition and not self.using_lbfgs and epoch % self.diagnostics_interval == 0:
                    if sample_batch is None:
                        sample_batch = next(iter(self.data_handler.get_dataloader()))
                    extra_diag['ntk_condition'] = self.trainer.compute_ntk_condition(
                        sample_batch,
                        max_samples=self.ntk_sample_size
                    )

                # Средние скорости на inlet/outlet для мониторинга баланса масс
                avg_u_in, avg_u_out = self.metrics.get_boundary_avg_velocities(
                    epoch,
                    interval=self.mass_metrics_interval
                )
                mass_metrics = self.metrics.get_mass_flow_metrics(
                    epoch,
                    interval=self.mass_metrics_interval
                )
                extra_diag['avg_u_in'] = avg_u_in
                extra_diag['avg_u_out'] = avg_u_out
                extra_diag.update(mass_metrics)

                # Обновление метрик
                self.metrics.update_history(
                    loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max, pde_rate, grad_norm,
                    extra_diag=extra_diag
                )

                epoch_time = time.time() - start_time
                total_time = time.time() - self.start_time

                # Адаптивные веса BC (активируется через config['loss_weights']['adaptive_bc'])
                if epoch % 50 == 0 and self.loss_computer.adaptive_bc:
                    self.trainer.update_adaptive_weights(self.loss_computer.loss_weights)

                # Периодический сброс оптимизатора с шумом
                if (
                    self.trainer.reset_interval > 0
                    and not self.using_lbfgs
                    and epoch % self.trainer.reset_interval == 0
                ):
                    self.trainer.reset_optimizer(reset_noise=self.trainer.reset_noise)
                    self._recreate_optimizer()

                # RAR
                if self.trainer.rar_enable and not self.using_lbfgs and epoch % self.trainer.rar_interval == 0:
                    added_rar_points = self.trainer.apply_rar()
                    if added_rar_points:
                        self.metrics.invalidate_boundary_cache()
                    if (
                        added_rar_points
                        or (hasattr(self.trainer, 'sa_weights_changed') and self.trainer.sa_weights_changed)
                    ):
                        self._recreate_optimizer()
                        self.trainer.sa_weights_changed = False

                # Детектор локальных минимумов
                if self.minima_detector is not None and self.minima_detector.update(loss):
                    logger.info("Обнаружено плато, увеличиваю learning rate")
                    for _ in range(self.lr_recovery_epochs):
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.lr_boost_factor

                # Обработка горячих клавиш
                if self.key_listener:
                    self.interactive.handle_keys(epoch)
                    if self.interactive.optimizer_reset_requested:
                        self._recreate_optimizer()
                        self.interactive.optimizer_reset_requested = False

                # Обновление графиков
                if self.visualizer.enabled and epoch % self.visualizer.plot_frequency == 0:
                    self.visualizer.update_plot(epoch, self.metrics.history)

                # Шаг планировщика (если используется и не L-BFGS)
                if self.scheduler is not None and not self.using_lbfgs:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss)
                    elif isinstance(self.scheduler, CyclicLR):
                        self.scheduler.step()
                    # Другие типы планировщиков (например, StepLR) должны быть обработаны аналогично

                # Вывод в консоль (avg_u_in/out уже вычислены выше)
                self.metrics.print_epoch_table(
                    epoch, epochs, loss, pde_raw, bc_comp, integral_loss, pde_max, epoch_time, total_time,
                    fvm_loss=self.trainer._last_fvm_loss
                )
                line2 = (
                    f"   avg u_inlet = {avg_u_in:.4f} (норм.), "
                    f"avg u_outlet = {avg_u_out:.4f} (норм.), "
                    f"mass err = {mass_metrics['mass_balance_error']:.3e}"
                )
                print(line2)
                self.metrics.save_log(f"Epoch {epoch}: loss={loss:.2e}, " +
                                      ", ".join([f"{k}={v:.2e}" for k, v in pde_raw.items()]))
                self.metrics.save_log(line2)

                # Сохранение чекпоинта
                checkpoint_interval = self.config['checkpoint']['interval']
                if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                    self._save_checkpoint(epoch)
                    self.visualizer.save_plots(self._get_plots_dir(), epoch)
                    self._save_checkpoint_flow_images(epoch)

                # Диагностика
                if epoch % self.diagnostics_interval == 0:
                    self._run_diagnostics(epoch)

                if self._update_early_stopping(epoch, loss):
                    break

                # Очистка кэша CUDA (редко, чтобы не замедлять)
                if self.device.type == 'cuda' and epoch % 50 == 0:
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.warning("Обучение прервано пользователем. Сохраняю модель и метрики...")
            self._save_checkpoint('interrupted')
            self.visualizer.save_plots(self._get_plots_dir(), 'interrupted')
            self.interactive._save_current_results(epoch)
            self._finalize()
            return
        except Exception as e:
            logger.error(f"Критическая ошибка во время обучения: {e}", exc_info=True)
            self._save_checkpoint('error')
            self._finalize()
            raise
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self._save_checkpoint('final')
        self.visualizer.save_plots(self._get_plots_dir(), 'final')
        self.interactive._save_current_results(epoch)
        self._finalize()

    def _get_pde_multiplier(self, epoch: int) -> float:
        if not self.two_stage:
            return 1.0
        if epoch <= self.stage1_epochs:
            return 0.0
        ramp_progress = (epoch - self.stage1_epochs) / max(1, self.stage2_ramp_epochs)
        ramp_progress = min(1.0, max(0.0, ramp_progress))
        return self.stage2_pde_weight_start + ramp_progress * (self.stage2_pde_weight_end - self.stage2_pde_weight_start)

    def _capture_trainable_state(self) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        base_model = self._unwrap_model_for_state(self.model)
        model_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        sa_state = None
        if self.sa_weights is not None:
            sa_state = {k: v.detach().cpu().clone() for k, v in self.sa_weights.state_dict().items()}
        return model_state, sa_state

    def _restore_trainable_state(
        self,
        model_state: Dict[str, torch.Tensor],
        sa_state: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        base_model = self._unwrap_model_for_state(self.model)
        base_model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()})
        if self.sa_weights is not None and sa_state is not None:
            self.sa_weights.load_state_dict({k: v.to(self.device) for k, v in sa_state.items()})

    def _update_early_stopping(self, epoch: int, loss: float) -> bool:
        if not self.early_stopping_enabled:
            return False

        if loss < self._early_best_loss - self.early_stopping_min_delta:
            self._early_best_loss = loss
            self._early_best_epoch = epoch
            self._early_bad_epochs = 0
            if self.early_stopping_restore_best:
                self._early_best_model_state, self._early_best_sa_state = self._capture_trainable_state()
            return False

        self._early_bad_epochs += 1
        if self._early_bad_epochs < self.early_stopping_patience:
            return False

        logger.info(
            "Early stopping: нет улучшения %d эпох. Лучшая эпоха: %d, loss=%.4e",
            self.early_stopping_patience,
            self._early_best_epoch,
            self._early_best_loss,
        )
        if (
            self.early_stopping_restore_best
            and self._early_best_model_state is not None
        ):
            self._restore_trainable_state(self._early_best_model_state, self._early_best_sa_state)
            logger.info("Восстановлено лучшее состояние модели перед финальным сохранением.")
        return True

    def _switch_to_lbfgs(self):
        logger.info(f"Переключение на L-BFGS после эпохи {self.adam_epochs}")
        if self.sa_weights is not None:
            for parameter in self.sa_weights.parameters():
                parameter.requires_grad_(False)
            logger.info(
                "SA-PINN attention weights are frozen for L-BFGS; their "
                "min-max ascent is performed only during the Adam stage."
            )
        self.optimizer = self._init_optimizer(optim_type='lbfgs')
        self.scheduler = None
        self.using_lbfgs = True
        self.use_amp = False
        self.scaler = None
        self.trainer.optimizer = self.optimizer
        self.trainer.using_lbfgs = True
        self.trainer.use_amp = False
        self.trainer.scaler = None

    def _recreate_optimizer(self):
        if self.using_lbfgs:
            return
        current_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = self._init_optimizer(optim_type='adam')
        for pg in self.optimizer.param_groups:
            pg['lr'] = current_lr
        self.scheduler = self._init_scheduler()
        self.trainer.optimizer = self.optimizer
        logger.info("Оптимизатор пересоздан.")

    def _finalize(self):
        if self.key_listener:
            self.key_listener.stop()
        self.visualizer.close()
        self.metrics.close()
        plt.close('all')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
