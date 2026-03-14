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
- Для карты невязки используется point_residual (скользящее среднее)
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

from utils import check_memory_usage, set_seed
from data_handler import DataHandler
from models import PINNIncompressible, SelfAdaptiveWeights
from checkpoint import CheckpointManager
from key_listener import KeyListener
from visualizer import Visualizer
from reference import load_reference, interpolate_reference, compute_l2_errors

# Компоненты из solver_components
from solver_components.optimizers import LocalMinimaDetector, CyclicLR
from solver_components.loss_computer import LossComputer
from solver_components.trainer import Trainer
from solver_components.metrics import MetricsCollector
from solver_components.interactive import InteractiveHandler

logger = logging.getLogger(__name__)


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
                           (если None, создаётся уникальная папка в Checkpoints)
        """
        self.config = config
        self.config_name = config.get('project_name', 'unnamed')
        self.experiment_dir = experiment_dir

        # Установка seed
        seed = config.get('process', {}).get('seed', 42)
        set_seed(seed)

        # Устройство и тип данных
        self.device = self._init_device(config['process'])
        self.dtype = self._init_dtype(config['process']['dtype'])
        self.use_amp = config['process'].get('use_amp', False)

        # Параметры мониторинга памяти
        self.memory_threshold_gb = config.get('process', {}).get('memory_threshold_gb', 0.5)

        # Загрузка данных
        self.data_handler = DataHandler(
            points_file=config['data']['points_file'],
            batch_size=config['training']['batch_size'],
            shuffle=True
        )

        # Константы PDE (нормированные)
        self.rho_norm = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.mu_norm = torch.tensor(1.0 / self.data_handler.Re, device=self.device, dtype=self.dtype)

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

        # Оптимизатор и планировщик
        self.optimizer = self._init_optimizer(optim_type='adam')
        self.scheduler = self._init_scheduler()

        # Исправление FutureWarning для GradScaler
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Загрузка из чекпоинта (если указано)
        self.start_epoch = 0
        self._load_checkpoint_if_needed()

        # Диагностическая конфигурация
        self.diagnostics_config = config.get('diagnostics', {})
        self.diagnostics_interval = self.diagnostics_config.get('interval', 50)
        self.profile_sections = self.diagnostics_config.get('profile_sections', [0.2, 0.5, 0.8])
        self.compute_grad_cosine = self.diagnostics_config.get('compute_grad_cosine', False)
        # Параметр ограничения по y для профилей (доля от y_max)
        self.profile_y_max_ratio = self.diagnostics_config.get('profile_y_max_ratio', 0.95)
        # Включение двумерного спектра
        self.plot_2d_spectrum = self.diagnostics_config.get('plot_2d_spectrum', False)
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
            diagnostics_config=self.diagnostics_config
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

        # Визуализация
        out_cfg = config['output']
        plot_freq = out_cfg['plot_frequency'] if not out_cfg['console_only'] else None
        self.visualizer = Visualizer(plot_freq, config_name=self.config_name)
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
        x = x.to(self.device)
        y = y.to(self.device)
        sdf = sdf.to(self.device)
        if bt.shape[1] > 0:
            bt = bt.to(self.device)
        else:
            bt = None
        input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
        actual_input_dim = input_tensor.shape[1]

        return PINNIncompressible(
            input_dim=actual_input_dim,
            layers=arch['hidden_layers'],
            activation=arch.get('activation', 'tanh'),
            output_dim=arch['output_dim'],
            output_bias=arch['output_bias']
        )

    def _init_optimizer(self, optim_type='adam') -> optim.Optimizer:
        opt_params = self.config.get('optimizer_params', {'lr': 1e-3})
        lr = opt_params.get('lr', 1e-3)
        params = [{'params': self.model.parameters()}]
        if self.sa_weights is not None:
            params.append({'params': self.sa_weights.parameters(), 'lr': self.config.get('sa_lr', 0.001)})
        if optim_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=opt_params.get('weight_decay', 0))
        else:  # LBFGS
            all_params = []
            for group in params:
                all_params.extend(group['params'])
            return optim.LBFGS(all_params, lr=lr,
                               max_iter=self.config['training']['lbfgs_max_iter'],
                               history_size=self.config['training']['lbfgs_history_size'],
                               line_search_fn='strong_wolfe')

    def _init_scheduler(self):
        lr_schedule = self.config['training']['lr_schedule']
        if lr_schedule == 'cyclic':
            base_lr = self.optimizer.param_groups[0]['lr']
            max_lr = base_lr * (1 + self.config['training']['lr_cyclic_amplitude'])
            return CyclicLR(self.optimizer, base_lr, max_lr, self.config['training']['lr_cyclic_period'])
        elif lr_schedule == 'detector':
            return None
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                         factor=0.5, patience=50)

    def _load_checkpoint_if_needed(self):
        load_path = self.config['process'].get('load_model_path')
        if not load_path or not os.path.exists(load_path):
            return
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            # Загрузка модели (адаптация ключей для DataParallel)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if isinstance(self.model, nn.DataParallel):
                    if all(not k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {'module.' + k: v for k, v in state_dict.items()}
                else:
                    if all(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                logger.info("Модель загружена из чекпоинта")
            # Загрузка оптимизатора
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
            # Загрузка SA-весов
            if 'sa_weights_state_dict' in checkpoint and self.sa_weights:
                self.sa_weights.load_state_dict(checkpoint['sa_weights_state_dict'])
            # Загрузка истории и эпохи
            if 'history' in checkpoint:
                self.metrics.history = checkpoint['history']
            if 'epoch' in checkpoint and checkpoint['epoch'] is not None:
                self.start_epoch = checkpoint['epoch']
            logger.info(f"Чекпоинт загружен, продолжение с эпохи {self.start_epoch}")
        except Exception as e:
            logger.error(f"Ошибка загрузки чекпоинта: {e}")

    def _get_checkpoint_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "checkpoints")
        else:
            base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_dir = os.path.join(base_dir, f"{self.config_name}_{timestamp}")
            return os.path.join(unique_dir, "checkpoints")

    def _get_logs_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "logs")
        else:
            base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_dir = os.path.join(base_dir, f"{self.config_name}_{timestamp}")
            return os.path.join(unique_dir, "logs")

    def _get_plots_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "plots")
        else:
            base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_dir = os.path.join(base_dir, f"{self.config_name}_{timestamp}")
            return os.path.join(unique_dir, "plots")

    def _save_checkpoint(self, identifier: Union[int, str]) -> None:
        filename = f"pinn_{identifier}.pt" if isinstance(identifier, int) else f"pinn_{identifier}.pt"
        self.checkpoint_manager.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler if not self.using_lbfgs else None,
            loss_weights=self.loss_computer.loss_weights,
            history=self.metrics.history,
            sa_weights=self.sa_weights,
            epoch=identifier if isinstance(identifier, int) else None,
            config=self.config
        )

    # ------------------------------------------------------------------
    # Диагностическая процедура
    # ------------------------------------------------------------------
    def _run_diagnostics(self, epoch: int):
        """Сбор и сохранение диагностических данных в заданную эпоху."""
        logger.info(f"Запуск диагностики на эпохе {epoch}")
        plots_dir = self._get_plots_dir()

        # 1. Карта невязки PDE (используем скользящее среднее point_residual вместо max_residual)
        if self.diagnostics_config.get('plot_residual_map', True):
            residual = self.trainer.point_residual.numpy()  # скользящее среднее невязки
            x_np = self.data_handler.dataset.x.cpu().numpy().flatten() * self.data_handler.chord
            y_np = self.data_handler.dataset.y.cpu().numpy().flatten() * self.data_handler.chord
            self.visualizer.plot_residual_map(x_np, y_np, residual, plots_dir, epoch)

        # 2. Профили скорости в нескольких сечениях с фильтрацией по y
        if self.diagnostics_config.get('plot_profiles', True):
            # Получаем предсказания модели на всём датасете (нормализованные значения)
            pred_norm = self._get_predictions_norm()
            u_pred_norm = pred_norm['u_norm']  # одномерный массив numpy
            profiles = {}
            for x_sec in self.profile_sections:
                x_norm = x_sec / self.data_handler.chord
                # Передаём предсказанные значения u_norm в interpolate_profile
                y_grid, u_interp_norm = self.data_handler.interpolate_profile(
                    x_norm,
                    u_vals=torch.from_numpy(u_pred_norm).float(),  # конвертируем обратно в тензор
                    num_points=200,
                    y_max_ratio=self.profile_y_max_ratio
                )
                y_phys = y_grid * self.data_handler.chord
                u_phys = u_interp_norm * self.data_handler.U_inf
                profiles[x_sec] = (y_phys, u_phys)
            self.visualizer.plot_velocity_profiles_multiple(profiles, plots_dir, epoch)

        # 3. Двумерный спектр (если включено) – с фильтрацией граничных точек
        if self.plot_2d_spectrum:
            # Используем предсказанные поля u (физические) из модели
            # Получаем все точки датасета
            dataset = self.data_handler.dataset
            x_np = dataset.x.cpu().numpy().flatten() * self.data_handler.chord
            y_np = dataset.y.cpu().numpy().flatten() * self.data_handler.chord
            # Получаем предсказания модели для всех точек
            self.model.eval()
            with torch.no_grad():
                x_t = dataset.x.to(self.device, dtype=self.dtype)
                y_t = dataset.y.to(self.device, dtype=self.dtype)
                sdf_t = dataset.sdf.to(self.device, dtype=self.dtype)
                bt_t = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None
                input_tensor = self.loss_computer.build_input_tensor(x_t, y_t, sdf_t, bt_t)
                out = self.model(input_tensor)
                u_norm = out[:, 0].cpu().numpy().flatten()
                # Переводим в физические единицы
                u_phys = u_norm * self.data_handler.U_inf

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
            if hasattr(self.metrics.history, 'grad_norm_pde') and len(self.metrics.history['grad_norm_pde']) > 0:
                self.visualizer.plot_gradient_norms_separate(
                    self.metrics.history['grad_norm_pde'],
                    self.metrics.history['grad_norm_bc'],
                    self.metrics.history['grad_cosine'],
                    plots_dir, epoch
                )

        # 5. Гистограмма RAR
        if self.diagnostics_config.get('save_rar_history', True) and hasattr(self.trainer, 'rar_history') and self.trainer.rar_history:
            self.visualizer.plot_rar_histogram(self.trainer.rar_history, self.data_handler.boundary_type_names, plots_dir, epoch)

    def _get_predictions_norm(self) -> Dict[str, np.ndarray]:
        """
        Возвращает предсказания модели на всём датасете в нормализованных единицах.
        Возвращает словарь с ключами 'x_norm', 'y_norm', 'u_norm', 'v_norm', 'p_norm'.
        """
        self.model.eval()
        dataloader = self.data_handler.get_dataloader()
        all_u = []
        all_v = []
        all_p = []
        all_x = []
        all_y = []

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = sdf.to(self.device, dtype=self.dtype)
                if bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None

                input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
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

                all_u.append(u.cpu().numpy().flatten())
                all_v.append(v.cpu().numpy().flatten())
                all_p.append(p.cpu().numpy().flatten())
                all_x.append(x.cpu().numpy().flatten())
                all_y.append(y.cpu().numpy().flatten())

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
                extra_diag = {}
                if self.compute_grad_cosine and not self.using_lbfgs and epoch % self.diagnostics_interval == 0:
                    # Берём один батч
                    sample_batch = next(iter(self.data_handler.get_dataloader()))
                    grad_metrics = self.trainer.compute_grad_norms_separately(sample_batch)
                    extra_diag['grad_norm_pde'] = grad_metrics['pde_norm']
                    extra_diag['grad_norm_bc'] = grad_metrics['bc_norm']
                    extra_diag['grad_cosine'] = grad_metrics['cosine']

                # Вычисляем avg_u_in, avg_u_out, mass_balance_error (уже в metrics)
                avg_u_in, avg_u_out = self.metrics.get_boundary_avg_velocities(epoch)
                Qin = avg_u_in
                Qout = avg_u_out
                mass_error = abs(Qin - Qout) / (abs(Qin) + 1e-12) if Qin != 0 else 0.0
                extra_diag['avg_u_in'] = avg_u_in
                extra_diag['avg_u_out'] = avg_u_out
                extra_diag['mass_balance_error'] = mass_error

                # Обновление метрик
                self.metrics.update_history(
                    loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max, pde_rate, grad_norm,
                    extra_diag=extra_diag
                )

                epoch_time = time.time() - start_time
                total_time = time.time() - self.start_time

                # Адаптивные веса BC
                if epoch % 50 == 0 and self.loss_computer.loss_weights.get('adaptive_bc', False):
                    self.trainer.update_adaptive_weights(self.loss_computer.loss_weights)

                # RAR
                if self.trainer.rar_enable and not self.using_lbfgs and epoch % self.trainer.rar_interval == 0:
                    self.trainer.apply_rar()
                    if hasattr(self.trainer, 'sa_weights_changed') and self.trainer.sa_weights_changed:
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

                # Вывод в консоль
                avg_u_in, avg_u_out = self.metrics.get_boundary_avg_velocities(epoch)
                self.metrics.print_epoch_table(
                    epoch, epochs, loss, pde_raw, bc_comp, integral_loss, pde_max, epoch_time, total_time
                )
                line2 = f"   avg u_inlet = {avg_u_in:.4f} (норм.), avg u_outlet = {avg_u_out:.4f} (норм.)"
                print(line2)
                self.metrics.save_log(f"Epoch {epoch}: loss={loss:.2e}, " +
                                      ", ".join([f"{k}={v:.2e}" for k, v in pde_raw.items()]))
                self.metrics.save_log(line2)

                # Сохранение чекпоинта
                checkpoint_interval = self.config['checkpoint']['interval']
                if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                    self._save_checkpoint(epoch)
                    self.visualizer.save_plots(self._get_plots_dir(), epoch)

                # Диагностика
                if epoch % self.diagnostics_interval == 0:
                    self._run_diagnostics(epoch)

                # Очистка кэша CUDA (редко, чтобы не замедлять)
                if self.device.type == 'cuda' and epoch % 50 == 0:
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.warning("Обучение прервано пользователем. Сохраняю модель и метрики...")
            self._save_checkpoint('interrupted')
            self.visualizer.save_plots(self._get_plots_dir(), 'interrupted')
            self.interactive._save_current_results(epoch)
            self._finalize()
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
        self.interactive._save_current_results(epochs)
        self._finalize()

    def _get_pde_multiplier(self, epoch: int) -> float:
        if not self.two_stage:
            return 1.0
        if epoch <= self.stage1_epochs:
            return 0.0
        ramp_progress = (epoch - self.stage1_epochs) / max(1, self.stage2_ramp_epochs)
        ramp_progress = min(1.0, max(0.0, ramp_progress))
        return self.stage2_pde_weight_start + ramp_progress * (self.stage2_pde_weight_end - self.stage2_pde_weight_start)

    def _switch_to_lbfgs(self):
        logger.info(f"Переключение на L-BFGS после эпохи {self.adam_epochs}")
        self.optimizer = self._init_optimizer(optim_type='lbfgs')
        self.scheduler = None
        self.using_lbfgs = True
        self.use_amp = False
        self.scaler = None
        self.trainer.optimizer = self.optimizer
        self.trainer.using_lbfgs = True
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
        self.trainer.scheduler = self.scheduler
        logger.info("Оптимизатор пересоздан.")

    def _finalize(self):
        if self.key_listener:
            self.key_listener.stop()
        self.visualizer.close()
        self.metrics.close()
        plt.close('all')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()