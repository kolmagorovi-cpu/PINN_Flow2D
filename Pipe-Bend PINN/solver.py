"""
Основной решатель PINN для Pipe-Bend PINN.

Координирует работу компонентов: LossComputer, Trainer, MetricsCollector, InteractiveHandler.

Исправленная версия:
- Правильные границы области из data_handler
- Добавлены: _load_checkpoint_if_needed, _switch_to_lbfgs, _recreate_optimizer, _post_training_diagnostics
- Полный train() с scheduler, RAR, детектором минимумов, early stopping, сбросом оптимизатора

Автор: Адаптировано из Airfoil PINN
Версия: 1.1
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
import numpy as np

from utils import check_memory_usage, set_seed
from data_handler import DataHandler
from models import PINNIncompressible, SelfAdaptiveWeights
from checkpoint import CheckpointManager
from key_listener import KeyListener
from visualizer import Visualizer
from reference import load_reference, interpolate_reference, compute_l2_errors

from solver_components.optimizers import LocalMinimaDetector, CyclicLR
from solver_components.loss_computer import LossComputer
from solver_components.trainer import Trainer
from solver_components.metrics import MetricsCollector
from solver_components.interactive import InteractiveHandler

logger = logging.getLogger(__name__)


class PINNSolver:
    """Основной класс-дирижёр, управляющий процессом обучения PINN."""

    def __init__(self, config: Dict, experiment_dir: Optional[str] = None):
        self.config = config
        self.config_name = config.get('project_name', 'unnamed')
        self.experiment_dir = experiment_dir

        seed = config.get('process', {}).get('seed', 42)
        set_seed(seed)

        self.device = self._init_device(config['process'])
        self.dtype = self._init_dtype(config['process']['dtype'])
        self.use_amp = config['process'].get('use_amp', False)

        # DataHandler — загрузка данных (с указанием device для GPU)
        self.data_handler = DataHandler(
            points_file=config['data']['points_file'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            device=self.device
        )

        # Проверка Re
        Re = self.data_handler.Re
        if Re <= 0:
            logger.warning(f"Re = {Re} <= 0, установлено в 1.0")
            Re = 1.0

        # Нормализованные константы PDE
        self.rho_norm = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.mu_norm = torch.tensor(1.0 / Re, device=self.device, dtype=self.dtype)

        # Границы области из data_handler (исправлено: не хардкод)
        self.x_min = self.data_handler.dataset.x.min().item()
        self.x_max = self.data_handler.dataset.x.max().item()
        self.y_min = self.data_handler.dataset.y.min().item()
        self.y_max = self.data_handler.dataset.y.max().item()

        # LossComputer
        self.loss_computer = LossComputer(
            config=config, data_handler=self.data_handler,
            rho_norm=self.rho_norm, mu_norm=self.mu_norm,
            x_min=self.x_min, x_max=self.x_max,
            y_min=self.y_min, y_max=self.y_max,
            device=self.device, dtype=self.dtype
        )

        # Self-Adaptive Weights
        self.sa_weights = None
        if config['loss_weights']['use_sa_weights']:
            n_eqns = 3
            if config['loss_weights'].get('sa_apply_to_bc', False):
                n_eqns += len(self.data_handler.boundary_type_names)
            self.sa_weights = SelfAdaptiveWeights(
                self.data_handler.n_points, n_eqns, init_val=1.0
            ).to(self.device, self.dtype)

        # Модель
        self.model = self._build_model().to(self.device, self.dtype)
        if config['process'].get('num_gpus', 1) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(config['process']['num_gpus'])))

        # Оптимизатор и планировщик
        self.optimizer = self._init_optimizer('adam')
        self.scheduler = self._init_scheduler()

        # AMP
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.start_epoch = 0

        # Trainer
        self.trainer = Trainer(
            model=self.model, optimizer=self.optimizer, loss_computer=self.loss_computer,
            data_handler=self.data_handler, config=config['training'],
            device=self.device, dtype=self.dtype, sa_weights=self.sa_weights,
            scaler=self.scaler, using_lbfgs=False,
            diagnostics_config=config.get('diagnostics', {})
        )

        # MetricsCollector
        self.metrics = MetricsCollector(
            model=self.model, data_handler=self.data_handler, loss_computer=self.loss_computer,
            config=config, device=self.device, dtype=self.dtype,
            logs_dir=self._get_logs_dir(), config_name=self.config_name
        )

        # Директории
        self._setup_experiment_dir(experiment_dir)

        # Визуализация
        out_cfg = config['output']
        plot_freq = out_cfg['plot_frequency'] if not out_cfg.get('console_only', True) else None
        self.visualizer = Visualizer(plot_freq, config_name=self.config_name, config=config)
        if plot_freq:
            self.visualizer.create_live_plot()

        # Клавиатурный слушатель
        self.key_listener = None
        if config.get('enable_key_listener', True):
            self.key_listener = KeyListener()
            self.key_listener.start()

        # CheckpointManager
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        # InteractiveHandler
        self.interactive = InteractiveHandler(
            trainer=self.trainer, metrics_collector=self.metrics, visualizer=self.visualizer,
            checkpoint_manager=self.checkpoint_manager, key_listener=self.key_listener,
            loss_computer=self.loss_computer, data_handler=self.data_handler,
            model=self.model, device=self.device, dtype=self.dtype,
            config=config, sa_weights=self.sa_weights, plots_dir=self.plots_dir
        )

        # Параметры обучения
        self.two_stage = config['training']['two_stage']
        self.stage1_epochs = config['training']['stage1_epochs']
        self.stage2_pde_weight_start = config['training']['stage2_pde_weight_start']
        self.stage2_pde_weight_end = config['training']['stage2_pde_weight_end']
        self.stage2_ramp_epochs = config['training']['stage2_ramp_epochs']
        self.switch_to_lbfgs_epoch = config['training'].get('switch_to_lbfgs_epoch', 0)

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

        # Детектор локальных минимумов
        self.minima_detector = None
        if config['training']['lr_schedule'] == 'detector':
            self.minima_detector = LocalMinimaDetector(
                window=20, threshold=1e-4,
                patience=config['training']['lr_detector_patience']
            )
            self.lr_boost_factor = config['training']['lr_detector_factor']
            self.lr_recovery_epochs = config['training']['lr_detector_recovery']

        self.start_time = None
        self.using_lbfgs = False

        # Загрузка чекпоинта (если указан)
        self._load_checkpoint_if_needed()

        logger.info("PINNSolver инициализирован")
        logger.info(f"  Устройство: {self.device}")
        logger.info(f"  Re = {Re:.3e}")
        logger.info(f"  Границы: x=[{self.x_min:.3f}, {self.x_max:.3f}], y=[{self.y_min:.3f}, {self.y_max:.3f}]")

    def _init_device(self, proc: Dict) -> torch.device:
        dev = proc.get('device', 'auto')
        if dev == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(dev)

    def _init_dtype(self, dtype_str: str) -> torch.dtype:
        return {'float32': torch.float32, 'float64': torch.float64, 'float16': torch.float16}[dtype_str]

    def _build_model(self) -> PINNIncompressible:
        arch = self.config['architecture']
        sample_batch = next(iter(self.data_handler.get_dataloader()))
        _, x, y, sdf, bt, _, _, _, _, _, _ = sample_batch
        x = x.to(self.device)
        y = y.to(self.device)
        sdf = sdf.to(self.device)
        bt = bt.to(self.device) if bt.shape[1] > 0 else None
        input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
        actual_input_dim = input_tensor.shape[1]
        coord_scaling = arch.get('coordinate_scaling', {'enabled': False, 'scale_factor': 1.0})
        return PINNIncompressible(
            input_dim=actual_input_dim,
            layers=arch['hidden_layers'],
            activation=arch.get('activation', 'tanh'),
            output_dim=arch['output_dim'],
            output_bias=arch['output_bias'],
            coordinate_scaling=coord_scaling
        )

    def _init_optimizer(self, optim_type='adam') -> optim.Optimizer:
        opt_params = self.config.get('optimizer_params', {'lr': 1e-3})
        lr = opt_params.get('lr', 1e-3)
        params = [{'params': self.model.parameters()}]
        if self.sa_weights is not None:
            params.append({'params': self.sa_weights.parameters(), 'lr': self.config.get('sa_lr', 0.001)})
        if optim_type == 'adam':
            return optim.Adam(params, lr=lr)
        else:
            all_params = []
            for group in params:
                all_params.extend(group['params'])
            return optim.LBFGS(all_params, lr=lr,
                               max_iter=self.config['training'].get('lbfgs_max_iter', 20),
                               history_size=self.config['training'].get('lbfgs_history_size', 100),
                               line_search_fn='strong_wolfe')

    def _init_scheduler(self):
        lr_schedule = self.config['training']['lr_schedule']
        if lr_schedule == 'cyclic':
            base_lr = self.optimizer.param_groups[0]['lr']
            max_lr = base_lr * (1 + self.config['training']['lr_cyclic_amplitude'])
            return CyclicLR(self.optimizer, base_lr, max_lr,
                            self.config['training']['lr_cyclic_period'])
        elif lr_schedule == 'detector':
            return None
        else:
            return None

    def _setup_experiment_dir(self, experiment_dir: Optional[str]) -> None:
        """Настраивает директории для чекпоинтов, логов и графиков."""
        if experiment_dir:
            self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            self.logs_dir = os.path.join(experiment_dir, "logs")
            self.plots_dir = os.path.join(experiment_dir, "plots")
        else:
            base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(base_dir, f"{self.config_name}_{timestamp}")
            self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
            self.logs_dir = os.path.join(run_dir, "logs")
            self.plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _get_checkpoint_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "checkpoints")
        base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(base_dir, f"{self.config_name}_{timestamp}", "checkpoints")

    def _get_logs_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "logs")
        base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(base_dir, f"{self.config_name}_{timestamp}", "logs")

    def _get_plots_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "plots")
        base_dir = self.config.get('checkpoint_dir', 'Checkpoints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(base_dir, f"{self.config_name}_{timestamp}", "plots")

    def _save_checkpoint(self, identifier: Union[int, str]) -> None:
        filename = f"pinn_{identifier}.pt"
        self.checkpoint_manager.save(
            filename=filename, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler if not self.using_lbfgs else None,
            loss_weights=self.loss_computer.loss_weights,
            history=self.metrics.history, sa_weights=self.sa_weights,
            epoch=identifier if isinstance(identifier, int) else None,
            config=self.config
        )

    def _load_checkpoint_if_needed(self) -> None:
        """Загружает чекпоинт, если указан путь в конфиге."""
        load_path = self.config['process'].get('load_model_path')
        if not load_path or not os.path.exists(load_path):
            return
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Модель загружена из чекпоинта")
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить оптимизатор: {e}")
            if 'history' in checkpoint:
                self.metrics.history = checkpoint['history']
            if 'epoch' in checkpoint and checkpoint['epoch'] is not None:
                self.start_epoch = checkpoint['epoch']
            logger.info(f"Чекпоинт загружен, продолжение с эпохи {self.start_epoch}")
        except Exception as e:
            logger.error(f"Ошибка загрузки чекпоинта: {e}")

    def _switch_to_lbfgs(self) -> None:
        """Переключение на L-BFGS оптимизатор."""
        logger.info("Переключение на L-BFGS оптимизатор...")
        self.using_lbfgs = True
        lbfgs_config = self.config['training']
        lr = self.config.get('optimizer_params', {}).get('lr', 1.0)
        self.optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=lbfgs_config.get('lbfgs_max_iter', 20),
            history_size=lbfgs_config.get('lbfgs_history_size', 100),
            line_search_fn='strong_wolfe',
        )
        self.trainer.optimizer = self.optimizer
        self.trainer.using_lbfgs = True
        logger.info(f"L-BFGS активирован")

    def _recreate_optimizer(self) -> None:
        """Пересоздаёт оптимизатор (после RAR или сброса)."""
        self.optimizer = self._init_optimizer('adam')
        self.trainer.optimizer = self.optimizer
        logger.info("Оптимизатор пересоздан")

    def _save_field_plots(self, suffix: str = "final") -> None:
        """Сохраняет графики полей u, v, p после обучения."""
        import matplotlib
        matplotlib.use('Agg')  # без GUI
        import matplotlib.pyplot as plt
        
        try:
            self.model.eval()
            dataset = self.data_handler.dataset
            with torch.no_grad():
                x = dataset.x.to(self.device, dtype=self.dtype)
                y = dataset.y.to(self.device, dtype=self.dtype)
                sdf = dataset.sdf.to(self.device, dtype=self.dtype)
                bt = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None
                inp = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                out = self.model(inp)
                u_pred = out[:, 0].cpu().numpy()
                v_pred = out[:, 1].cpu().numpy()
                p_pred = out[:, 2].cpu().numpy()
            
            chord_val: float = self.data_handler.chord or 1.0
            u_inf_val: float = self.data_handler.U_inf or 1.0
            rho_val: float = self.data_handler.rho or 1.0
            p_ref_val: float = self.data_handler.p_ref or 0.0
            
            x_phys = dataset.x.cpu().numpy().flatten() * chord_val
            y_phys = dataset.y.cpu().numpy().flatten() * chord_val
            u_phys = u_pred * u_inf_val
            v_phys = v_pred * u_inf_val
            p_phys = p_pred * (rho_val * u_inf_val ** 2) + p_ref_val
            
            fields = [
                ('u', u_phys, 'U, м/с'),
                ('v', v_phys, 'V, м/с'),
                ('p', p_phys, 'P, Па'),
                ('speed', np.sqrt(u_phys**2 + v_phys**2), 'Скорость, м/с'),
            ]
            
            for fname, vals, label in fields:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc = ax.scatter(x_phys, y_phys, c=vals, s=5, cmap='jet', alpha=0.7)
                cbar = plt.colorbar(sc, ax=ax, label=label)
                ax.set_xlabel('x, м')
                ax.set_ylabel('y, м')
                ax.set_title(f'{label} — {suffix}')
                ax.axis('equal')
                ax.grid(alpha=0.3)
                # Рисуем контур отвода
                domain_path = os.path.join(os.path.dirname(self.data_handler.points_file), 
                                           '..', '..', 'Geometry', 'Data', 'pipe_bend_domain.npy')
                if os.path.exists(domain_path):
                    domain = np.load(domain_path)
                    ax.plot(domain[:, 0], domain[:, 1], 'k-', linewidth=1.5, alpha=0.5)
                save_path = os.path.join(self.plots_dir, f"field_{fname}_{suffix}.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"График поля сохранён: {save_path}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить графики полей: {e}")

    def _post_training_diagnostics(self, final_epoch: int) -> None:
        """Финальные расчёты после завершения обучения."""
        logger.info("Пост-обработка...")
        try:
            self.metrics.save_log(f"Обучение завершено на эпохе {final_epoch}")
        except Exception as e:
            logger.warning(f"Ошибка в пост-обработке: {e}")

    def _get_pde_multiplier(self, epoch: int) -> float:
        if not self.two_stage:
            return 1.0
        if epoch <= self.stage1_epochs:
            return 0.0
        ramp_progress = (epoch - self.stage1_epochs) / max(1, self.stage2_ramp_epochs)
        ramp_progress = min(1.0, max(0.0, ramp_progress))
        return self.stage2_pde_weight_start + ramp_progress * (self.stage2_pde_weight_end - self.stage2_pde_weight_start)

    def train(self, epochs: Optional[int] = None):
        if epochs is None:
            epochs = self.config['training']['epochs']
        logger.info("🚀 Начало обучения...")
        self.start_time = time.time()

        try:
            for epoch in range(self.start_epoch + 1, epochs + 1):
                start_time = time.time()

                # Мониторинг памяти (каждые 10 эпох)
                if epoch % 10 == 0:
                    mem = check_memory_usage(self.device, return_free=True)
                    if mem is not None and mem < 0.5:
                        logger.warning(f"⚠️ Мало памяти ({mem:.2f} ГБ). Прерываю.")
                        self._save_checkpoint('interrupted')
                        break

                # Множитель PDE
                pde_mult = self._get_pde_multiplier(epoch)

                # Переключение на L-BFGS
                if self.switch_to_lbfgs_epoch > 0 and epoch >= self.switch_to_lbfgs_epoch and not self.using_lbfgs:
                    self._switch_to_lbfgs()

                # Шаг обучения
                if self.using_lbfgs:
                    loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max, pde_rate, grad_norm = \
                        self.trainer.train_epoch_lbfgs()
                else:
                    loss, pde_raw, bc_comp, integral_loss, wall_speed, pde_max, pde_rate, grad_norm = \
                        self.trainer.train_epoch_adam(pde_mult)

                # Сбор метрик
                avg_u_in, avg_u_out = self.metrics.get_boundary_avg_velocities(epoch)
                extra_diag = {
                    'avg_u_in': avg_u_in,
                    'avg_u_out': avg_u_out,
                    'mass_balance_error': 0.0,
                }
                self.metrics.update_history(
                    loss, pde_raw, bc_comp, integral_loss,
                    wall_speed, pde_max, pde_rate, grad_norm,
                    extra_diag
                )

                epoch_time = time.time() - start_time
                total_time = time.time() - self.start_time

                # Вывод в консоль
                self.metrics.print_epoch_table(
                    epoch, epochs, loss, pde_raw, bc_comp,
                    integral_loss, pde_max, epoch_time, total_time
                )

                # Шаг планировщика
                if self.scheduler is not None and not self.using_lbfgs:
                    if isinstance(self.scheduler, CyclicLR):
                        self.scheduler.step()
                    elif hasattr(self.scheduler, 'step'):
                        self.scheduler.step(loss)

                # RAR (Residual-based Adaptive Refinement)
                if (self.trainer.rar_enable and not self.using_lbfgs
                        and epoch % self.trainer.rar_interval == 0):
                    self.trainer.apply_rar()
                    self._recreate_optimizer()

                # Детектор локальных минимумов
                if self.minima_detector is not None and self.minima_detector.update(loss):
                    logger.info("Обнаружено плато, увеличиваю learning rate")
                    for _ in range(self.lr_recovery_epochs):
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.lr_boost_factor
                    self.minima_detector = LocalMinimaDetector(
                        window=20, threshold=1e-4,
                        patience=self.config['training']['lr_detector_patience']
                    )

                # Early stopping
                if self.early_stopping_enabled:
                    if loss < self._early_best_loss - self.early_stopping_min_delta:
                        self._early_best_loss = loss
                        self._early_best_epoch = epoch
                        self._early_bad_epochs = 0
                        if self.early_stopping_restore_best:
                            self._early_best_model_state = {
                                k: v.clone() for k, v in self.model.state_dict().items()
                            }
                    else:
                        self._early_bad_epochs += 1
                    if self._early_bad_epochs >= self.early_stopping_patience:
                        logger.info(f"Early stopping на эпохе {epoch} (лучшая: {self._early_best_epoch})")
                        if self.early_stopping_restore_best and self._early_best_model_state:
                            self.model.load_state_dict(self._early_best_model_state)
                        self._save_checkpoint('early_stop')
                        break

                # Сброс оптимизатора с шумом
                if (self.trainer.reset_interval > 0 and not self.using_lbfgs
                        and epoch % self.trainer.reset_interval == 0):
                    self._recreate_optimizer()

                # Обработка горячих клавиш
                if self.key_listener:
                    self.interactive.handle_keys(epoch)

                # Визуализация
                if (self.visualizer is not None and self.visualizer.enabled 
                        and self.visualizer.plot_frequency is not None 
                        and epoch % self.visualizer.plot_frequency == 0):
                    self.visualizer.update_plot(epoch, self.metrics.history)

                # Чекпоинт по интервалу
                chk_interval = self.config.get('checkpoint', {}).get('interval', 0)
                if chk_interval and epoch % chk_interval == 0:
                    self._save_checkpoint(epoch)

            # После всех эпох
            self._save_checkpoint('final')
            if self.visualizer:
                self.visualizer.save_plots(self.plots_dir, 'final')
            self._save_field_plots('final')
            self._post_training_diagnostics(epochs if epochs else 0)
            logger.info("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

        except KeyboardInterrupt:
            logger.warning("Обучение прервано пользователем")
            self._save_checkpoint('interrupted')
            if self.visualizer:
                self.visualizer.save_plots(self.plots_dir, 'interrupted')
            self._save_field_plots('interrupted')
            self._post_training_diagnostics(self.start_epoch)
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}", exc_info=True)
        finally:
            if self.key_listener:
                self.key_listener.stop()
            if self.visualizer:
                self.visualizer.close()
            if self.metrics:
                self.metrics.close()