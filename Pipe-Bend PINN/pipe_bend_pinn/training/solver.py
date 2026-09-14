from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

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

from pipe_bend_pinn.utils.helpers import check_memory_usage, set_seed
from pipe_bend_pinn.data.handler import DataHandler
from pipe_bend_pinn.training.models import PINNIncompressible, SelfAdaptiveWeights
from pipe_bend_pinn.training.checkpoint import CheckpointManager
from pipe_bend_pinn.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
)
from pipe_bend_pinn.training.key_listener import KeyListener
from pipe_bend_pinn.training.visualizer import Visualizer
from pipe_bend_pinn.training.reference import load_reference, interpolate_reference, compute_l2_errors

from pipe_bend_pinn.training.components.optimizers import LocalMinimaDetector, CyclicLR
from pipe_bend_pinn.training.components.loss_computer import LossComputer
from pipe_bend_pinn.training.components.trainer import Trainer
from pipe_bend_pinn.training.components.metrics import MetricsCollector
from pipe_bend_pinn.training.components.interactive import InteractiveHandler

logger = logging.getLogger(__name__)


class PINNSolver:
    """Основной класс-дирижёр, управляющий процессом обучения PINN."""

    def __init__(self, config: Dict, experiment_dir: Optional[str] = None):
        self.config = config
        self.config_name = config.get('project_name', 'unnamed')
        self.experiment_dir = experiment_dir
        # Единый timestamp для ВСЕХ директорий эксперимента
        # (иначе _get_logs_dir/_get_checkpoint_dir создают разные папки)
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

        # Нормали к стенкам для давление-Нейман члена (wall_pressure). Считаются
        # один раз по контуру области (геометрия-агностично) и хранятся на
        # data_handler, откуда их забирает LossComputer.
        self._attach_wall_normals()

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
        self.model = self._maybe_compile_model(self.model)

        # Оптимизатор и планировщик
        self.optimizer = self._init_optimizer('adam')
        self.scheduler = self._init_scheduler()

        # AMP
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.start_epoch = 0

        # Опциональная FVM-невязка на Mesh2D: строится ДО Trainer, т.к. FVM
        # участвует в loss (loss_weights.fvm), а не только в телеметрии.
        self.fvm_residuals = None
        self.fvm_mesh = None
        self.fvm_metrics_enabled = False
        self.fvm_metrics_interval = 1
        self._fvm_node_sdf = None
        self._setup_fvm_diagnostics()

        # Trainer
        lw_cfg = config.get('loss_weights', {})
        self.trainer = Trainer(
            model=self.model, optimizer=self.optimizer, loss_computer=self.loss_computer,
            data_handler=self.data_handler, config=config['training'],
            device=self.device, dtype=self.dtype, sa_weights=self.sa_weights,
            scaler=self.scaler, using_lbfgs=False,
            diagnostics_config=config.get('diagnostics', {}),
            fvm_residuals=self.fvm_residuals if self.fvm_metrics_enabled else None,
            fvm_coords=(self.fvm_mesh.coords if self.fvm_mesh is not None else None),
            fvm_sdf=self._fvm_node_sdf,
            fvm_weight=float(lw_cfg.get('fvm', {}).get('weight', 10.0)),
            fvm_reduction=lw_cfg.get('fvm', {}).get('reduction', 'divergence_l2_volume'),
            gauge_weight=float(lw_cfg.get('pressure_gauge', {}).get('weight', 0.0)),
            balance_weight=float(lw_cfg.get('fvm', {}).get('balance_weight', 0.0)),
            pipe_width=float(getattr(self.data_handler, 'pipe_width', 1.0) or 1.0),
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
        self._trainable_state_before_lbfgs = None

        # Загрузка чекпоинта (если указан)
        self._load_checkpoint_if_needed()

        logger.info("PINNSolver инициализирован")
        logger.info(f"  Устройство: {self.device}")
        logger.info(f"  Re = {Re:.3e}")
        logger.info(f"  Границы: x=[{self.x_min:.3f}, {self.x_max:.3f}], y=[{self.y_min:.3f}, {self.y_max:.3f}]")

    def _attach_wall_normals(self) -> None:
        """Вычисляет внутренние нормали к стенкам для всех точек датасета.

        Используется слагаемым ``wall_pressure`` (давление-Нейман на стенке):
        данные берутся из файла точек (``points_file``), домен — из контура
        области. Нормаль направлена внутрь жидкости. При любой ошибке просто
        оставляет ``wall_normals = None`` (слагаемое выключится).
        """
        try:
            import numpy as _np

            from pipe_bend_pinn.sampling.core import load_domain
            from pipe_bend_pinn.geometry.sdf import compute_wall_normals

            arr = _np.load(self.config['data']['points_file'])
            names = arr.dtype.names or ()
            if 'boundary_type' not in names:
                return
            wall_mask = arr['boundary_type'] == 'wall'
            if not bool(wall_mask.any()):
                return
            if 'x_norm' in names and 'y_norm' in names:
                pts = _np.column_stack((arr['x_norm'], arr['y_norm']))
            elif 'x' in names and 'y' in names:
                pw = float(getattr(self.data_handler, 'pipe_width', 0.5) or 0.5)
                pts = _np.column_stack((arr['x'], arr['y'])) / pw
            else:
                return
            domain = _np.asarray(load_domain('pipe_bend_domain.npy'), dtype=_np.float64)
            if domain.ndim == 1 and domain.dtype.names is not None:
                domain = _np.column_stack((domain['x'], domain['y']))
            pw = float(getattr(self.data_handler, 'pipe_width', 0.5) or 0.5)
            domain = domain / pw
            normals = compute_wall_normals(pts, domain, wall_mask)
            self.data_handler.wall_normals = normals
            logger.info(
                "Нормали к стенкам вычислены: %d wall-точек (max|n|=%.3f)",
                int(wall_mask.sum()), float(_np.abs(normals).max()),
            )
        except Exception as exc:
            logger.warning(f"Нормали к стенкам не вычислены (wall_pressure выкл): {exc}")
            self.data_handler.wall_normals = None

    def _init_device(self, proc: Dict) -> torch.device:
        dev = proc.get('device', 'auto')
        if dev == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(dev)

    def _init_dtype(self, dtype_str: str) -> torch.dtype:
        return {'float32': torch.float32, 'float64': torch.float64, 'float16': torch.float16}[dtype_str]

    def _setup_fvm_diagnostics(self) -> None:
        """Строит FVM-сетку и FVMResiduals, если включён ``loss_weights.fvm``.

        FVM-невязка неразрывности вычисляется на всей сетке (контрольные
        объёмы Mesh2D) и пишется в историю метрик как ``fvm_mse`` / ``fvm_rms``.
        """
        fvm_cfg = self.config.get('loss_weights', {}).get('fvm', {})
        if not fvm_cfg.get('enabled', False):
            return
        try:
            from pipe_bend_pinn.sampling.mesh import build_pipe_bend_mesh
            from pipe_bend_pinn.physics.fvm import FVMResiduals
            # Сетка строится по точкам коллокации (размерные x/y), а не по полю.
            arr = self._load_collocation_array()
            if arr is None:
                logger.warning("FVM диагностика выключена: файл точек коллокации не найден")
                return
            self.fvm_mesh = build_pipe_bend_mesh(arr, verbose=False)
            self.fvm_residuals = FVMResiduals(self.fvm_mesh).to(self.device, self.dtype)
            self.fvm_metrics_enabled = True
            self.fvm_metrics_interval = max(1, int(fvm_cfg.get('batch_interval', 1)))
            # SDF узлов сетки не меняется в ходе обучения — вычисляем один раз,
            # чтобы не тратить CPU на пересчёт в каждой эпохе.
            self._fvm_node_sdf = self._sdf_for_nodes(np.asarray(self.fvm_mesh.coords, dtype=np.float64))
            logger.info(
                "FVM диагностика активна: N=%d узлов, консервативность=%s",
                self.fvm_mesh.coords.shape[0],
                self.fvm_mesh.is_conservative(),
            )
        except Exception as exc:
            logger.warning(f"FVM диагностика не инициализирована: {exc}")

    def _load_collocation_array(self) -> Optional[np.ndarray]:
        """Возвращает structured-массив точек коллокации (x/y), если он доступен."""
        candidates = []
        colloc = os.path.join(SAMPLING_COLLOCATION_DIR, "collocation_points.npy")
        if os.path.exists(colloc):
            candidates.append(colloc)
        points_path = self.config.get('data', {}).get('points_file')
        if points_path and os.path.exists(points_path):
            candidates.append(points_path)
        for path in candidates:
            try:
                arr = np.load(path, allow_pickle=True)
                names = arr.dtype.names or ()
                if ("x" in names and "y" in names) or (
                    "x_norm" in names and "y_norm" in names
                ):
                    return arr
            except Exception:
                continue
        return None

    def _compute_fvm_metrics(self) -> Optional[Tuple[float, float]]:
        """Считает (fvm_mse, fvm_rms) по предсказанию модели на узлах сетки."""
        if not self.fvm_metrics_enabled or self.fvm_mesh is None or self.fvm_residuals is None:
            return None
        try:
            coords = np.asarray(self.fvm_mesh.coords, dtype=np.float64)
            # SDF узлов сетки закеширован при инициализации (не меняется).
            sdf = self._fvm_node_sdf
            if sdf is None:
                sdf = self._sdf_for_nodes(coords)
            if sdf is None:
                return None
            # Модель обучена на НОРМИРОВАННЫХ входах (x/W, y/W, sdf/W).
            pipe_w = float(getattr(self.data_handler, 'pipe_width', 1.0) or 1.0)
            coords = coords / pipe_w
            sdf = np.asarray(sdf, dtype=np.float64) / pipe_w
            inlet_vel = self._fvm_inlet_velocity()
            from pipe_bend_pinn.physics.fvm import evaluate_fvm_metrics

            mse, rms = evaluate_fvm_metrics(
                self.model,
                self.fvm_residuals,
                coords[:, 0], coords[:, 1], sdf,
                self.device, self.dtype,
                reduction=self.config.get('loss_weights', {}).get('fvm', {}).get('reduction', 'divergence_l2_volume'),
                inlet_velocity=inlet_vel,
            )
            return mse, rms
        except Exception as exc:
            logger.warning(f"Ошибка вычисления FVM-метрик: {exc}")
            return None

    def _sdf_for_nodes(self, coords: np.ndarray) -> Optional[np.ndarray]:
        """Точный сегментный SDF до стенок для произвольных координат."""
        from pipe_bend_pinn.geometry.sdf import PolylineDistanceField
        # Берём стенки из точек коллокации (упорядочены по периметру контура).
        try:
            arr = self._load_collocation_array()
            if arr is None or arr.dtype.names is None or 'boundary_type' not in arr.dtype.names:
                return None
            names = arr.dtype.names
            if 'x' in names and 'y' in names:
                wall_mask = arr['boundary_type'] == 'wall'
                wall_pts = np.column_stack((arr['x'][wall_mask], arr['y'][wall_mask]))
            elif 'x_norm' in names and 'y_norm' in names:
                wall_mask = arr['boundary_type'] == 'wall'
                wall_pts = np.column_stack((arr['x_norm'][wall_mask], arr['y_norm'][wall_mask]))
            else:
                return None
        except Exception:
            return None
        if len(wall_pts) < 2:
            return None
        pts = np.asarray(wall_pts, dtype=np.float64)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0:1]])
        delta = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        threshold = 2.0 * max(float(np.median(delta)), 1e-12)
        breaks = np.where(delta > threshold)[0]
        chains = []
        start = 0
        for boundary in np.append(breaks, len(pts) - 1):
            chain = pts[start:boundary + 1]
            if len(chain) >= 2:
                chains.append(chain)
            start = boundary + 1
        if not chains:
            return None
        distance = None
        for chain in chains:
            field = PolylineDistanceField(chain, closed=False, segment_chunk_size=256)
            d = field.evaluate_numpy(coords, point_chunk_size=8192)
            distance = d if distance is None else np.minimum(distance, d)
        return distance

    def _fvm_inlet_velocity(self) -> Optional[torch.Tensor]:
        """Средняя скорость на входе из данных коллокации (для prescribed BC)."""
        try:
            dataset = self.data_handler.dataset
            names = self.data_handler.boundary_type_names
            if dataset.boundary_types is None or 'inlet' not in names:
                return None
            idx = names.index('inlet')
            mask = dataset.boundary_types[:, idx] > 0.5
            if not mask.any():
                return None
            u_in = dataset.u_target[mask].mean().item()
            v_in = dataset.v_target[mask].mean().item()
            return torch.tensor([u_in, v_in], device=self.device, dtype=self.dtype)
        except Exception:
            return None

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
            coordinate_scaling=coord_scaling,
            fourier_features=arch.get('fourier_features', None),
            residual=arch.get('residual', False),
        )

    def _maybe_compile_model(self, model: nn.Module) -> nn.Module:
        """Опционально компилирует модель через ``torch.compile``.

        Управляется флагом ``process.torch_compile`` в конфигурации (по
        умолчанию выключено). При любой ошибке компиляции тихо откатываемся
        на исходную модель, чтобы обучение не прерывалось.
        """
        compile_flag = self.config.get('process', {}).get('torch_compile', False)
        if not compile_flag:
            return model
        if isinstance(model, nn.DataParallel):
            return model
        try:
            compiled = torch.compile(model)
            logger.info("Модель скомпилирована через torch.compile")
            return compiled
        except Exception as exc:
            logger.warning(f"torch.compile недоступен, использую исходную модель: {exc}")
            return model

    def _capture_trainable_state(self) -> Dict:
        """Снапшот весов модели и SA-весов для отката L-BFGS при NaN."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state = {
            'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
            'sa_weights_state_dict': None,
        }
        if self.sa_weights is not None:
            state['sa_weights_state_dict'] = {
                k: v.clone() for k, v in self.sa_weights.state_dict().items()
            }
        return state

    def _restore_trainable_state(self, state: Optional[Dict]) -> None:
        """Восстанавливает веса модели и SA-весов из снапшота."""
        if not state:
            return
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if state.get('model_state_dict') is not None:
            model.load_state_dict(state['model_state_dict'])
        if self.sa_weights is not None and state.get('sa_weights_state_dict') is not None:
            try:
                self.sa_weights.load_state_dict(state['sa_weights_state_dict'])
            except Exception as exc:
                logger.warning(f"Не удалось восстановить SA-веса: {exc}")

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
        elif lr_schedule == 'cosine':
            epochs = int(self.config['training'].get('epochs', 20000))
            eta_min = float(self.config['training'].get('lr_cosine_eta_min', 1e-5))
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=eta_min
            )
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
            base_dir = self.config.get('checkpoint_dir', EXPERIMENTS_DIR)
            run_dir = os.path.join(base_dir, f"{self.config_name}_{self.run_timestamp}")
            self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
            self.logs_dir = os.path.join(run_dir, "logs")
            self.plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _get_checkpoint_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "checkpoints")
        base_dir = self.config.get('checkpoint_dir', EXPERIMENTS_DIR)
        return os.path.join(base_dir, f"{self.config_name}_{self.run_timestamp}", "checkpoints")

    def _get_logs_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "logs")
        base_dir = self.config.get('checkpoint_dir', EXPERIMENTS_DIR)
        return os.path.join(base_dir, f"{self.config_name}_{self.run_timestamp}", "logs")

    def _get_plots_dir(self) -> str:
        if self.experiment_dir:
            return os.path.join(self.experiment_dir, "plots")
        base_dir = self.config.get('checkpoint_dir', EXPERIMENTS_DIR)
        return os.path.join(base_dir, f"{self.config_name}_{self.run_timestamp}", "plots")

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
            if ('sa_weights_state_dict' in checkpoint and self.sa_weights is not None):
                try:
                    self.sa_weights.load_state_dict(checkpoint['sa_weights_state_dict'])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить SA-веса (размер мог измениться после RAR): {e}")
            # Воспроизводимый resume: восстанавливаем состояние ГСЧ из чекпоинта
            # (Python, NumPy, torch CPU/CUDA), чтобы стохастическая траектория
            # продолжилась, а не перезапустилась.
            restore_rng_state(checkpoint.get('rng_state'))
            logger.info(f"Чекпоинт загружен, продолжение с эпохи {self.start_epoch}")
            # Опционально: переинициализировать случайный Fourier-базис B под sigma
            # из конфига (иначе load_state_dict перезапишет его B из чекпоинта).
            arch_ff = self.config.get('architecture', {}).get('fourier_features', {})
            if (isinstance(arch_ff, dict) and arch_ff.get('enabled', False)
                    and arch_ff.get('reseed_on_resume', False)):
                fourier = getattr(self.model, 'fourier', None)
                if fourier is not None and hasattr(fourier, 'B'):
                    sigma = float(arch_ff.get('sigma', 3.0))
                    b_shape = fourier.B.shape
                    fourier.B.data = (
                        torch.randn(b_shape[0], b_shape[1], device=fourier.B.device,
                                    dtype=fourier.B.dtype) * sigma
                    )
                    logger.info(
                        f"Fourier-базис B переинициализирован после resume "
                        f"(shape={tuple(b_shape)}, sigma={sigma})"
                    )
        except Exception as e:
            logger.error(f"Ошибка загрузки чекпоинта: {e}")

    def _switch_to_lbfgs(self) -> None:
        """Переключение на L-BFGS оптимизатор.

        SA-веса замораживаются: их log-параметры разрослись за время Adam-фазы,
        и line search (strong_wolfe) на них расходится (NaN). L-BFGS оптимизирует
        только параметры сети; SA-веса фиксируются на достигнутом значении.
        """
        logger.info("Переключение на L-BFGS оптимизатор...")
        # Снапшот весов перед переключением: если L-BFGS даст NaN, откатываемся
        # на это состояние и продолжаем Adam-фазой.
        self._trainable_state_before_lbfgs = self._capture_trainable_state()
        self.using_lbfgs = True
        lbfgs_config = self.config['training']
        lr = self.config.get('optimizer_params', {}).get('lbfgs_lr', 1.0)
        if self.sa_weights is not None:
            for p in self.sa_weights.parameters():
                p.requires_grad_(False)
        self.optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=lbfgs_config.get('lbfgs_max_iter', 20),
            history_size=lbfgs_config.get('lbfgs_history_size', 100),
            line_search_fn='strong_wolfe',
        )
        self.trainer.optimizer = self.optimizer
        self.trainer.using_lbfgs = True
        logger.info(f"L-BFGS активирован (lr={lr}, SA-веса заморожены)")

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
            p_phys = p_pred * (0.5 * rho_val * u_inf_val ** 2) + p_ref_val
            
            fields = [
                ('u', u_phys, 'U, м/с'),
                ('v', v_phys, 'V, м/с'),
                ('p', p_phys, 'P, Па'),
                ('speed', np.sqrt(u_phys**2 + v_phys**2), 'Скорость, м/с'),
            ]
            
            n_points = len(x_phys)
            marker_size = max(1, min(20, 50000 / max(n_points, 1)))
            
            for fname, vals, label in fields:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc = ax.scatter(x_phys, y_phys, c=vals, s=marker_size, cmap='jet', alpha=0.7)
                cbar = plt.colorbar(sc, ax=ax, label=label)
                ax.set_xlabel('x, м')
                ax.set_ylabel('y, м')
                ax.set_title(f'{label} — {suffix}')
                ax.axis('equal')
                ax.grid(alpha=0.3)
                # Рисуем контур отвода
                domain_path = os.path.join(GEOMETRY_DOMAINS_DIR, 'pipe_bend_domain.npy')
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
        logger.info("=== Начало обучения... ===")
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

                # Откат с L-BFGS на Adam, если L-BFGS дал NaN
                if self.trainer.optimizer is None:
                    logger.warning("L-BFGS дал NaN — переключаюсь обратно на Adam")
                    # Восстанавливаем веса на момент переключения на L-BFGS,
                    # чтобы NaN не остался в модели.
                    self._restore_trainable_state(self._trainable_state_before_lbfgs)
                    self.using_lbfgs = False
                    self.trainer.using_lbfgs = False
                    if self.sa_weights is not None:
                        for p in self.sa_weights.parameters():
                            p.requires_grad_(True)
                    self._recreate_optimizer()

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
                # Запись строки в CSV (иначе metrics.csv остаётся пустым)
                try:
                    self.metrics.add_row_to_csv(epoch, loss, pde_raw, bc_comp, wall_speed, pde_max)
                except Exception as e:
                    logger.warning(f"Не удалось записать CSV: {e}")

                # FVM-диагностика (порт из Airfoil PINN): fvm_mse / fvm_rms
                if self.fvm_metrics_enabled and epoch % self.fvm_metrics_interval == 0:
                    fvm_metrics = self._compute_fvm_metrics()
                    if fvm_metrics is not None:
                        fvm_mse, fvm_rms = fvm_metrics
                        self.metrics.history.setdefault('fvm_mse', []).append(fvm_mse)
                        self.metrics.history.setdefault('fvm_rms', []).append(fvm_rms)
                        logger.info(f"  FVM: mse={fvm_mse:.3e}, rms={fvm_rms:.3e}")
                    else:
                        self.metrics.history.setdefault('fvm_mse', []).append(float('nan'))
                        self.metrics.history.setdefault('fvm_rms', []).append(float('nan'))

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
                    else:
                        # CosineAnnealingLR и др. не принимают loss
                        self.scheduler.step()

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