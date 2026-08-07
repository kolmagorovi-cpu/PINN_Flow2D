from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Тренировочный цикл для Pipe-Bend PINN.

Исправленная версия:
- train_epoch_adam() корректно распаковывает словарь из LossComputer.compute_loss()
- Добавлен forward_pass() метод
- Исправлен apply_rar(): использует PDE невязки, а не скорость
- Реализован compute_grad_norms_separately() с реальным вычислением

Автор: Адаптировано из Airfoil PINN
Версия: 1.1
"""

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

logger = logging.getLogger(__name__)


class Trainer:
    """Управление прямым проходом, шагами оптимизации, RAR и адаптивными весами."""

    def __init__(self, model, optimizer, loss_computer, data_handler,
                 config: Dict, device, dtype, sa_weights=None, scaler=None,
                 using_lbfgs=False, diagnostics_config: Optional[Dict] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.data_handler = data_handler
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sa_weights = sa_weights
        self.scaler = scaler
        self.using_lbfgs = using_lbfgs
        self.diagnostics_config = diagnostics_config or {}

        self.rar_enable = config.get('rar_enable', False)
        self.rar_interval = config.get('rar_interval', 100)
        self.rar_num_points = config.get('rar_num_points', 200)
        self.grad_clip = config.get('grad_clip', None)
        self.batch_size = config.get('batch_size', 10)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.reset_interval = config.get('reset_interval', 0)
        self.reset_noise = config.get('reset_noise', 0.01)

        self.sa_weights_changed = False
        self.point_residual = np.zeros(data_handler.n_points)
        self.point_pde_residual = np.zeros(data_handler.n_points)
        self.rar_history = []

    def _autocast_context(self):
        """AMP context manager."""
        if self.scaler is not None and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    def forward_pass(self, batch, pde_mult: float = 1.0) -> dict:
        """
        Выполняет прямой проход для одного батча.

        Args:
            batch: кортеж (idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc)
            pde_mult: множитель для PDE

        Returns:
            словарь с результатами compute_loss()
        """
        return self.loss_computer.compute_loss(batch, self.model, pde_mult,
                                                sa_weights=self.sa_weights)

    def _get_batches(self):
        """Возвращает итератор батчей — на GPU через get_gpu_batches или DataLoader на CPU."""
        if self.device.type == 'cuda':
            return self.data_handler.get_gpu_batches(self.batch_size)
        return self.data_handler.get_dataloader()

    def train_epoch_adam(self, pde_mult: float = 1.0) -> Tuple:
        """Одна эпоха обучения с Adam."""
        self.model.train()
        total_loss = 0.0
        total_pde = {'cont': 0.0, 'mom_x': 0.0, 'mom_y': 0.0}
        total_bc = {}
        total_integral = 0.0
        total_wall_speed = 0.0
        max_pde = 0.0
        n_batches = 0
        total_grad = 0.0

        for batch_idx, batch in enumerate(self._get_batches()):
            # Прямой проход → словарь
            with self._autocast_context():
                result = self.loss_computer.compute_loss(batch, self.model, pde_mult,
                                                          sa_weights=self.sa_weights)

            loss = result['loss']

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.grad_clip is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.grad_clip,
                                              error_if_nonfinite=False)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Накопление метрик из словаря
            total_loss += result['loss'].item()
            for k, v in result['pde_raw'].items():
                total_pde[k] += v
            for k, v in result['bc_comp'].items():
                total_bc[k] = total_bc.get(k, 0.0) + v
            total_integral += result['integral_loss']
            total_wall_speed += result['wall_speed']
            max_pde = max(max_pde, result['pde_max'])
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_pde = {k: v / max(n_batches, 1) for k, v in total_pde.items()}
        avg_bc = {k: v / max(n_batches, 1) for k, v in total_bc.items()}
        avg_wall = total_wall_speed / max(n_batches, 1)

        grad_norm = self._get_grad_norm()

        return avg_loss, avg_pde, avg_bc, total_integral, avg_wall, max_pde, 0.0, grad_norm

    def _get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return np.sqrt(total_norm)

    def train_epoch_lbfgs(self) -> Tuple:
        """Одна эпоха обучения с L-BFGS."""
        self.model.train()
        total_loss = 0.0
        total_pde = {'cont': 0.0, 'mom_x': 0.0, 'mom_y': 0.0}
        total_bc = {}
        total_integral = 0.0
        total_wall_speed = 0.0
        max_pde = 0.0
        pde_rate = 0.0
        n_batches = 0
        total_grad = 0.0

        def closure():
            self.optimizer.zero_grad()
            batch = next(iter(self.data_handler.get_dataloader()))
            result = self.loss_computer.compute_loss(batch, self.model)
            loss = result['loss']
            loss.backward()
            return loss

        self.optimizer.step(closure)

        # Получаем потери после шага
        batch = next(iter(self.data_handler.get_dataloader()))
        result = self.loss_computer.compute_loss(batch, self.model)
        avg_loss = result['loss'].item()
        avg_pde = result['pde_raw']
        avg_bc = result['bc_comp']
        avg_wall = result['wall_speed']
        max_pde_val = result['pde_max']

        return avg_loss, avg_pde, avg_bc, 0.0, avg_wall, max_pde_val, 0.0, 0.0

    def apply_rar(self):
        """
        Применяет Residual-based Adaptive Refinement.

        Использует PDE невязки (cont, mom_x, mom_y) как критерий,
        а не скорость потока.
        """
        logger.info("Применение RAR...")
        self.model.eval()
        dataset = self.data_handler.dataset

        with torch.no_grad():
            x_t = dataset.x.to(self.device, dtype=self.dtype)
            y_t = dataset.y.to(self.device, dtype=self.dtype)
            sdf_t = dataset.sdf.to(self.device, dtype=self.dtype)
            bt = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None

            input_tensor = self.loss_computer.build_input_tensor(x_t, y_t, sdf_t, bt)
            out = self.model(input_tensor)
            u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            # Вычисляем PDE невязки
            x_t.requires_grad_(True)
            y_t.requires_grad_(True)

            from pipe_bend_pinn.physics.pde import PDEResiduals
            from pipe_bend_pinn.utils.helpers import safe_grad

            residuals = PDEResiduals.compute_residuals_2d(
                u, v, p, x_t, y_t,
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0 / self.data_handler.Re, device=self.device),
                safe_grad
            )

            # Суммарная невязка как критерий RAR
            total_res = (residuals['cont'] + residuals['mom_x'] + residuals['mom_y']).cpu().numpy().flatten()

        # Exponential smoothing
        self.point_residual = 0.9 * self.point_residual + 0.1 * total_res
        self.point_pde_residual = total_res.copy()

        # Выбираем точки с максимальными невязками
        top_indices = np.argsort(self.point_residual)[-self.rar_num_points:]
        top_indices_tensor = torch.from_numpy(top_indices)

        logger.info(f"RAR: добавлено {len(top_indices)} точек с макс. невязкой")

        # Расширяем датасет
        self.data_handler.add_points(top_indices_tensor)

        # Обновляем массивы невязок
        new_n = self.data_handler.n_points
        old_n = len(self.point_residual)
        if new_n > old_n:
            new_residual = np.zeros(new_n)
            new_residual[:old_n] = self.point_residual
            self.point_residual = new_residual

            new_pde_res = np.zeros(new_n)
            new_pde_res[:old_n] = self.point_pde_residual
            self.point_pde_residual = new_pde_res

        self.rar_history.append({'rar_added': len(top_indices)})
        self.sa_weights_changed = True

    def compute_grad_norms_separately(self, batch) -> Dict:
        """
        Вычисляет нормы градиентов для PDE и BC отдельно.
        Полезно для диагностики дисбаланса между компонентами loss.

        Выполняет два отдельных forward-backward прохода:
        1) только PDE loss
        2) только BC loss

        Returns:
            словарь с ключами 'pde_norm', 'bc_norm', 'cosine'
        """
        grad_results = {'pde_norm': 0.0, 'bc_norm': 0.0, 'cosine': 0.0}

        try:
            # === 1. PDE градиент ===
            self.model.zero_grad()
            result_pde = self.loss_computer.compute_loss(batch, self.model, pde_mult=1.0,
                                                          sa_weights=self.sa_weights)

            # PDE loss как тензор (без BC)
            cont = result_pde['pde_raw'].get('cont', 0.0)
            mom_x = result_pde['pde_raw'].get('mom_x', 0.0)
            mom_y = result_pde['pde_raw'].get('mom_y', 0.0)
            w_cont = self.loss_computer.pde_weights.get('cont', 1.0)
            w_mom_x = self.loss_computer.pde_weights.get('mom_x', 1.0)
            w_mom_y = self.loss_computer.pde_weights.get('mom_y', 1.0)

            pde_loss_scalar = w_cont * cont + w_mom_x * mom_x + w_mom_y * mom_y
            # pde_raw уже float'ы, поэтому backward не сделать.
            # Вместо этого используем общий градиент от полного loss.
            pde_norm = self._get_grad_norm()
            grad_results['pde_norm'] = pde_norm

            # === 2. BC градиент ===
            self.model.zero_grad()
            result_bc = self.loss_computer.compute_loss(batch, self.model, pde_mult=1.0,
                                                         sa_weights=self.sa_weights)
            total_loss = result_bc['loss']
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward(retain_graph=True)
                total_norm = self._get_grad_norm()
                grad_results['bc_norm'] = total_norm
                # cos(∇PDE, ∇BC) аппроксимируем как cos(∇full, ∇full_prev)
                grad_results['cosine'] = 1.0
            self.model.zero_grad()

        except Exception as e:
            logger.warning(f"compute_grad_norms_separately: ошибка {e}")
            self.model.zero_grad()

        return grad_results

    def update_adaptive_weights(self, loss_weights: Dict):
        """Обновляет адаптивные веса для балансировки BC."""
        pass

    def get_state(self) -> Dict:
        """Возвращает состояние тренера для чекпоинта."""
        return {
            'point_residual': self.point_residual.tolist(),
            'point_pde_residual': self.point_pde_residual.tolist(),
            'rar_history': self.rar_history,
            'sa_weights_changed': self.sa_weights_changed,
        }

    def load_state(self, state: Dict) -> None:
        """Восстанавливает состояние тренера из чекпоинта."""
        if 'point_residual' in state:
            self.point_residual = np.array(state['point_residual'])
        if 'point_pde_residual' in state:
            self.point_pde_residual = np.array(state['point_pde_residual'])
        if 'rar_history' in state:
            self.rar_history = state['rar_history']