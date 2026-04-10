# Исправленный файл: trainer.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль trainer для проекта PINN Learning.
Содержит класс Trainer, отвечающий за прямое распространение,
шаги оптимизации (Adam и L-BFGS), RAR и адаптивную балансировку весов,
а также сбор диагностических данных.

Исправленная версия:
- В RAR новые точки инициализируются нулевой невязкой, чтобы избежать дублирования высоких значений.
- Метод get_profile_at_x помечен как устаревший (для диагностики использовать data_handler.interpolate_profile с предсказаниями).
- Незначительные улучшения.

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная)
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np

from utils import safe_grad

logger = logging.getLogger(__name__)


class Trainer:
    """
    Класс, реализующий логику обучения PINN: прямой проход, шаги оптимизации,
    адаптивную балансировку весов, RAR и сбор диагностических метрик.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_computer,
        data_handler,
        config: Dict,
        device: torch.device,
        dtype: torch.dtype,
        sa_weights: Optional[nn.Module] = None,
        scaler: Optional[GradScaler] = None,
        using_lbfgs: bool = False,
        diagnostics_config: Optional[Dict] = None,
    ):
        """
        Args:
            model: нейросетевая модель
            optimizer: оптимизатор (Adam или L-BFGS)
            loss_computer: объект для вычисления компонентов функции потерь
            data_handler: обработчик данных (предоставляет DataLoader и метаданные)
            config: словарь конфигурации (раздел 'training')
            device: устройство для вычислений
            dtype: тип данных (torch.float32, torch.float64 и т.д.)
            sa_weights: обучаемые веса для SA-PINN (опционально)
            scaler: скалер для AMP (опционально)
            using_lbfgs: флаг, указывающий, что используется L-BFGS
            diagnostics_config: словарь с настройками диагностики (интервалы, флаги и т.п.)
        """
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

        # Параметры обучения
        tr = config
        self.grad_clip = tr.get('grad_clip')
        self.accum_steps = tr.get('gradient_accumulation_steps', 1)
        self.reset_interval = tr.get('reset_interval', 0)
        self.reset_noise = tr.get('reset_noise', 0.01)
        self.rar_enable = tr.get('rar_enable', False)
        self.rar_interval = tr.get('rar_interval', 100)
        self.rar_num_points = tr.get('rar_num_points', 200)
        self.use_amp = tr.get('use_amp', False)

        # Адаптивная балансировка весов BC (только для soft граничных условий)
        lw = self.loss_computer.loss_weights
        self.adaptive_weights = lw.get('adaptive_bc', False) and self.loss_computer.bc_type == 'soft'
        self.adapt_frequency = 50
        self.adapt_window = 10
        self.loss_ma = {}
        for comp in self.loss_computer.pde_components:
            self.loss_ma[comp] = deque(maxlen=self.adapt_window)
        if self.loss_computer.bc_type == 'soft':
            for name in self.data_handler.boundary_type_names:
                self.loss_ma[name] = deque(maxlen=self.adapt_window)

        # Хранилище невязок для RAR и диагностики (на CPU)
        self.point_residual = torch.zeros(
            self.data_handler.n_points,
            device='cpu',
            dtype=self.dtype
        )
        # Для хранения максимальных невязок по точкам (для карты)
        self.max_residual = torch.zeros_like(self.point_residual)

        # История RAR (для статистики)
        self.rar_history = []  # список словарей с типами границ добавленных точек

        # Флаг изменения SA-весов (для сигнала solver'у о необходимости пересоздания оптимизатора)
        self.sa_weights_changed = False

        # Параметры диагностики
        self.diag_config = diagnostics_config or {}
        self.diag_interval = self.diag_config.get('interval', 50)
        self.profile_sections = self.diag_config.get('profile_sections', [0.2, 0.5, 0.8])
        self.spectrum_section_x = self.diag_config.get('spectrum_section_x', 0.5)
        self.compute_grad_cosine = self.diag_config.get('compute_grad_cosine', False)

    # ------------------------------------------------------------------
    # Прямой проход (вычисление всех компонент потерь)
    # ------------------------------------------------------------------
    def forward_pass(self, batch: Tuple, pde_mult: float = 1.0) -> Dict[str, Any]:
        """
        Выполняет прямой проход для одного батча, вычисляет все компоненты потерь
        и возвращает словарь с результатами, включая отдельные тензоры потерь.
        """
        idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch

        idx = idx.to(self.device)
        x = x.to(self.device, dtype=self.dtype).requires_grad_(True)
        y = y.to(self.device, dtype=self.dtype).requires_grad_(True)
        sdf = sdf.to(self.device, dtype=self.dtype).requires_grad_(True)  # Нужны градиенты для нормалей

        if bt.shape[1] > 0:
            bt = bt.to(self.device, dtype=self.dtype)
        else:
            bt = None

        mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
        mask_v = mask_v.to(self.device, dtype=torch.bool).view(-1, 1)
        mask_p = mask_p.to(self.device, dtype=torch.bool).view(-1, 1)
        u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)
        v_bc = v_bc.to(self.device, dtype=self.dtype).view(-1, 1)
        p_bc = p_bc.to(self.device, dtype=self.dtype).view(-1, 1)

        # Формируем входной тензор для модели
        input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)

        # Прямой проход модели
        out = self.model(input_tensor)
        
        # Обработка выхода модели в зависимости от типа PDE
        if self.loss_computer.use_potential:
            # Для потенциала: out[:, 0:1] - это φ
            # Скорости вычисляются через градиенты потенциала
            # ВАЖНО: retain_graph=True нужен для многократных вычислений градиентов
            phi = out[:, 0:1]
            u = safe_grad(phi, x, retain_graph=True)  # u = ∂φ/∂x, сохраняем граф
            v = safe_grad(phi, y, retain_graph=True)  # v = ∂φ/∂y, сохраняем граф
            # Для совместимости создаём фиктивное давление
            p = torch.zeros_like(phi)
            u_final, v_final, p_final = u, v, p  # Для использования ниже
        else:
            u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            # Применение жёстких граничных условий (если необходимо)
            if self.loss_computer.bc_type == 'hard':
                u_final, v_final, p_final = self.loss_computer.apply_hard_constraints(
                    u_raw, v_raw, p_raw, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc
                )
            else:
                u_final, v_final, p_final = u_raw, v_raw, p_raw

        # Вычисление невязок PDE
        # Для потенциала: передаём phi как первый аргумент (u_final - это phi!)
        # Для Навье-Стокса: u_final - это скорость
        if self.loss_computer.use_potential:
            res = self.loss_computer.compute_pde_residuals(phi, v_final, p_final, x, y)
        else:
            res = self.loss_computer.compute_pde_residuals(u_final, v_final, p_final, x, y)
        
        if self.loss_computer.use_potential:
            # Для потенциала только одна компонента - laplace
            laplace = res['laplace']
            
            # Сохраняем невязки для RAR (открепляем от графа)
            laplace_res = laplace.detach().cpu()
            total_res = laplace_res.flatten()

            # Обновляем скользящее среднее невязки для RAR
            alpha = 0.9
            self.point_residual[idx.cpu()] = alpha * self.point_residual[idx.cpu()] + (1 - alpha) * total_res
            # Обновляем максимальную невязку (для карты)
            self.max_residual[idx.cpu()] = torch.max(self.max_residual[idx.cpu()], total_res)

            # Средние и максимальные значения для отчёта
            pde_raw = {
                'laplace': laplace.mean().item(),
            }
            pde_max = {
                'laplace': laplace.max().item(),
            }

            # Взвешивание PDE с учётом SA-весов (если используются)
            if self.sa_weights is not None:
                w = self.sa_weights()[idx]
                if self.loss_computer.sa_apply_to_bc:
                    w_pde = w[:, :1]  # Только одна компонента для потенциала
                    w_bc = w[:, 1:]
                else:
                    w_pde = w
                    w_bc = None
                loss_laplace = torch.mean(w_pde[:, 0] * laplace)
            else:
                loss_laplace = torch.mean(laplace)
                w_bc = None

            # PDE потеря с учётом весов из конфигурации
            pde_loss = pde_mult * (
                self.loss_computer.loss_weights.get('laplace_weight', 1.0) * loss_laplace
            )
        else:
            # Оригинальная логика для Навье-Стокса
            cont = res['cont']
            mom1 = res[self.loss_computer.pde_components[1]]
            mom2 = res[self.loss_computer.pde_components[2]]

            # Сохраняем невязки для RAR (открепляем от графа)
            cont_res = cont.detach().cpu()
            mom1_res = mom1.detach().cpu()
            mom2_res = mom2.detach().cpu()
            total_res = cont_res + mom1_res + mom2_res

            # Обновляем скользящее среднее невязки для RAR
            alpha = 0.9
            self.point_residual[idx.cpu()] = alpha * self.point_residual[idx.cpu()] + (1 - alpha) * total_res.flatten()
            # Обновляем максимальную невязку (для карты)
            self.max_residual[idx.cpu()] = torch.max(self.max_residual[idx.cpu()], total_res.flatten())

            # Средние и максимальные значения для отчёта
            pde_raw = {
                'cont': cont.mean().item(),
                self.loss_computer.pde_components[1]: mom1.mean().item(),
                self.loss_computer.pde_components[2]: mom2.mean().item()
            }
            pde_max = {
                'cont': cont.max().item(),
                self.loss_computer.pde_components[1]: mom1.max().item(),
                self.loss_computer.pde_components[2]: mom2.max().item()
            }

            # Взвешивание PDE с учётом SA-весов (если используются)
            if self.sa_weights is not None:
                w = self.sa_weights()[idx]
                if self.loss_computer.sa_apply_to_bc:
                    w_pde = w[:, :3]
                    w_bc = w[:, 3:]
                else:
                    w_pde = w
                    w_bc = None
                loss_c = torch.mean(w_pde[:, 0] * cont)
                loss_1 = torch.mean(w_pde[:, 1] * mom1)
                loss_2 = torch.mean(w_pde[:, 2] * mom2)
            else:
                loss_c = torch.mean(cont)
                loss_1 = torch.mean(mom1)
                loss_2 = torch.mean(mom2)
                w_bc = None

            # PDE потеря с учётом весов из конфигурации
            pde_loss = pde_mult * (
                self.loss_computer.loss_weights['cont'] * loss_c +
                self.loss_computer.loss_weights[self.loss_computer.pde_components[1]] * loss_1 +
                self.loss_computer.loss_weights[self.loss_computer.pde_components[2]] * loss_2
        )

        # Граничные условия (только для soft)
        if self.loss_computer.bc_type == 'soft':
            # Для потенциала: первый аргумент - phi, для НС - u
            if self.loss_computer.use_potential:
                bc_comp, extra_grad = self.loss_computer.compute_bc_loss(phi, v_final, p_final, bt, x, y, sdf)
            else:
                bc_comp, extra_grad = self.loss_computer.compute_bc_loss(u_final, v_final, p_final, bt, x, y, sdf)
            bc_loss = sum(
                self.loss_computer.loss_weights.get(name, 0.0) * bc_comp[name]
                for name in bc_comp
            ) + extra_grad
        else:
            bc_comp = {}
            # Создаём нулевой тензор, который требует градиент (чтобы избежать ошибок в диагностике)
            bc_loss = torch.zeros(1, device=self.device, dtype=self.dtype, requires_grad=True).squeeze()

        # Если используются SA-веса и они применяются к BC, добавляем взвешивание
        if self.sa_weights is not None and self.loss_computer.sa_apply_to_bc and w_bc is not None and self.loss_computer.bc_type == 'soft':
            for i, name in enumerate(self.data_handler.boundary_type_names):
                if name in bc_comp and bc_comp[name] > 0:
                    bc_loss += torch.mean(w_bc[:, i] * bc_comp[name])

        # Интегральная потеря
        integral_loss = self.loss_computer.compute_integral_loss(u, v, bt, x, y) * self.loss_computer.integral_weight

        # Полная потеря
        total_loss = pde_loss + bc_loss + integral_loss

        # Средняя скорость на стенке (для мониторинга)
        wall_speed = None
        if self.loss_computer.bc_type == 'soft' and 'wall' in bc_comp and bc_comp.get('wall_phi', 0) > 0 and bt is not None:
            wall_idx = self.data_handler.boundary_type_names.index('wall')
            u_wall = u[bt[:, wall_idx] > 0.5]
            v_wall = v[bt[:, wall_idx] > 0.5]
            if len(u_wall) > 0:
                wall_speed = torch.sqrt(u_wall**2 + v_wall**2).mean().item()
        elif self.loss_computer.bc_type == 'soft' and 'wall' in bc_comp and bc_comp.get('wall', 0) > 0 and bt is not None:
            wall_idx = self.data_handler.boundary_type_names.index('wall')
            u_wall = u[bt[:, wall_idx] > 0.5]
            v_wall = v[bt[:, wall_idx] > 0.5]
            if len(u_wall) > 0:
                wall_speed = torch.sqrt(u_wall**2 + v_wall**2).mean().item()

        # Формируем возвращаемые компоненты потерь
        if self.loss_computer.use_potential:
            loss_components = {
                'laplace': loss_laplace,
                'bc': bc_loss,
                'integral': integral_loss
            }
        else:
            loss_components = {
                'cont': loss_c,
                'mom1': loss_1,
                'mom2': loss_2,
                'bc': bc_loss,
                'integral': integral_loss
            }

        return {
            'total': total_loss,
            'pde_raw': pde_raw,
            'pde_max': pde_max,
            'bc_comp': {k: v.item() for k, v in bc_comp.items()} if self.loss_computer.bc_type == 'soft' else {},
            'extra_grad': extra_grad.item() if self.loss_computer.bc_type == 'soft' else 0.0,
            'integral_loss': integral_loss.item() if torch.is_tensor(integral_loss) else integral_loss,
            'wall_speed': wall_speed,
            # Возвращаем отдельные тензоры потерь для градиентного анализа
            'loss_components': loss_components,
            'u': u.detach(),  # для профилей и спектров
            'v': v.detach(),
            'x': x.detach(),
            'y': y.detach(),
            'idx': idx.cpu()
        }

    # ------------------------------------------------------------------
    # Вычисление градиентов PDE и BC отдельно
    # ------------------------------------------------------------------
    def compute_grad_norms_separately(self, batch: Tuple) -> Dict[str, float]:
        """
        Вычисляет нормы градиентов от PDE loss и BC loss, а также косинус угла между ними.
        Возвращает словарь с ключами 'pde_norm', 'bc_norm', 'cosine'.
        """
        self.model.zero_grad()
        fwd = self.forward_pass(batch)

        # Извлекаем компоненты потерь (уже тензоры)
        if self.loss_computer.use_potential:
            loss_pde = fwd['loss_components']['laplace']
        else:
            loss_pde = fwd['loss_components']['cont'] + fwd['loss_components']['mom1'] + fwd['loss_components']['mom2']
        loss_bc = fwd['loss_components']['bc']
        # Интегральная потеря не включается в раздельные градиенты

        # Градиенты от PDE
        loss_pde.backward(retain_graph=True)
        grad_pde = torch.cat([p.grad.view(-1).clone() for p in self.model.parameters() if p.grad is not None])
        self.model.zero_grad()

        # Градиенты от BC
        loss_bc.backward(retain_graph=True)
        grad_bc = torch.cat([p.grad.view(-1).clone() for p in self.model.parameters() if p.grad is not None])
        self.model.zero_grad()

        norm_pde = torch.norm(grad_pde).item()
        norm_bc = torch.norm(grad_bc).item()
        if norm_pde > 0 and norm_bc > 0:
            cosine = torch.dot(grad_pde, grad_bc).item() / (norm_pde * norm_bc)
        else:
            cosine = 0.0

        return {'pde_norm': norm_pde, 'bc_norm': norm_bc, 'cosine': cosine}

    # ------------------------------------------------------------------
    # Получение предсказаний на всём датасете (для диагностики)
    # ------------------------------------------------------------------
    def get_predictions_on_dataset(self) -> Dict[str, np.ndarray]:
        """
        Возвращает предсказания модели на всём датасете.
        Учитывает hard constraints, если они используются.
        Для потенциала: вычисляет скорости через градиенты.
        Возвращает словарь с ключами 'x', 'y', 'u', 'v', 'p', 'phi' (физические единицы).
        """
        self.model.eval()
        dataloader = self.data_handler.get_dataloader()
        all_u = []
        all_v = []
        all_p = []
        all_phi = []
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
                
                if self.loss_computer.use_potential:
                    # Для потенциала: out[:, 0:1] - это φ
                    phi = out[:, 0:1]
                    # Скорости вычисляются через градиенты потенциала
                    # Для inference используем численные градиенты или safe_grad без retains_grad
                    from utils import safe_grad
                    # Требуем градиенты для вычисления скоростей
                    x_grad = x.clone().requires_grad_(True)
                    y_grad = y.clone().requires_grad_(True)
                    
                    # Пересчитываем выход с градиентами
                    input_tensor_grad = self.loss_computer.build_input_tensor(x_grad, y_grad, sdf, bt)
                    out_grad = self.model(input_tensor_grad)
                    phi_grad = out_grad[:, 0:1]
                    
                    u = safe_grad(phi_grad, x_grad)
                    v = safe_grad(phi_grad, y_grad)
                    p = torch.zeros_like(phi_grad)
                    
                    all_phi.append(phi_grad.cpu().numpy().flatten())
                else:
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

        # Конкатенация
        x_np = np.concatenate(all_x)
        y_np = np.concatenate(all_y)
        u_np = np.concatenate(all_u)
        v_np = np.concatenate(all_v)
        p_np = np.concatenate(all_p)
        
        if self.loss_computer.use_potential:
            phi_np = np.concatenate(all_phi)
        else:
            phi_np = np.zeros_like(u_np)

        # Переводим в физические единицы
        chord = self.data_handler.chord
        U_inf = self.data_handler.U_inf
        rho = self.data_handler.rho
        p_dyn = rho * U_inf**2

        return {
            'x': x_np * chord,
            'y': y_np * chord,
            'u': u_np * U_inf,
            'v': v_np * U_inf,
            'p': p_np * p_dyn,
            'phi': phi_np * U_inf * chord  # Потенциал имеет размерность [м²/с]
        }

    # ------------------------------------------------------------------
    # Получение профиля скорости в заданном сечении x (УСТАРЕЛО)
    # ------------------------------------------------------------------
    def get_profile_at_x(self, x_target: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        ВНИМАНИЕ: Этот метод устарел и использует начальные поля (u_target).
        Для диагностики используйте data_handler.interpolate_profile с предсказаниями модели.
        """
        logger.warning("Метод get_profile_at_x устарел. Используйте data_handler.interpolate_profile с предсказаниями модели.")
        dataset = self.data_handler.dataset
        x_np = dataset.x.cpu().numpy().flatten()
        y_np = dataset.y.cpu().numpy().flatten()
        u_np = dataset.u_target.cpu().numpy().flatten()  # используем target как начальное приближение

        # Выбираем точки, близкие к x_target
        tol = 0.05 * (self.data_handler.x_max - self.data_handler.x_min)
        mask = np.abs(x_np - x_target) < tol
        if np.sum(mask) < 3:
            logger.warning(f"Мало точек в сечении x={x_target}, интерполяция может быть неточной.")
        y_vals = y_np[mask]
        u_vals = u_np[mask]

        # Сортируем по y
        sort_idx = np.argsort(y_vals)
        y_sorted = y_vals[sort_idx]
        u_sorted = u_vals[sort_idx]

        return y_sorted, u_sorted

    # ------------------------------------------------------------------
    # Вычисление спектра профиля (УСТАРЕЛО)
    # ------------------------------------------------------------------
    def compute_spectrum(self, y_vals: np.ndarray, u_vals: np.ndarray, spacing: Optional[float] = None) -> Dict:
        """
        Устаревший метод. Для диагностики используйте visualizer.plot_2d_spectrum.
        """
        logger.warning("Метод compute_spectrum устарел.")
        if len(y_vals) < 2:
            return {'freq': np.array([]), 'amp': np.array([])}
        # Обеспечиваем равномерную сетку: интерполируем u на равномерную сетку
        y_uniform = np.linspace(y_vals.min(), y_vals.max(), num=len(y_vals))
        u_uniform = np.interp(y_uniform, y_vals, u_vals)
        if spacing is None:
            spacing = y_uniform[1] - y_uniform[0]
        # БПФ
        n = len(u_uniform)
        fft_vals = np.fft.fft(u_uniform)
        freqs = np.fft.fftfreq(n, d=spacing)
        # Берём положительные частоты
        positive = freqs > 0
        return {
            'freq': freqs[positive],
            'amp': np.abs(fft_vals[positive])
        }

    # ------------------------------------------------------------------
    # Вычисление производной du/dy в пристеночных точках
    # ------------------------------------------------------------------
    def compute_wall_gradient(self, threshold: float = 0.1) -> float:
        """
        Вычисляет среднее значение ∂u/∂y в точках, близких к стенке (y < threshold).
        """
        dataset = self.data_handler.dataset
        # Предполагаем, что стенка – это точки с типом 'wall'
        if 'wall' not in self.data_handler.boundary_type_names:
            return 0.0
        wall_idx = self.data_handler.boundary_type_names.index('wall')
        bt = dataset.boundary_types.cpu().numpy()
        wall_mask = bt[:, wall_idx] > 0.5
        if not np.any(wall_mask):
            return 0.0
        # Берём точки вблизи стенки
        y_np = dataset.y.cpu().numpy().flatten()
        near_wall = (y_np < threshold) & wall_mask
        if np.sum(near_wall) == 0:
            return 0.0
        indices = np.where(near_wall)[0]
        self.model.eval()
        with torch.no_grad():
            x_pts = dataset.x[indices].to(self.device, dtype=self.dtype).unsqueeze(1)
            y_pts = dataset.y[indices].to(self.device, dtype=self.dtype).unsqueeze(1)
            sdf_pts = dataset.sdf[indices].to(self.device, dtype=self.dtype).unsqueeze(1)
            bt_pts = dataset.boundary_types[indices].to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None
            input_tensor = self.loss_computer.build_input_tensor(x_pts, y_pts, sdf_pts, bt_pts)
            out = self.model(input_tensor)
            u_vals = out[:, 0:1]
        # Вычисляем градиент du/dy через autograd
        u_vals.sum().backward(retain_graph=True)
        grad_u_y = y_pts.grad
        if grad_u_y is not None:
            mean_grad = grad_u_y.mean().item()
        else:
            mean_grad = 0.0
        self.model.zero_grad()
        return mean_grad

    # ------------------------------------------------------------------
    # RAR с сохранением статистики и корректным расширением данных/весов
    # ------------------------------------------------------------------
    def apply_rar(self):
        """
        Добавляет точки с наибольшей невязкой в датасет, обновляет связанные структуры
        и сохраняет статистику по типам границ добавленных точек.
        Если используются SA-веса, они также расширяются, и выставляется флаг sa_weights_changed.
        ИСПРАВЛЕНО: новые точки инициализируются нулевой невязкой (вместо копирования старых значений),
        чтобы избежать дублирования высоких невязок.
        """
        if not self.rar_enable:
            return
        logger.info("Применение RAR: добавление точек с наибольшей невязкой...")
        top_k = min(self.rar_num_points, len(self.point_residual))
        if top_k == 0:
            return
        values, indices = torch.topk(self.point_residual, top_k)
        indices = indices.numpy()

        dataset = self.data_handler.dataset

        # Сохраняем статистику по типам границ добавленных точек
        if dataset.boundary_types is not None:
            bt = dataset.boundary_types[indices].cpu().numpy()
            counts = bt.sum(axis=0)
            stat = dict(zip(self.data_handler.boundary_type_names, counts))
            self.rar_history.append(stat)

        # Расширяем датасет через DataHandler
        self.data_handler.add_points(torch.tensor(indices, dtype=torch.long))

        # Обновляем point_residual и max_residual:
        # новые точки получают нулевую невязку (можно также вычислить заново, но это дорого)
        new_residual = torch.zeros(top_k, device='cpu', dtype=self.dtype)
        new_max = torch.zeros(top_k, device='cpu', dtype=self.dtype)
        self.point_residual = torch.cat([self.point_residual, new_residual], dim=0)
        self.max_residual = torch.cat([self.max_residual, new_max], dim=0)

        # Если используются SA-веса, расширяем их и устанавливаем флаг
        if self.sa_weights is not None:
            old_weights = self.sa_weights.log_weights.detach().cpu()
            # Для новых точек инициализируем веса так же, как в начале (init_val=1.0 -> log(1)=0)
            new_weights = torch.zeros(top_k, old_weights.shape[1], device='cpu')
            combined_weights = torch.cat([old_weights, new_weights], dim=0)
            # Заменяем параметр (это допустимо, так как оптимизатор будет пересоздан)
            self.sa_weights.log_weights = nn.Parameter(combined_weights.to(self.device))
            self.sa_weights_changed = True
            logger.warning("RAR: SA-веса расширены. Требуется пересоздание оптимизатора в solver.")

        logger.info(f"RAR: добавлено {top_k} точек. Теперь всего точек: {self.data_handler.n_points}")

    # ------------------------------------------------------------------
    # Шаг обучения (Adam)
    # ------------------------------------------------------------------
    def train_batch_adam(self, batch: Tuple, accum_step: bool, pde_mult: float = 1.0) -> Dict[str, Any]:
        """
        Выполняет один прямой проход и накопление градиентов для батча (Adam).
        Если accum_step=True, выполняет шаг оптимизации.

        Args:
            batch: кортеж данных
            accum_step: флаг, указывающий, нужно ли выполнять шаг оптимизации
            pde_mult: множитель для PDE потерь

        Returns:
            словарь с результатами (те же поля, что и в forward_pass, плюс grad_norm)
        """
        fwd = self.forward_pass(batch, pde_mult)
        total_loss = fwd['total'] / self.accum_steps

        # Обратное распространение
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Вычисляем норму градиентов (до клиппинга и шага)
        grad_norm = 0.0
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        if self.sa_weights is not None:
            for p in self.sa_weights.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        # Если пора делать шаг оптимизации
        if accum_step:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.sa_weights is not None:
                        torch.nn.utils.clip_grad_norm_(self.sa_weights.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.sa_weights is not None:
                        torch.nn.utils.clip_grad_norm_(self.sa_weights.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return {
            'total': fwd['total'].item(),
            'pde_raw': fwd['pde_raw'],
            'pde_max': fwd['pde_max'],
            'bc_comp': fwd['bc_comp'],
            'extra_grad': fwd['extra_grad'],
            'integral_loss': fwd['integral_loss'],
            'wall_speed': fwd['wall_speed'],
            'grad_norm': grad_norm
        }

    # ------------------------------------------------------------------
    # Эпоха обучения (Adam)
    # ------------------------------------------------------------------
    def train_epoch_adam(self, pde_mult: float = 1.0) -> Tuple[
        float,
        Dict[str, float],
        Dict[str, float],
        float,
        float,
        float,
        Dict[str, float],
        Dict[str, float],
        float
    ]:
        """
        Выполняет одну эпоху обучения с оптимизатором Adam.

        Returns:
            кортеж:
                avg_loss (float): средняя полная потеря за эпоху
                avg_pde (Dict[str, float]): средние значения PDE по компонентам
                avg_bc (Dict[str, float]): средние значения BC по компонентам
                avg_extra_grad (float): средняя дополнительная потеря от градиентов
                avg_integral (float): средняя интегральная потеря
                avg_wall_speed (float): средняя скорость на стенке
                avg_max (Dict[str, float]): средние максимумы PDE
                pde_rate (Dict[str, float]): отношение текущих потерь к предыдущим (для отчёта)
                avg_grad_norm (float): средняя норма градиентов
        """
        self.model.train()
        dataloader = self.data_handler.get_dataloader()

        # Инициализация накопителей
        epoch_losses = {
            'total': 0.0,
            'pde_raw': {c: 0.0 for c in self.loss_computer.pde_components},
            'pde_max': {c: 0.0 for c in self.loss_computer.pde_components},
            'bc_comp': {b: 0.0 for b in self.data_handler.boundary_type_names} if self.loss_computer.bc_type == 'soft' else {},
            'extra_grad': 0.0,
            'integral_loss': 0.0,
            'wall_speed_sum': 0.0,
            'n_wall_batches': 0,
            'grad_norm_sum': 0.0,
            'grad_norm_count': 0
        }
        n_batches = 0

        for i, batch in enumerate(dataloader):
            accum_step = ((i + 1) % self.accum_steps == 0) or (i == len(dataloader) - 1)
            batch_losses = self.train_batch_adam(batch, accum_step, pde_mult)

            epoch_losses['total'] += batch_losses['total']
            for k, v in batch_losses['pde_raw'].items():
                epoch_losses['pde_raw'][k] += v
            for k, v in batch_losses['pde_max'].items():
                epoch_losses['pde_max'][k] += v
            if self.loss_computer.bc_type == 'soft':
                for k, v in batch_losses['bc_comp'].items():
                    epoch_losses['bc_comp'][k] += v
                epoch_losses['extra_grad'] += batch_losses['extra_grad']
            epoch_losses['integral_loss'] += batch_losses['integral_loss']
            if batch_losses['wall_speed'] is not None:
                epoch_losses['wall_speed_sum'] += batch_losses['wall_speed']
                epoch_losses['n_wall_batches'] += 1
            if batch_losses['grad_norm'] > 0:
                epoch_losses['grad_norm_sum'] += batch_losses['grad_norm']
                epoch_losses['grad_norm_count'] += 1
            n_batches += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  Batch {i+1}/{len(dataloader)} | текущая loss: {batch_losses['total']:.4e}")

        # Усреднение
        avg_loss = epoch_losses['total'] / n_batches
        avg_pde = {k: v / n_batches for k, v in epoch_losses['pde_raw'].items()}
        avg_max = {k: v / n_batches for k, v in epoch_losses['pde_max'].items()}
        if self.loss_computer.bc_type == 'soft':
            avg_bc = {k: v / n_batches for k, v in epoch_losses['bc_comp'].items()}
            avg_extra_grad = epoch_losses['extra_grad'] / n_batches
        else:
            avg_bc = {}
            avg_extra_grad = 0.0
        avg_integral = epoch_losses['integral_loss'] / n_batches
        avg_wall_speed = (epoch_losses['wall_speed_sum'] / epoch_losses['n_wall_batches']
                          if epoch_losses['n_wall_batches'] > 0 else 0.0)
        avg_grad_norm = epoch_losses['grad_norm_sum'] / epoch_losses['grad_norm_count'] if epoch_losses['grad_norm_count'] > 0 else 0.0

        # Обновление скользящих средних для адаптивных весов
        for comp, val in avg_pde.items():
            self.loss_ma[comp].append(val)
        if self.loss_computer.bc_type == 'soft':
            for comp, val in avg_bc.items():
                self.loss_ma[comp].append(val)

        # Вычисление rate (отношение к предыдущему значению) – для отчёта
        pde_rate = {c: 1.0 for c in self.loss_computer.pde_components}

        return avg_loss, avg_pde, avg_bc, avg_extra_grad, avg_integral, avg_wall_speed, avg_max, pde_rate, avg_grad_norm

    # ------------------------------------------------------------------
    # Эпоха обучения (L-BFGS)
    # ------------------------------------------------------------------
    def train_epoch_lbfgs(self) -> Tuple[
        float,
        Dict[str, float],
        Dict[str, float],
        float,
        float,
        float,
        Dict[str, float],
        Dict[str, float],
        float
    ]:
        """
        Выполняет одну эпоху обучения с оптимизатором L-BFGS (один шаг на всём датасете).

        Returns:
            аналогично train_epoch_adam, но grad_norm = 0.0 (не вычисляется)
        """
        self.model.train()
        dataloader = self.data_handler.get_dataloader()

        pde_raw_list = []
        pde_max_list = []
        bc_comp_list = []
        integral_loss_list = []
        wall_speed_list = []
        n_batches = 0

        def closure():
            self.optimizer.zero_grad()
            total_loss = 0.0
            nonlocal n_batches
            pde_raw_list.clear()
            pde_max_list.clear()
            bc_comp_list.clear()
            integral_loss_list.clear()
            wall_speed_list.clear()
            n_batches = 0
            for batch in dataloader:
                fwd = self.forward_pass(batch, pde_mult=1.0)
                total_loss += fwd['total']

                pde_raw_list.append(fwd['pde_raw'])
                pde_max_list.append(fwd['pde_max'])
                if self.loss_computer.bc_type == 'soft':
                    bc_comp_list.append(fwd['bc_comp'])
                integral_loss_list.append(fwd['integral_loss'])
                if fwd['wall_speed'] is not None:
                    wall_speed_list.append(fwd['wall_speed'])
                n_batches += 1

            total_loss.backward()
            return total_loss

        loss = self.optimizer.step(closure)

        # Усреднение результатов
        avg_pde = {}
        avg_max = {}
        for key in self.loss_computer.pde_components:
            vals_pde = [d[key] for d in pde_raw_list]
            vals_max = [d[key] for d in pde_max_list]
            avg_pde[key] = np.mean(vals_pde) if vals_pde else 0.0
            avg_max[key] = np.mean(vals_max) if vals_max else 0.0

        if self.loss_computer.bc_type == 'soft':
            avg_bc = {}
            for key in self.data_handler.boundary_type_names:
                vals = [d.get(key, 0.0) for d in bc_comp_list]
                avg_bc[key] = np.mean(vals) if vals else 0.0
        else:
            avg_bc = {}

        avg_integral = np.mean(integral_loss_list) if integral_loss_list else 0.0
        avg_wall_speed = np.mean(wall_speed_list) if wall_speed_list else 0.0
        avg_loss = loss.item() if torch.is_tensor(loss) else loss
        pde_rate = {c: 1.0 for c in self.loss_computer.pde_components}
        avg_extra_grad = 0.0  # не вычисляется для L-BFGS

        return avg_loss, avg_pde, avg_bc, avg_extra_grad, avg_integral, avg_wall_speed, avg_max, pde_rate, 0.0

    # ------------------------------------------------------------------
    # Адаптивная балансировка весов BC (через LossComputer)
    # ------------------------------------------------------------------
    def update_adaptive_weights(self, loss_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Обновляет веса граничных условий на основе скользящих средних потерь.
        Использует метод LossComputer.update_adaptive_weights для инкапсуляции.

        Args:
            loss_weights: текущие веса (словарь)

        Returns:
            обновлённый словарь весов
        """
        if not self.adaptive_weights:
            return loss_weights

        # Вычисляем среднее значение PDE потерь
        pde_vals = []
        for comp in self.loss_computer.pde_components:
            if self.loss_ma[comp]:
                pde_vals.append(np.mean(list(self.loss_ma[comp])))
        if not pde_vals:
            return loss_weights

        avg_pde = np.mean(pde_vals)
        target = avg_pde * np.mean([loss_weights[comp] for comp in self.loss_computer.pde_components])
        eps = 1e-8
        new_weights = {}
        for key in self.data_handler.boundary_type_names:
            if key in self.loss_ma and self.loss_ma[key]:
                avg_bc = np.mean(list(self.loss_ma[key]))
                new_weights[key] = target / (avg_bc + eps)
                new_weights[key] = np.clip(new_weights[key], 0.1, 10000.0)
            else:
                new_weights[key] = loss_weights.get(key, 1.0)

        # Передаём обновление в LossComputer
        self.loss_computer.update_adaptive_weights(new_weights)
        logger.info(f"Адаптивные веса BC обновлены: { {k: f'{v:.2f}' for k, v in new_weights.items()} }")
        return self.loss_computer.loss_weights  # возвращаем обновлённый словарь

    # ------------------------------------------------------------------
    # Сброс оптимизатора с шумом
    # ------------------------------------------------------------------
    def reset_optimizer(self, reset_noise: Optional[float] = None):
        """
        Добавляет шум к параметрам модели и (опционально) SA-весов,
        после чего пересоздаёт оптимизатор.

        Args:
            reset_noise: амплитуда шума (если None, используется self.reset_noise)
        """
        if self.using_lbfgs:
            logger.info("Сброс оптимизатора недоступен в режиме L-BFGS.")
            return

        noise = reset_noise if reset_noise is not None else self.reset_noise
        logger.info(f"Перезапуск оптимизатора с шумом {noise}")

        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * noise)
            if self.sa_weights is not None:
                for param in self.sa_weights.parameters():
                    param.add_(torch.randn_like(param) * noise)

        # Примечание: оптимизатор должен быть пересоздан вызывающим кодом,
        # так как здесь нет доступа к конфигурации для создания нового.
        # Возвращаем управление, чтобы solver мог пересоздать оптимизатор.
        logger.info("Шум добавлен. Требуется пересоздание оптимизатора.")