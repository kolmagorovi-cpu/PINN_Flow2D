# Исправленный файл: metrics.py (должен находиться в solver_components/ или в корне)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль metrics для проекта PINN Learning.
Содержит класс MetricsCollector, отвечающий за сбор, вывод и сохранение метрик обучения,
включая расширенный набор диагностических метрик.

Исправленная версия:
- Уточнены комментарии о размерности возвращаемых значений.
- Добавлена защита от отсутствия ключей в истории.
- Метод get_boundary_avg_velocities теперь явно возвращает нормированные значения.
- Добавлен метод get_boundary_avg_velocities_physical для получения физических скоростей.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Сбор, вывод в консоль, логирование и финальный расчёт метрик обучения.
    Хранит историю обучения и предоставляет методы для её обновления и визуализации в текстовом виде.
    """

    def __init__(
        self,
        model: nn.Module,
        data_handler,
        loss_computer,
        config: Dict,
        device: torch.device,
        dtype: torch.dtype,
        logs_dir: str,
        config_name: str = "unnamed",
        reference_data: Optional[Dict] = None
    ):
        """
        Args:
            model: нейросетевая модель
            data_handler: объект DataHandler (с метаданными, датасетом)
            loss_computer: объект LossComputer (знает компоненты PDE, имена границ)
            config: полная конфигурация (для доступа к настройкам вывода и диагностики)
            device: устройство для вычислений
            dtype: тип данных (torch.float32, torch.float64)
            logs_dir: директория для сохранения логов
            config_name: имя конфигурации (для заголовков)
            reference_data: словарь с эталонным решением (опционально)
        """
        self.model = model
        self.data_handler = data_handler
        self.loss_computer = loss_computer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.logs_dir = logs_dir
        self.config_name = config_name
        self.reference_data = reference_data

        # История обучения (расширенная)
        self.history = {
            'loss': [],
            'pde': {c: [] for c in loss_computer.pde_components},
            'bc': {b: [] for b in data_handler.boundary_type_names} if loss_computer.bc_type == 'soft' else {},
            'wall_speed': [],
            'sa_weights_mean': [] if loss_computer.use_sa_weights else None,
            'integral_loss': [] if loss_computer.integral_condition else None,
            'max_res': {c: [] for c in loss_computer.pde_components},
            'rate': {c: [] for c in loss_computer.pde_components},
            'grad_norm': [],
            'total_bc': [] if loss_computer.bc_type == 'soft' else None,
            'total_pde': [],
            'fvm_loss': [],
            # Новые поля для диагностики
            'grad_norm_pde': [],
            'grad_norm_bc': [],
            'grad_cosine': [],
            'mass_balance_error': [],
            'max_v_top': [],
            'avg_u_out': [],
            'avg_u_in': [],
            'wall_gradient': [],
            'ntk_condition': [],
        }

        # Для кэширования средних скоростей на границах (нормированные)
        self._last_avg_epoch = -1
        self._cached_u_in_norm = 0.0
        self._cached_u_out_norm = 0.0

        # Файл лога
        self.log_file = None

        os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Обновление истории
    # ------------------------------------------------------------------
    def update_history(
        self,
        loss: float,
        pde_raw: Dict[str, float],
        bc_comp: Dict[str, float],
        integral_loss: float,
        wall_speed: float,
        pde_max: Dict[str, float],
        pde_rate: Dict[str, float],
        grad_norm: float,
        extra_diag: Optional[Dict] = None
    ) -> None:
        """
        Добавляет метрики текущей эпохи в историю.

        Args:
            loss: полная потеря
            pde_raw: словарь средних PDE невязок
            bc_comp: словарь потерь на границах
            integral_loss: интегральная потеря
            wall_speed: средняя скорость на стенке
            pde_max: словарь максимальных PDE невязок
            pde_rate: отношение текущих потерь к предыдущим
            grad_norm: норма градиентов
            extra_diag: словарь с дополнительными диагностическими метриками (опционально)
        """
        self.history['loss'].append(loss)
        for k, v in pde_raw.items():
            self.history['pde'][k].append(v)
        if self.loss_computer.bc_type == 'soft':
            for k, v in bc_comp.items():
                self.history['bc'][k].append(v)
        self.history['wall_speed'].append(wall_speed)
        if self.history['sa_weights_mean'] is not None:
            if extra_diag and 'sa_weights_mean' in extra_diag:
                self.history['sa_weights_mean'].append(extra_diag['sa_weights_mean'])
        if self.history['integral_loss'] is not None:
            self.history['integral_loss'].append(integral_loss)
        for k, v in pde_max.items():
            self.history['max_res'][k].append(v)
        for k, v in pde_rate.items():
            self.history['rate'][k].append(v)
        self.history['grad_norm'].append(grad_norm)

        total_pde = sum(pde_raw.values())
        self.history['total_pde'].append(total_pde)
        if self.loss_computer.bc_type == 'soft':
            total_bc = sum(bc_comp.values())
            self.history['total_bc'].append(total_bc)

        # Добавляем дополнительные метрики, если они есть
        if extra_diag:
            if 'fvm_loss' in extra_diag:
                self.history['fvm_loss'].append(extra_diag['fvm_loss'])
            if 'grad_norm_pde' in extra_diag:
                self.history['grad_norm_pde'].append(extra_diag['grad_norm_pde'])
            if 'grad_norm_bc' in extra_diag:
                self.history['grad_norm_bc'].append(extra_diag['grad_norm_bc'])
            if 'grad_cosine' in extra_diag:
                self.history['grad_cosine'].append(extra_diag['grad_cosine'])
            if 'mass_balance_error' in extra_diag:
                self.history['mass_balance_error'].append(extra_diag['mass_balance_error'])
            if 'max_v_top' in extra_diag:
                self.history['max_v_top'].append(extra_diag['max_v_top'])
            if 'avg_u_in' in extra_diag:
                self.history['avg_u_in'].append(extra_diag['avg_u_in'])
            if 'avg_u_out' in extra_diag:
                self.history['avg_u_out'].append(extra_diag['avg_u_out'])
            if 'wall_gradient' in extra_diag:
                self.history['wall_gradient'].append(extra_diag['wall_gradient'])
            if 'ntk_condition' in extra_diag:
                self.history['ntk_condition'].append(extra_diag['ntk_condition'])

    # ------------------------------------------------------------------
    # Вычисление средней скорости на входе/выходе (с кэшированием)
    # ------------------------------------------------------------------
    def get_boundary_avg_velocities(self, epoch: int, interval: int = 1) -> Tuple[float, float]:
        """
        Возвращает средние значения u на inlet и outlet (НОРМИРОВАННЫЕ).
        Вычисление происходит только если epoch кратно interval и эпоха изменилась.

        Args:
            epoch: текущий номер эпохи
            interval: интервал обновления (по умолчанию 1 – обновляется каждую эпоху)

        Returns:
            (avg_u_in_norm, avg_u_out_norm) – средние скорости на входе и выходе (нормированные).
        """
        if epoch % interval == 0 and epoch != self._last_avg_epoch:
            self._update_boundary_avg_velocities()
            self._last_avg_epoch = epoch
        return self._cached_u_in_norm, self._cached_u_out_norm

    def get_boundary_avg_velocities_physical(self, epoch: int, interval: int = 1) -> Tuple[float, float]:
        """
        Возвращает средние значения u на inlet и outlet в физических единицах (м/с).
        """
        u_in_norm, u_out_norm = self.get_boundary_avg_velocities(epoch, interval)
        return u_in_norm * self.data_handler.U_inf, u_out_norm * self.data_handler.U_inf

    def _update_boundary_avg_velocities(self) -> None:
        """Вычисляет средние скорости на inlet/outlet с помощью модели (нормированные)."""
        was_training = self.model.training
        self.model.eval()
        dataloader = self.data_handler.get_boundary_eval_dataloader(('inlet', 'outlet'))
        u_inlet_sum = 0.0
        u_outlet_sum = 0.0
        count_inlet = 0
        count_outlet = 0

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                # Приводим к нужному устройству и типу данных
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = sdf.to(self.device, dtype=self.dtype)
                if bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None
                if bt is None:
                    continue
                mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
                u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)

                # Построение входного тензора
                input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                out = self.model(input_tensor)
                u_raw = out[:, 0:1]

                # Применение hard constraints (если нужно)
                if self.loss_computer.bc_type == 'hard':
                    u = torch.where(mask_u, u_bc, u_raw)
                else:
                    u = u_raw

                if 'inlet' in self.data_handler.boundary_type_names:
                    inlet_idx = self.data_handler.boundary_type_names.index('inlet')
                    mask = bt[:, inlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_inlet_sum += u[mask].sum().item()
                        count_inlet += mask.sum().item()
                if 'outlet' in self.data_handler.boundary_type_names:
                    outlet_idx = self.data_handler.boundary_type_names.index('outlet')
                    mask = bt[:, outlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_outlet_sum += u[mask].sum().item()
                        count_outlet += mask.sum().item()

        self._cached_u_in_norm = u_inlet_sum / count_inlet if count_inlet > 0 else 0.0
        self._cached_u_out_norm = u_outlet_sum / count_outlet if count_outlet > 0 else 0.0
        if was_training:
            self.model.train()

    # ------------------------------------------------------------------
    # Вывод таблицы потерь в консоль
    # ------------------------------------------------------------------
    def print_epoch_table(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        pde_raw: Dict[str, float],
        bc_comp: Dict[str, float],
        integral_loss: float,
        pde_max: Dict[str, float],
        epoch_time: float,
        total_time: float,
        fvm_loss: float = 0.0
    ) -> None:
        """
        Выводит подробную таблицу с компонентами потерь за эпоху.

        Args:
            epoch: текущая эпоха
            total_epochs: общее число эпох
            loss: полная потеря
            pde_raw: словарь средних PDE невязок
            bc_comp: словарь потерь на границах
            integral_loss: интегральная потеря
            pde_max: словарь максимальных PDE невязок
            epoch_time: время выполнения эпохи (сек)
            total_time: общее время с начала обучения (сек)
        """
        print("\n" + "=" * 100)
        print(f"Epoch = {epoch:4d}/{total_epochs}    Time = {epoch_time:6.2f} s    Total time = {total_time:8.2f} s")
        print("-" * 100)
        print("|       Loss       | Rate | RMS    | Max    | Weight |  W RMS  |  W Max  |")
        print("+------------------+------+--------+--------+--------+---------+---------+")

        def fmt_line(name, rate, rms, max_val, weight, w_rms, w_max):
            return f"| {name:<16} | {rate:4.2f} | {rms:7.1e} | {max_val:7.1e} | {weight:7.1e} | {w_rms:7.1e} | {w_max:7.1e} |"

        w = self.loss_computer.loss_weights

        # Continuity
        rate_c = self._compute_rate('pde', 'cont', pde_raw.get('cont', 0))
        w_c = w.get('cont', 0)
        w_rms_c = pde_raw.get('cont', 0) * w_c
        w_max_c = pde_max.get('cont', 0) * w_c
        print(fmt_line('Continuity', rate_c, pde_raw.get('cont', 0),
                       pde_max.get('cont', 0), w_c, w_rms_c, w_max_c))

        # Momentum X/R
        mom_x_key = self.loss_computer.pde_components[1]
        rate_mx = self._compute_rate('pde', mom_x_key, pde_raw.get(mom_x_key, 0))
        w_mx = w.get(mom_x_key, 0)
        w_rms_mx = pde_raw.get(mom_x_key, 0) * w_mx
        w_max_mx = pde_max.get(mom_x_key, 0) * w_mx
        mom_x_name = 'Momentum X' if self.loss_computer.flow_type != 'axisymmetric' else 'Momentum R'
        print(fmt_line(mom_x_name, rate_mx, pde_raw.get(mom_x_key, 0),
                       pde_max.get(mom_x_key, 0), w_mx, w_rms_mx, w_max_mx))

        # Momentum Y/Z
        mom_y_key = self.loss_computer.pde_components[2]
        rate_my = self._compute_rate('pde', mom_y_key, pde_raw.get(mom_y_key, 0))
        w_my = w.get(mom_y_key, 0)
        w_rms_my = pde_raw.get(mom_y_key, 0) * w_my
        w_max_my = pde_max.get(mom_y_key, 0) * w_my
        mom_y_name = 'Momentum Y' if self.loss_computer.flow_type != 'axisymmetric' else 'Momentum Z'
        print(fmt_line(mom_y_name, rate_my, pde_raw.get(mom_y_key, 0),
                       pde_max.get(mom_y_key, 0), w_my, w_rms_my, w_max_my))

        # Boundary conditions
        if self.loss_computer.bc_type == 'soft':
            for name in self.data_handler.boundary_type_names:
                val = bc_comp.get(name, 0)
                rate_bc = self._compute_rate('bc', name, val)
                weight = w.get(name, 0)
                w_rms_bc = val * weight
                w_max_bc = val * weight
                disp_name = name.replace('_', ' ').title()
                print(fmt_line(disp_name, rate_bc, val, val, weight, w_rms_bc, w_max_bc))
        else:
            # При hard границах потерь BC нет, но выводим заглушки
            for name in self.data_handler.boundary_type_names:
                print(fmt_line(name.replace('_', ' ').title(), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Integral loss
        if self.loss_computer.integral_condition:
            rate_int = self._compute_rate('integral', None, integral_loss)
            w_int = self.loss_computer.integral_weight
            w_rms_int = integral_loss * w_int
            w_max_int = integral_loss * w_int
            print(fmt_line('Integral Flow', rate_int, integral_loss, integral_loss,
                           w_int, w_rms_int, w_max_int))

        # FVM loss (если включена)
        fvm_w = getattr(self.loss_computer, 'fvm_weight', 0.0)
        if fvm_w > 0.0:
            rate_fvm = self._compute_rate('fvm', None, fvm_loss)
            fvm_raw = fvm_loss / fvm_w if fvm_w else fvm_loss
            print(fmt_line('FVM Loss', rate_fvm, fvm_raw, fvm_raw, fvm_w, fvm_loss, fvm_loss))

        # Totals
        total_bc = sum(bc_comp.values()) if self.loss_computer.bc_type == 'soft' else 0.0
        rate_tot_bc = self._compute_rate('total_bc', None, total_bc)
        print(fmt_line('Total BC', rate_tot_bc, total_bc, total_bc, 0.0, 0.0, 0.0))

        total_pde = sum(pde_raw.values())
        rate_tot_pde = self._compute_rate('total_pde', None, total_pde)
        print(fmt_line('Total PDE', rate_tot_pde, total_pde, total_pde, 0.0, 0.0, 0.0))

        rate_tot = self._compute_rate('loss', None, loss)
        print(fmt_line('Total Loss', rate_tot, loss, loss, 0.0, 0.0, 0.0))

        print("=" * 100)

    def _compute_rate(self, category: str, key: Optional[str], current: float) -> float:
        """
        Вычисляет отношение текущего значения к предыдущему для заданной категории.

        Args:
            category: 'pde', 'bc', 'integral', 'total_bc', 'total_pde', 'loss'
            key: ключ внутри категории (для PDE и BC)
            current: текущее значение

        Returns:
            отношение current / previous (с защитой от деления на ноль)
        """
        eps = 1e-12
        if len(self.history['loss']) < 1:
            return 1.0

        if category == 'pde' and key is not None:
            hist = self.history['pde'].get(key, [])
            if len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'bc' and key is not None:
            hist = self.history['bc'].get(key, [])
            if len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'integral':
            hist = self.history.get('integral_loss', [])
            if hist and len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'total_bc':
            if self.history.get('total_bc') and len(self.history['total_bc']) >= 2:
                prev = self.history['total_bc'][-2]
            else:
                return 1.0
        elif category == 'total_pde':
            if self.history.get('total_pde') and len(self.history['total_pde']) >= 2:
                prev = self.history['total_pde'][-2]
            else:
                return 1.0
        elif category == 'fvm':
            hist = self.history.get('fvm_loss', [])
            if hist and len(hist) >= 2:
                prev = hist[-2]
            else:
                return 1.0
        elif category == 'loss':
            if len(self.history['loss']) >= 2:
                prev = self.history['loss'][-2]
            else:
                return 1.0
        else:
            return 1.0

        return current / (prev + eps)

    # ------------------------------------------------------------------
    # Логирование в файл
    # ------------------------------------------------------------------
    def save_log(self, text: str) -> None:
        """
        Записывает строку в лог-файл (создаётся при первом вызове).

        Args:
            text: текст для записи
        """
        if self.log_file is None:
            log_name = f"{self.config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_path = os.path.join(self.logs_dir, log_name)
            self.log_file = open(log_path, 'w', encoding='utf-8')
        self.log_file.write(text + "\n")
        self.log_file.flush()

    # ------------------------------------------------------------------
    # Финальные метрики
    # ------------------------------------------------------------------
    def compute_final_metrics(self) -> Dict[str, Any]:
        """
        Вычисляет итоговые метрики после завершения обучения:
        - средние скорости на входе/выходе (нормированные и физические)
        - ошибка баланса масс
        - финальные потери (PDE, BC, integral)
        - лучшая эпоха
        - финальная норма градиентов
        - относительная L2-ошибка, если есть эталонное решение

        Returns:
            словарь с метриками
        """
        # Последние значения из истории (с защитой от пустых списков)
        pde_hist = self.history.get('pde', {})
        if pde_hist and all(len(pde_hist.get(c, [])) > 0 for c in self.loss_computer.pde_components):
            final_pde_loss = sum(pde_hist[c][-1] for c in self.loss_computer.pde_components)
        else:
            final_pde_loss = 0.0

        bc_hist = self.history.get('bc', {})
        final_bc_loss = sum(v[-1] for v in bc_hist.values() if v and len(v) > 0) if bc_hist else 0.0

        int_hist = self.history.get('integral_loss')
        final_integral_loss = int_hist[-1] if int_hist and len(int_hist) > 0 else 0.0

        # Средние скорости на границах (нормированные)
        avg_u_in_norm, avg_u_out_norm = self._cached_u_in_norm, self._cached_u_out_norm
        avg_u_in_phys = avg_u_in_norm * self.data_handler.U_inf
        avg_u_out_phys = avg_u_out_norm * self.data_handler.U_inf

        # Ошибка баланса масс (по нормализованным значениям)
        Qin = avg_u_in_norm
        Qout = avg_u_out_norm
        mass_balance_error = abs(Qin - Qout) / (abs(Qin) + 1e-12) if Qin != 0 else 0.0

        # Лучшая эпоха
        best_epoch = np.argmin(self.history['loss']) + 1 if self.history['loss'] else 0
        best_loss = min(self.history['loss']) if self.history['loss'] else float('inf')

        # Финальная норма градиентов
        final_grad_norm = self.history['grad_norm'][-1] if self.history['grad_norm'] else 0.0

        fvm_hist = self.history.get('fvm_loss', [])
        final_fvm_loss = fvm_hist[-1] if fvm_hist else 0.0

        metrics = {
            'experiment': self.config_name,
            'final_u_out_phys': avg_u_out_phys,
            'final_u_in_phys': avg_u_in_phys,
            'final_u_out_norm': avg_u_out_norm,
            'final_u_in_norm': avg_u_in_norm,
            'mass_balance_error': mass_balance_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_integral_loss': final_integral_loss,
            'final_fvm_loss': final_fvm_loss,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'final_gradient_norm': final_grad_norm,
            'relative_l2_error': 'N/A',
        }

        # Если есть эталонное решение, вычисляем L2-ошибку
        if self.reference_data is not None:
            try:
                from reference import interpolate_reference, compute_l2_errors

                # Получаем предсказания на всех точках датасета
                dataset = self.data_handler.dataset
                x_norm = dataset.x.to(self.device, dtype=self.dtype)
                y_norm = dataset.y.to(self.device, dtype=self.dtype)
                sdf_norm = dataset.sdf.to(self.device, dtype=self.dtype)
                bt = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None

                self.model.eval()
                with torch.no_grad():
                    input_tensor = self.loss_computer.build_input_tensor(x_norm, y_norm, sdf_norm, bt)
                    out = self.model(input_tensor)
                    u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                    if self.loss_computer.bc_type == 'hard':
                        u_pred_t, v_pred_t, p_pred_t = self.loss_computer.apply_hard_constraints(
                            u_raw, v_raw, p_raw,
                            dataset.mask_u_fixed.to(self.device).view(-1, 1),
                            dataset.mask_v_fixed.to(self.device).view(-1, 1),
                            dataset.mask_p_fixed.to(self.device).view(-1, 1),
                            dataset.u_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                            dataset.v_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                            dataset.p_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                        )
                    else:
                        u_pred_t, v_pred_t, p_pred_t = u_raw, v_raw, p_raw

                    p_dyn = 0.5 * self.data_handler.rho * self.data_handler.U_inf ** 2
                    u_pred = (u_pred_t * self.data_handler.U_inf).cpu().numpy().flatten()
                    v_pred = (v_pred_t * self.data_handler.U_inf).cpu().numpy().flatten()
                    p_pred = (p_pred_t * p_dyn + self.data_handler.p_ref).cpu().numpy().flatten()

                # Физические координаты
                x_phys = x_norm.cpu().numpy().flatten() * self.data_handler.chord
                y_phys = y_norm.cpu().numpy().flatten() * self.data_handler.chord

                # Интерполяция эталонных данных
                ref_interp = interpolate_reference(
                    self.reference_data,
                    x_phys,
                    y_phys,
                    method='linear'
                )

                # Вычисление ошибок (относительные)
                errors = compute_l2_errors(
                    u_pred, v_pred, p_pred,
                    ref_interp['u'], ref_interp['v'], ref_interp['p'],
                    norm='relative'
                )
                metrics['relative_l2_error'] = errors['mean']
                metrics['l2_u'] = errors['u']
                metrics['l2_v'] = errors['v']
                metrics['l2_p'] = errors['p']

                logger.info(f"Относительная L2-ошибка: {errors['mean']:.6e}")
            except Exception as e:
                logger.warning(f"Не удалось вычислить L2-ошибку: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Закрытие лог-файла
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Закрывает файл лога, если он открыт."""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
