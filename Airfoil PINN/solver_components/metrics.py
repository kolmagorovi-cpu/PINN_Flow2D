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
            # Новые поля для диагностики
            'grad_norm_pde': [],
            'grad_norm_bc': [],
            'grad_cosine': [],
            'mass_balance_error': [],
            'max_v_top': [],
            'avg_u_out': [],
            'avg_u_in': [],
            'avg_v_in': [],
            'avg_v_out': [],
            'wall_gradient': [],
        }

        # Для кэширования средних скоростей на границах (нормированные)
        self._last_avg_epoch = -1
        self._cached_u_in_norm = 0.0
        self._cached_u_out_norm = 0.0
        self._cached_v_in_norm = 0.0
        self._cached_v_out_norm = 0.0

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
                if k not in self.history['bc']:
                    self.history['bc'][k] = []
                self.history['bc'][k].append(v)
        self.history['wall_speed'].append(wall_speed)
        if self.history['sa_weights_mean'] is not None:
            # SA-веса обновляются отдельно в solver, добавим позже через отдельный метод
            pass
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
            if 'avg_v_in' in extra_diag:
                self.history['avg_v_in'].append(extra_diag['avg_v_in'])
            if 'avg_v_out' in extra_diag:
                self.history['avg_v_out'].append(extra_diag['avg_v_out'])
            if 'wall_gradient' in extra_diag:
                self.history['wall_gradient'].append(extra_diag['wall_gradient'])

    # ------------------------------------------------------------------
    # Вычисление средней скорости на входе/выходе (с кэшированием)
    # ------------------------------------------------------------------
    def get_boundary_avg_velocities(self, epoch: int, interval: int = 1) -> Tuple[float, float, float, float]:
        """
        Возвращает средние значения u и v на inlet и outlet (НОРМИРОВАННЫЕ).
        Вычисление происходит только если epoch кратно interval и эпоха изменилась.

        Args:
            epoch: текущий номер эпохи
            interval: интервал обновления (по умолчанию 1 – обновляется каждую эпоху)

        Returns:
            (avg_u_in_norm, avg_u_out_norm, avg_v_in_norm, avg_v_out_norm) – 
            средние скорости на входе и выходе (нормированные).
        """
        if epoch % interval == 0 and epoch != self._last_avg_epoch:
            self._update_boundary_avg_velocities()
            self._last_avg_epoch = epoch
        return self._cached_u_in_norm, self._cached_u_out_norm, self._cached_v_in_norm, self._cached_v_out_norm

    def get_boundary_avg_velocities_physical(self, epoch: int, interval: int = 1) -> Tuple[float, float, float, float]:
        """
        Возвращает средние значения u и v на inlet и outlet в физических единицах (м/с).
        """
        u_in_norm, u_out_norm, v_in_norm, v_out_norm = self.get_boundary_avg_velocities(epoch, interval)
        return (
            u_in_norm * self.data_handler.U_inf,
            u_out_norm * self.data_handler.U_inf,
            v_in_norm * self.data_handler.U_inf,
            v_out_norm * self.data_handler.U_inf,
        )

    def _update_boundary_avg_velocities(self) -> None:
        """Вычисляет средние скорости на inlet/outlet с помощью модели (нормированные)."""
        self.model.eval()
        dataloader = self.data_handler.get_dataloader()
        u_inlet_sum = 0.0
        u_outlet_sum = 0.0
        v_inlet_sum = 0.0
        v_outlet_sum = 0.0
        count_inlet = 0
        count_outlet = 0

        # Для потенциала скорости нужно вычислять u = ∂φ/∂x, v = ∂φ/∂y через градиент
        is_potential = self.loss_computer.use_potential

        with torch.no_grad():
            for batch in dataloader:
                idx, x, y, sdf, bt, mask_u, mask_v, mask_p, u_bc, v_bc, p_bc = batch
                # Приводим к нужному устройству и типу данных
                x = x.to(self.device, dtype=self.dtype)
                y = y.to(self.device, dtype=self.dtype)
                sdf = sdf.to(self.device, dtype=self.dtype)
                if bt is not None and bt.shape[1] > 0:
                    bt = bt.to(self.device, dtype=self.dtype)
                else:
                    bt = None
                mask_u = mask_u.to(self.device, dtype=torch.bool).view(-1, 1)
                u_bc = u_bc.to(self.device, dtype=self.dtype).view(-1, 1)

                # Построение входного тензора
                input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)

                if is_potential:
                    # Для потенциала: вычисляем u = ∂φ/∂x, v = ∂φ/∂y через градиент
                    # Внутри no_grad нужен enable_grad для вычисления градиентов
                    with torch.enable_grad():
                        x = x.requires_grad_(True)
                        y = y.requires_grad_(True)
                        input_tensor = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                        phi = self.model(input_tensor)
                        u = torch.autograd.grad(
                            phi, x, grad_outputs=torch.ones_like(phi),
                            create_graph=False, retain_graph=False, allow_unused=True
                        )[0]
                        v = torch.autograd.grad(
                            phi, y, grad_outputs=torch.ones_like(phi),
                            create_graph=False, retain_graph=False, allow_unused=True
                        )[0]
                        if u is None:
                            u = torch.zeros_like(x)
                        if v is None:
                            v = torch.zeros_like(y)
                else:
                    # Для Навье-Стокса: u, v — это прямой выход модели
                    out = self.model(input_tensor)
                    u_raw = out[:, 0:1]
                    v_raw = out[:, 1:2]

                    # Применение hard constraints (если нужно)
                    if self.loss_computer.bc_type == 'hard':
                        u = torch.where(mask_u, u_bc, u_raw)
                        mask_v = mask_v.to(self.device, dtype=torch.bool).view(-1, 1)
                        v_bc = v_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                        v = torch.where(mask_v, v_bc, v_raw)
                    else:
                        u = u_raw
                        v = v_raw

                if 'inlet' in self.data_handler.boundary_type_names:
                    inlet_idx = self.data_handler.boundary_type_names.index('inlet')
                    mask = bt[:, inlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_inlet_sum += u[mask].sum().item()
                        v_inlet_sum += v[mask].sum().item()
                        count_inlet += mask.sum().item()
                if 'outlet' in self.data_handler.boundary_type_names:
                    outlet_idx = self.data_handler.boundary_type_names.index('outlet')
                    mask = bt[:, outlet_idx] > 0.5 if bt is not None else torch.zeros_like(u, dtype=torch.bool)
                    if mask.sum() > 0:
                        u_outlet_sum += u[mask].sum().item()
                        v_outlet_sum += v[mask].sum().item()
                        count_outlet += mask.sum().item()

        self._cached_u_in_norm = u_inlet_sum / count_inlet if count_inlet > 0 else 0.0
        self._cached_u_out_norm = u_outlet_sum / count_outlet if count_outlet > 0 else 0.0
        self._cached_v_in_norm = v_inlet_sum / count_inlet if count_inlet > 0 else 0.0
        self._cached_v_out_norm = v_outlet_sum / count_outlet if count_outlet > 0 else 0.0

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
        extra_grad: float = 0.0
    ) -> None:
        """
        Выводит подробную таблицу с компонентами потерь за эпоху.
        Для потенциала скорости — упрощённый вывод (без PDE таблицы).
        Для Навье-Стокса — полная таблица с PDE невязками.

        Все столбцы имеют фиксированную ширину — ничего не съезжает.
        """
        is_potential = self.loss_computer.use_potential
        w = self.loss_computer.loss_weights

        # Форматирование строки с фиксированной шириной столбцов
        # | Название         | Rate | RMS       | Max       | Weight    | W RMS     | W Max     |
        #  18                 6      11          11          11          11          11
        COL_NAME   = 18
        COL_RATE   = 6
        COL_RMS    = 11
        COL_MAX    = 11
        COL_WEIGHT = 11
        COL_WRMS   = 11
        COL_WMAX   = 11

        def fmt_line(name, rate, rms, max_val, weight, w_rms, w_max):
            return (
                f"| {name:<{COL_NAME}} "
                f"| {rate:>{COL_RATE}.2f} "
                f"| {rms:>{COL_RMS}.1e} "
                f"| {max_val:>{COL_MAX}.1e} "
                f"| {weight:>{COL_WEIGHT}.1e} "
                f"| {w_rms:>{COL_WRMS}.1e} "
                f"| {w_max:>{COL_WMAX}.1e} |"
            )

        # Ширина всей таблицы: 1 + COL_NAME+2 + 1 + COL_RATE+2 + 1 + COL_RMS+2 + 1 + COL_MAX+2
        #                      + 1 + COL_WEIGHT+2 + 1 + COL_WRMS+2 + 1 + COL_WMAX+2 + 1
        TABLE_WIDTH = (
            1
            + (COL_NAME + 2) + 1 + (COL_RATE + 2) + 1 + (COL_RMS + 2) + 1 + (COL_MAX + 2)
            + 1 + (COL_WEIGHT + 2) + 1 + (COL_WRMS + 2) + 1 + (COL_WMAX + 2) + 1
        )

        HEADER = "|       Loss       | Rate | RMS       | Max       | Weight    | W RMS     | W Max     |"
        SEPARATOR = "+" + "-" * (COL_NAME + 2) + "+" + "-" * (COL_RATE + 2) + "+" \
                    + "-" * (COL_RMS + 2) + "+" + "-" * (COL_MAX + 2) + "+" \
                    + "-" * (COL_WEIGHT + 2) + "+" + "-" * (COL_WRMS + 2) + "+" \
                    + "-" * (COL_WMAX + 2) + "+"

        print()
        print("=" * TABLE_WIDTH)
        print(f"Epoch = {epoch:4d}/{total_epochs}    Time = {epoch_time:6.2f} s    Total time = {total_time:8.2f} s")
        print("-" * TABLE_WIDTH)

        if is_potential:
            # Для потенциала — компактный вывод без таблицы PDE
            laplace_val = pde_raw.get('laplace', 0.0)
            laplace_max = pde_max.get('laplace', 0.0)
            w_laplace = w.get('laplace', 1.0)

            print(f"  Laplace (PDE):  RMS = {laplace_val:11.1e}   Max = {laplace_max:11.1e}   "
                  f"Weight = {w_laplace:10.1e}   W_RMS = {laplace_val * w_laplace:11.1e}")
            print()

            # BC компоненты
            print(HEADER)
            print(SEPARATOR)

            pde_display_names = {
                'laplace': 'Laplace',
            }

            # PDE компонента (laplace)
            for comp in self.loss_computer.pde_components:
                disp_name = pde_display_names.get(comp, comp.upper())
                rate_comp = self._compute_rate('pde', comp, pde_raw.get(comp, 0))
                w_comp = w.get(comp, 0)
                rms_val = pde_raw.get(comp, 0)
                max_val = pde_max.get(comp, 0)
                w_rms = rms_val * w_comp
                w_max = max_val * w_comp
                print(fmt_line(disp_name, rate_comp, rms_val, max_val, w_comp, w_rms, w_max))

            # BC компоненты — только _phi ключи для потенциала
            if self.loss_computer.bc_type == 'soft':
                bc_keys = [k for k in bc_comp.keys() if k.endswith('_phi')]
                for name in bc_keys:
                    val = bc_comp.get(name, 0)
                    rate_bc = self._compute_rate('bc', name, val)
                    weight = w.get(name, 0)
                    if weight == 0 and name.endswith('_phi'):
                        base_name = name[:-4]
                        weight = w.get(base_name, 0)
                    # symmetry_grad и axis_symmetry_grad
                    base_name = name.replace('_phi', '')
                    if base_name == 'symmetry' and weight == 0:
                        weight = w.get('symmetry_grad', 0)
                    elif base_name == 'axis_symmetry' and weight == 0:
                        weight = w.get('axis_symmetry_grad', 0)
                    w_rms_bc = val * weight
                    w_max_bc = val * weight
                    disp_name = name.replace('_', ' ').title()
                    print(fmt_line(disp_name, rate_bc, val, val, weight, w_rms_bc, w_max_bc))
            else:
                for name in self.data_handler.boundary_type_names:
                    print(fmt_line(name.replace('_', ' ').title(), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

            # Extra grad (регуляризация)
            if extra_grad > 0:
                rate_eg = self._compute_rate('extra_grad', None, extra_grad)
                w_reg = w.get('uniform_reg', 0.0)
                print(fmt_line('Regularization', rate_eg, extra_grad, extra_grad, w_reg, extra_grad, extra_grad))

            # Totals
            total_bc = sum(bc_comp.values()) if self.loss_computer.bc_type == 'soft' else 0.0
            rate_tot_bc = self._compute_rate('total_bc', None, total_bc)
            print(fmt_line('Total BC', rate_tot_bc, total_bc, total_bc, 0.0, 0.0, 0.0))

            total_pde = sum(pde_raw.values())
            rate_tot_pde = self._compute_rate('total_pde', None, total_pde)
            print(fmt_line('Total PDE', rate_tot_pde, total_pde, total_pde, 0.0, 0.0, 0.0))

            rate_tot = self._compute_rate('loss', None, loss)
            print(fmt_line('Total Loss', rate_tot, loss, loss, 0.0, 0.0, 0.0))

        else:
            # Навье-Стокс — полная таблица
            print(HEADER)
            print(SEPARATOR)

            pde_display_names = {
                'cont': 'Continuity',
                'mom_x': 'Momentum X' if self.loss_computer.flow_type != 'axisymmetric' else 'Momentum R',
                'mom_y': 'Momentum Y' if self.loss_computer.flow_type != 'axisymmetric' else 'Momentum Z',
            }

            for comp in self.loss_computer.pde_components:
                disp_name = pde_display_names.get(comp, comp.upper())
                rate_comp = self._compute_rate('pde', comp, pde_raw.get(comp, 0))
                w_comp = w.get(comp, 0)
                rms_val = pde_raw.get(comp, 0)
                max_val = pde_max.get(comp, 0)
                w_rms = rms_val * w_comp
                w_max = max_val * w_comp
                print(fmt_line(disp_name, rate_comp, rms_val, max_val, w_comp, w_rms, w_max))

            # Boundary conditions — Навье-Стокс (ключи без '_phi')
            if self.loss_computer.bc_type == 'soft':
                bc_keys = [k for k in bc_comp.keys() if not k.endswith('_phi')]
                for name in bc_keys:
                    val = bc_comp.get(name, 0)
                    rate_bc = self._compute_rate('bc', name, val)
                    weight = w.get(name, 0)
                    w_rms_bc = val * weight
                    w_max_bc = val * weight
                    disp_name = name.replace('_', ' ').title()
                    print(fmt_line(disp_name, rate_bc, val, val, weight, w_rms_bc, w_max_bc))
            else:
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

            # Extra grad
            if extra_grad > 0:
                rate_eg = self._compute_rate('extra_grad', None, extra_grad)
                w_reg = w.get('symmetry_grad', 0)
                if w_reg == 0:
                    w_reg = w.get('axis_symmetry_grad', 0)
                print(fmt_line('Extra Grad', rate_eg, extra_grad, extra_grad, w_reg, extra_grad, extra_grad))

            # Totals
            total_bc = sum(bc_comp.values()) if self.loss_computer.bc_type == 'soft' else 0.0
            rate_tot_bc = self._compute_rate('total_bc', None, total_bc)
            print(fmt_line('Total BC', rate_tot_bc, total_bc, total_bc, 0.0, 0.0, 0.0))

            total_pde = sum(pde_raw.values())
            rate_tot_pde = self._compute_rate('total_pde', None, total_pde)
            print(fmt_line('Total PDE', rate_tot_pde, total_pde, total_pde, 0.0, 0.0, 0.0))

            rate_tot = self._compute_rate('loss', None, loss)
            print(fmt_line('Total Loss', rate_tot, loss, loss, 0.0, 0.0, 0.0))

        print("=" * TABLE_WIDTH)

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
        # Последние значения из истории
        final_pde_loss = sum(self.history['pde'][c][-1] for c in self.loss_computer.pde_components) if self.history['pde'] else 0.0
        final_bc_loss = sum(self.history['bc'].values())[-1] if self.history['bc'] else 0.0
        final_integral_loss = self.history['integral_loss'][-1] if self.history['integral_loss'] else 0.0

        # Средние скорости на границах (нормированные)
        avg_u_in_norm, avg_u_out_norm, avg_v_in_norm, avg_v_out_norm = \
            self._cached_u_in_norm, self._cached_u_out_norm, self._cached_v_in_norm, self._cached_v_out_norm
        avg_u_in_phys = avg_u_in_norm * self.data_handler.U_inf
        avg_u_out_phys = avg_u_out_norm * self.data_handler.U_inf
        avg_v_in_phys = avg_v_in_norm * self.data_handler.U_inf
        avg_v_out_phys = avg_v_out_norm * self.data_handler.U_inf

        # Ошибка баланса масс (по нормализованным значениям)
        Qin = avg_u_in_norm
        Qout = avg_u_out_norm
        mass_balance_error = abs(Qin - Qout) / (abs(Qin) + 1e-12) if Qin != 0 else 0.0

        # Лучшая эпоха
        best_epoch = np.argmin(self.history['loss']) + 1 if self.history['loss'] else 0
        best_loss = min(self.history['loss']) if self.history['loss'] else float('inf')

        # Финальная норма градиентов
        final_grad_norm = self.history['grad_norm'][-1] if self.history['grad_norm'] else 0.0

        metrics = {
            'experiment': self.config_name,
            'final_u_out_phys': avg_u_out_phys,
            'final_u_in_phys': avg_u_in_phys,
            'final_u_out_norm': avg_u_out_norm,
            'final_u_in_norm': avg_u_in_norm,
            'final_v_out_phys': avg_v_out_phys,
            'final_v_in_phys': avg_v_in_phys,
            'final_v_out_norm': avg_v_out_norm,
            'final_v_in_norm': avg_v_in_norm,
            'mass_balance_error': mass_balance_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_integral_loss': final_integral_loss,
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
                    if self.loss_computer.use_potential:
                        # Для потенциала: вычисляем скорости через градиенты
                        with torch.enable_grad():
                            x_norm_grad = x_norm.requires_grad_(True)
                            y_norm_grad = y_norm.requires_grad_(True)
                            input_tensor = self.loss_computer.build_input_tensor(x_norm_grad, y_norm_grad, sdf_norm, bt)
                            phi = self.model(input_tensor)
                            u_pred = torch.autograd.grad(phi, x_norm_grad, grad_outputs=torch.ones_like(phi),
                                                         create_graph=False, retain_graph=False, allow_unused=True)[0]
                            v_pred = torch.autograd.grad(phi, y_norm_grad, grad_outputs=torch.ones_like(phi),
                                                         create_graph=False, retain_graph=False, allow_unused=True)[0]
                            if u_pred is None:
                                u_pred = torch.zeros_like(x_norm_grad)
                            if v_pred is None:
                                v_pred = torch.zeros_like(y_norm_grad)
                            p_pred = torch.zeros_like(phi)
                    else:
                        input_tensor = self.loss_computer.build_input_tensor(x_norm, y_norm, sdf_norm, bt)
                        out = self.model(input_tensor)
                        u_pred = out[:, 0:1]
                        v_pred = out[:, 1:2]
                        p_pred = out[:, 2:3]
                    
                    u_pred = u_pred.cpu().numpy().flatten()
                    v_pred = v_pred.cpu().numpy().flatten()
                    p_pred = p_pred.cpu().numpy().flatten()

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