from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Сбор метрик для Pipe-Bend PINN.

Исправленная версия:
- save_final_fields, collect_epoch_metrics, get_mass_flow_metrics
- Динамические CSV-колонки на основе boundary_type_names

Автор: Адаптировано из Airfoil PINN
Версия: 1.1
"""

import os
import csv
import logging
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Сбор и логирование метрик обучения."""

    def __init__(self, model, data_handler, loss_computer, config: Dict,
                 device, dtype, logs_dir: str, config_name: str,
                 reference_data=None):
        self.model = model
        self.data_handler = data_handler
        self.loss_computer = loss_computer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.logs_dir = logs_dir
        self.config_name = config_name
        self.reference_data = reference_data
        self.history = defaultdict(list)

        os.makedirs(logs_dir, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Инициализирует CSV-файл с динамическими заголовками."""
        self.csv_path = os.path.join(self.logs_dir, "metrics.csv")
        # Динамические колонки на основе boundary_type_names
        bc_columns = self.data_handler.boundary_type_names
        self.csv_columns = ['epoch', 'loss', 'cont', 'mom_x', 'mom_y'] + bc_columns + ['wall_speed', 'pde_max']
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_columns)

    def update_history(self, loss, pde_raw, bc_comp, integral_loss,
                       wall_speed, pde_max, pde_rate, grad_norm,
                       extra_diag: Optional[Dict] = None):
        """Обновляет историю метрик."""
        self.history['loss'].append(loss)
        for k, v in pde_raw.items():
            key = f'pde_{k}'
            self.history.setdefault(key, []).append(v)
        for k, v in bc_comp.items():
            key = f'bc_{k}'
            self.history.setdefault(key, []).append(v)
        self.history.setdefault('wall_speed', []).append(wall_speed)
        self.history.setdefault('pde_max', []).append(pde_max)
        self.history.setdefault('grad_norm', []).append(grad_norm)

        if extra_diag:
            for k, v in extra_diag.items():
                self.history.setdefault(k, []).append(v)

    def get_boundary_avg_velocities(self, epoch: int) -> Tuple[float, float]:
        """
        Возвращает средние скорости на границах через предсказание модели.
        Возвращает (avg_u_in, avg_u_out).
        """
        try:
            # Предсказываем поля для inlet и outlet
            dataset = self.data_handler.dataset
            if dataset.boundary_types is None:
                return 0.0, 0.0

            type_names = self.data_handler.boundary_type_names
            avg_u_in = 0.0
            avg_u_out = 0.0

            # Inlet
            if 'inlet' in type_names:
                inlet_idx = type_names.index('inlet')
                inlet_mask = dataset.boundary_types[:, inlet_idx] > 0.5
                if inlet_mask.any():
                    with torch.no_grad():
                        x = dataset.x[inlet_mask].to(self.device, dtype=self.dtype)
                        y = dataset.y[inlet_mask].to(self.device, dtype=self.dtype)
                        sdf = dataset.sdf[inlet_mask].to(self.device, dtype=self.dtype)
                        bt = dataset.boundary_types[inlet_mask].to(self.device, dtype=self.dtype)
                        inp = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                        out = self.model(inp)
                        avg_u_in = out[:, 0].mean().item()

            # Outlet
            if 'outlet' in type_names:
                outlet_idx = type_names.index('outlet')
                outlet_mask = dataset.boundary_types[:, outlet_idx] > 0.5
                if outlet_mask.any():
                    with torch.no_grad():
                        x = dataset.x[outlet_mask].to(self.device, dtype=self.dtype)
                        y = dataset.y[outlet_mask].to(self.device, dtype=self.dtype)
                        sdf = dataset.sdf[outlet_mask].to(self.device, dtype=self.dtype)
                        bt = dataset.boundary_types[outlet_mask].to(self.device, dtype=self.dtype)
                        inp = self.loss_computer.build_input_tensor(x, y, sdf, bt)
                        out = self.model(inp)
                        avg_u_out = out[:, 1].mean().item()  # v компонента на выходе

            return avg_u_in, avg_u_out
        except Exception as e:
            logger.warning(f"get_boundary_avg_velocities: {e}")
            return 0.0, 0.0

    def get_mass_flow_metrics(self, epoch: int, interval: int = 1) -> Dict:
        """Вычисляет метрики баланса масс."""
        if interval <= 0 or epoch % interval != 0:
            return {}
        try:
            avg_u_in, avg_u_out = self.get_boundary_avg_velocities(epoch)
            return {
                'avg_u_in': avg_u_in,
                'avg_u_out': avg_u_out,
                'mass_balance_error': abs(avg_u_in - avg_u_out),
            }
        except Exception:
            return {}

    def print_epoch_table(self, epoch, epochs, loss, pde_raw, bc_comp,
                           integral_loss, pde_max, epoch_time, total_time):
        """Выводит таблицу метрик за эпоху."""
        print(f"\n{'='*80}")
        print(f"Эпоха {epoch}/{epochs} | Время: {epoch_time:.2f}с | Всего: {total_time:.2f}с")
        print(f"{'='*80}")
        print(f"Loss: {loss:.6e}")
        print(f"PDE: cont={pde_raw.get('cont',0):.2e}, mom_x={pde_raw.get('mom_x',0):.2e}, mom_y={pde_raw.get('mom_y',0):.2e}")
        bc_str = ", ".join(f"{k}={v:.2e}" for k, v in bc_comp.items())
        print(f"BC: {bc_str}")
        print(f"Max PDE residual: {pde_max:.2e}")
        print(f"{'='*80}")

    def save_log(self, message: str):
        """Сохраняет сообщение в лог-файл."""
        log_path = os.path.join(self.logs_dir, "training.log")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def add_row_to_csv(self, epoch, loss, pde_raw, bc_comp, wall_speed, pde_max):
        """Добавляет строку в CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, loss,
                   pde_raw.get('cont', 0.0),
                   pde_raw.get('mom_x', 0.0),
                   pde_raw.get('mom_y', 0.0)]
            for bt in self.data_handler.boundary_type_names:
                row.append(bc_comp.get(bt, 0.0))
            row.extend([wall_speed, pde_max])
            writer.writerow(row)

    def close(self):
        """Закрывает сборщик метрик."""
        pass