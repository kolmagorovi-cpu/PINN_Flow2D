"""
Обработка данных для Pipe-Bend PINN.

Исправленная версия:
- Добавлен device параметр
- Маски границ заполняются корректно
- Добавлены get_state() / load_state()

Автор: Адаптировано из Airfoil PINN
Версия: 1.1
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class PINNDataset(Dataset):
    """Dataset для точек коллокации."""

    def __init__(self, points_file: str) -> None:
        data = np.load(points_file, allow_pickle=True)
        if data.dtype.names is None:
            raise ValueError("Файл точек должен быть структурированным массивом")

        required = ['x_norm', 'y_norm', 'sdf_norm', 'p_norm', 'u_norm', 'v_norm']
        for fld in required:
            if fld not in data.dtype.names:
                raise ValueError(f"В файле отсутствует поле {fld}")

        self.x = torch.tensor(np.ascontiguousarray(data['x_norm']), dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(np.ascontiguousarray(data['y_norm']), dtype=torch.float32).view(-1, 1)
        self.sdf = torch.tensor(np.ascontiguousarray(data['sdf_norm']), dtype=torch.float32).view(-1, 1)

        self.p_target = torch.tensor(np.ascontiguousarray(data['p_norm']), dtype=torch.float32).view(-1, 1)
        self.u_target = torch.tensor(np.ascontiguousarray(data['u_norm']), dtype=torch.float32).view(-1, 1)
        self.v_target = torch.tensor(np.ascontiguousarray(data['v_norm']), dtype=torch.float32).view(-1, 1)

        self.boundary_types = None
        self.boundary_type_names = []
        self.type_to_idx = {}

        # Маски: какая точка имеет какое ГУ
        self.mask_u_fixed = torch.zeros(len(self.x), dtype=torch.bool)
        self.mask_v_fixed = torch.zeros(len(self.x), dtype=torch.bool)
        self.mask_p_fixed = torch.zeros(len(self.x), dtype=torch.bool)

        # Целевые значения BC (заполняются из данных)
        self.u_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)
        self.v_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)
        self.p_bc = torch.zeros(len(self.x), dtype=torch.float32).view(-1, 1)

        # Заполняем из данных
        self.u_bc[:] = torch.tensor(np.ascontiguousarray(data['u_norm']), dtype=torch.float32).view(-1, 1)
        self.v_bc[:] = torch.tensor(np.ascontiguousarray(data['v_norm']), dtype=torch.float32).view(-1, 1)
        self.p_bc[:] = torch.tensor(np.ascontiguousarray(data['p_norm']), dtype=torch.float32).view(-1, 1)

        # Обработка boundary_type
        if 'boundary_type' in data.dtype.names:
            unique_raw = np.unique(data['boundary_type'])
            self.boundary_type_names = [str(t).lower() for t in unique_raw]
            self.type_to_idx = {t: i for i, t in enumerate(self.boundary_type_names)}

            # One-hot encoding
            indices = []
            for bt in data['boundary_type']:
                bt_lower = str(bt).lower()
                indices.append(self.type_to_idx.get(bt_lower, 0))

            one_hot = np.zeros((len(indices), len(self.boundary_type_names)), dtype=np.float32)
            one_hot[np.arange(len(indices)), indices] = 1.0
            self.boundary_types = torch.tensor(np.ascontiguousarray(one_hot), dtype=torch.float32)

            # Заполняем маски
            for i in range(len(data)):
                bt_str = str(data['boundary_type'][i]).lower()
                if bt_str == 'inlet':
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # u_bc/v_bc уже заданы из данных
                elif bt_str == 'outlet':
                    self.mask_p_fixed[i] = True
                    # p_bc уже задана из данных
                elif bt_str == 'wall':
                    self.mask_u_fixed[i] = True
                    self.mask_v_fixed[i] = True
                    # Для стенки u_bc=0, v_bc=0, это уже в данных
                # Для interior: маски остаются False

        self.n_points = len(self.x)
        logger.info(f"Типы границ в данных: {self.boundary_type_names}")

    def set_inlet_values(self, u_norm: float, v_norm: float) -> None:
        """Устанавливает значения скорости на inlet."""
        if self.boundary_types is None:
            return
        inlet_idx = self.type_to_idx.get('inlet')
        if inlet_idx is None:
            return
        if inlet_idx >= self.boundary_types.shape[1]:
            return
        mask = self.boundary_types[:, inlet_idx] > 0.5
        self.u_bc[mask] = u_norm
        self.v_bc[mask] = v_norm

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> Tuple:
        result = (idx, self.x[idx], self.y[idx], self.sdf[idx])
        if self.boundary_types is not None:
            result += (self.boundary_types[idx],)
        else:
            result += (torch.zeros(0),)
        result += (self.mask_u_fixed[idx].view(1),
                   self.mask_v_fixed[idx].view(1),
                   self.mask_p_fixed[idx].view(1),
                   self.u_bc[idx].view(1),
                   self.v_bc[idx].view(1),
                   self.p_bc[idx].view(1))
        return result

    def extend(self, indices: torch.Tensor) -> 'PINNDataset':
        """Расширяет датасет указанными индексами (для RAR)."""
        new_dataset = PINNDataset.__new__(PINNDataset)
        for attr in ['x', 'y', 'sdf', 'p_target', 'u_target', 'v_target',
                     'mask_u_fixed', 'mask_v_fixed', 'mask_p_fixed',
                     'u_bc', 'v_bc', 'p_bc', 'boundary_types']:
            old = getattr(self, attr)
            if old is not None:
                new = torch.cat([old, old[indices]], dim=0)
                setattr(new_dataset, attr, new)
            else:
                setattr(new_dataset, attr, None)
        new_dataset.boundary_type_names = self.boundary_type_names
        new_dataset.type_to_idx = self.type_to_idx
        new_dataset.n_points = len(new_dataset.x)
        return new_dataset

    def get_state(self) -> Dict:
        """Возвращает состояние датасета для чекпоинта."""
        return {
            'x': self.x.cpu().numpy(),
            'y': self.y.cpu().numpy(),
            'sdf': self.sdf.cpu().numpy(),
            'boundary_types': self.boundary_types.cpu().numpy() if self.boundary_types is not None else None,
            'mask_u_fixed': self.mask_u_fixed.cpu().numpy(),
            'mask_v_fixed': self.mask_v_fixed.cpu().numpy(),
            'mask_p_fixed': self.mask_p_fixed.cpu().numpy(),
            'u_bc': self.u_bc.cpu().numpy(),
            'v_bc': self.v_bc.cpu().numpy(),
            'p_bc': self.p_bc.cpu().numpy(),
            'boundary_type_names': self.boundary_type_names,
            'type_to_idx': self.type_to_idx,
            'n_points': self.n_points,
        }


class DataHandler:
    """Загружает данные и метаданные, предоставляет DataLoader."""

    def __init__(self, points_file: str, batch_size: int, shuffle: bool = True,
                 device: Optional[torch.device] = None) -> None:
        self.points_file = points_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or torch.device('cpu')
        self.dataset = PINNDataset(points_file)
        self.load_metadata()

        # Переносим тензоры датасета на device (если GPU)
        if self.device.type == 'cuda':
            self.dataset.x = self.dataset.x.to(self.device)
            self.dataset.y = self.dataset.y.to(self.device)
            self.dataset.sdf = self.dataset.sdf.to(self.device)
            self.dataset.u_target = self.dataset.u_target.to(self.device)
            self.dataset.v_target = self.dataset.v_target.to(self.device)
            self.dataset.p_target = self.dataset.p_target.to(self.device)
            self.dataset.mask_u_fixed = self.dataset.mask_u_fixed.to(self.device)
            self.dataset.mask_v_fixed = self.dataset.mask_v_fixed.to(self.device)
            self.dataset.mask_p_fixed = self.dataset.mask_p_fixed.to(self.device)
            self.dataset.u_bc = self.dataset.u_bc.to(self.device)
            self.dataset.v_bc = self.dataset.v_bc.to(self.device)
            self.dataset.p_bc = self.dataset.p_bc.to(self.device)
            if self.dataset.boundary_types is not None:
                self.dataset.boundary_types = self.dataset.boundary_types.to(self.device)

        self.dataset.set_inlet_values(self.inlet_u_norm, self.inlet_v_norm)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def load_metadata(self) -> None:
        meta_path = self.points_file.replace('.npy', '_metadata.npy')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Файл метаданных {meta_path} не найден")
        meta = np.load(meta_path, allow_pickle=True).item()
        self.chord = meta.get('chord', meta.get('pipe_width', 1.0))
        self.U_inf = meta.get('U_inf', 1.0)
        self.p_ref = meta.get('p_ref', 0.0)
        self.rho = meta.get('rho', 1.0)
        self.mu = meta.get('mu', 1.0)
        self.Re = meta.get('Re', 1.0)
        self.inlet_u_norm = meta.get('inlet_u_norm', 1.0)
        self.inlet_v_norm = meta.get('inlet_v_norm', 0.0)
        self.x_min = self.dataset.x.min().item()
        self.x_max = self.dataset.x.max().item()
        self.y_min = self.dataset.y.min().item()
        self.y_max = self.dataset.y.max().item()
        self.pipe_width = meta.get('pipe_width', meta.get('chord', 1.0))
        logger.info(f"Метаданные: U_inf={self.U_inf:.4f} м/с, Re={self.Re:.2e}")
        logger.info(f"Вход: u_norm={self.inlet_u_norm:.3f}, v_norm={self.inlet_v_norm:.3f}")

    def get_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_gpu_batches(self, batch_size: Optional[int] = None, shuffle: bool = True):
        """
        Генерирует батчи напрямую на GPU, минуя DataLoader.
        Это устраняет накладные расходы на копирование CPU→GPU.
        """
        if batch_size is None:
            batch_size = self.batch_size
        ds = self.dataset
        n = ds.n_points
        indices = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, batch_size):
            idx = indices[i:min(i + batch_size, n)]
            cpu_idx = idx.cpu()
            bt = ds.boundary_types[idx] if ds.boundary_types is not None else torch.zeros(len(idx), 0, device=self.device)
            yield (cpu_idx, ds.x[idx], ds.y[idx], ds.sdf[idx], bt,
                   ds.mask_u_fixed[idx].view(-1, 1),
                   ds.mask_v_fixed[idx].view(-1, 1),
                   ds.mask_p_fixed[idx].view(-1, 1),
                   ds.u_bc[idx], ds.v_bc[idx], ds.p_bc[idx])

    @property
    def n_points(self) -> int:
        return self.dataset.n_points

    @property
    def boundary_type_names(self) -> List[str]:
        return self.dataset.boundary_type_names

    def add_points(self, indices: torch.Tensor) -> None:
        """Добавляет точки в датасет (для RAR)."""
        self.dataset = self.dataset.extend(indices)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        logger.info(f"Датасет расширен: добавлено {len(indices)} точек, всего {self.n_points}")

    def get_points_by_type(self, type_name: str) -> Dict[str, torch.Tensor]:
        if type_name not in self.boundary_type_names:
            raise ValueError(f"Тип границы '{type_name}' не найден")
        if self.dataset.boundary_types is None:
            return {'x': torch.empty(0), 'y': torch.empty(0), 'sdf': torch.empty(0),
                    'u_target': torch.empty(0), 'v_target': torch.empty(0), 'p_target': torch.empty(0),
                    'mask_u': torch.empty(0), 'mask_v': torch.empty(0), 'mask_p': torch.empty(0),
                    'u_bc': torch.empty(0), 'v_bc': torch.empty(0), 'p_bc': torch.empty(0)}
        idx = self.boundary_type_names.index(type_name)
        if idx >= self.dataset.boundary_types.shape[1]:
            return {'x': torch.empty(0), 'y': torch.empty(0)}
        mask = self.dataset.boundary_types[:, idx] > 0.5
        return {
            'x': self.dataset.x[mask], 'y': self.dataset.y[mask],
            'u_target': self.dataset.u_target[mask], 'v_target': self.dataset.v_target[mask],
            'p_target': self.dataset.p_target[mask], 'sdf': self.dataset.sdf[mask],
            'mask_u': self.dataset.mask_u_fixed[mask], 'mask_v': self.dataset.mask_v_fixed[mask],
            'mask_p': self.dataset.mask_p_fixed[mask],
            'u_bc': self.dataset.u_bc[mask], 'v_bc': self.dataset.v_bc[mask], 'p_bc': self.dataset.p_bc[mask],
        }

    def interpolate_profile(self, x_target: float, u_vals: Optional[torch.Tensor] = None,
                            y_grid: Optional[torch.Tensor] = None, num_points: int = 100,
                            y_max_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        tol = 0.05 * (self.x_max - self.x_min)
        mask_x = torch.abs(self.dataset.x.flatten() - x_target) < tol
        y_limit = y_max_ratio * self.y_max
        mask_y = self.dataset.y.flatten() < y_limit
        mask = mask_x & mask_y
        y_vals = self.dataset.y[mask].cpu().numpy().flatten()
        if u_vals is not None:
            u_vals_np = u_vals[mask].cpu().numpy().flatten()
        else:
            u_vals_np = self.dataset.u_target[mask].cpu().numpy().flatten()
        sort_idx = np.argsort(y_vals)
        y_sorted = y_vals[sort_idx]
        u_sorted = u_vals_np[sort_idx]
        if y_grid is None:
            y_grid = np.linspace(self.y_min, y_limit, num_points)
        u_interp = np.interp(y_grid, y_sorted, u_sorted, left=u_sorted[0], right=u_sorted[-1])
        return y_grid, u_interp

    def get_state(self) -> Dict:
        """Возвращает состояние DataHandler для чекпоинта."""
        return {
            'dataset_state': self.dataset.get_state(),
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }

    def load_state(self, state: Dict) -> None:
        """Восстанавливает состояние DataHandler из чекпоинта."""
        if 'dataset_state' in state:
            # Воссоздаём датасет из сохранённого состояния
            ds_state = state['dataset_state']
            new_dataset = PINNDataset.__new__(PINNDataset)
            for attr in ['x', 'y', 'sdf', 'boundary_types', 'mask_u_fixed',
                         'mask_v_fixed', 'mask_p_fixed', 'u_bc', 'v_bc', 'p_bc']:
                val = ds_state.get(attr)
                if val is not None:
                    setattr(new_dataset, attr, torch.from_numpy(val))
                else:
                    setattr(new_dataset, attr, None)
            new_dataset.boundary_type_names = ds_state.get('boundary_type_names', [])
            new_dataset.type_to_idx = ds_state.get('type_to_idx', {})
            new_dataset.n_points = ds_state.get('n_points', len(new_dataset.x))
            self.dataset = new_dataset
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            logger.info(f"DataHandler восстановлен: {self.n_points} точек")