from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль для работы с эталонными решениями.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Tuple

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy не установлен. Интерполяция эталонных решений недоступна.")

logger = logging.getLogger(__name__)


def load_reference(filepath: str, delimiter: str = ',', skiprows: int = 1,
                   columns: Tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)) -> Dict[str, np.ndarray]:
    """Загружает эталонное решение из текстового файла."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл эталонного решения не найден: {filepath}")
    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)
    except Exception:
        data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=skiprows, filling_values=np.nan)
        if np.any(np.isnan(data)):
            data = data[~np.isnan(data).any(axis=1)]
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError("Файл должен содержать как минимум 5 колонок (x, y, u, v, p)")
    x_col, y_col, u_col, v_col, p_col = columns
    return {
        'x': data[:, x_col].copy(), 'y': data[:, y_col].copy(),
        'u': data[:, u_col].copy(), 'v': data[:, v_col].copy(), 'p': data[:, p_col].copy()
    }


def interpolate_reference(ref_data: Dict[str, np.ndarray], points_x: np.ndarray,
                          points_y: np.ndarray, method: str = 'linear') -> Dict[str, np.ndarray]:
    """Интерполирует эталонные поля на заданные координаты."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy не установлен")
    src_points = np.column_stack((ref_data['x'], ref_data['y']))
    u_interp = griddata(src_points, ref_data['u'], (points_x, points_y), method=method)
    v_interp = griddata(src_points, ref_data['v'], (points_x, points_y), method=method)
    p_interp = griddata(src_points, ref_data['p'], (points_x, points_y), method=method)
    mask = np.isnan(u_interp)
    if np.any(mask):
        u_interp[mask] = griddata(src_points, ref_data['u'], (points_x[mask], points_y[mask]), method='nearest')
        v_interp[mask] = griddata(src_points, ref_data['v'], (points_x[mask], points_y[mask]), method='nearest')
        p_interp[mask] = griddata(src_points, ref_data['p'], (points_x[mask], points_y[mask]), method='nearest')
    return {'u': u_interp, 'v': v_interp, 'p': p_interp}


def compute_l2_errors(pred_u: np.ndarray, pred_v: np.ndarray, pred_p: np.ndarray,
                      ref_u: np.ndarray, ref_v: np.ndarray, ref_p: np.ndarray,
                      norm: str = 'relative') -> Dict[str, float]:
    """Вычисляет L2-ошибки между предсказанными и эталонными полями."""
    pred_u_flat = pred_u.flatten()
    pred_v_flat = pred_v.flatten()
    pred_p_flat = pred_p.flatten()
    ref_u_flat = ref_u.flatten()
    ref_v_flat = ref_v.flatten()
    ref_p_flat = ref_p.flatten()
    l2_u = np.sqrt(np.mean((pred_u_flat - ref_u_flat) ** 2))
    l2_v = np.sqrt(np.mean((pred_v_flat - ref_v_flat) ** 2))
    l2_p = np.sqrt(np.mean((pred_p_flat - ref_p_flat) ** 2))
    if norm == 'relative':
        l2_u /= (np.sqrt(np.mean(ref_u_flat ** 2)) + 1e-12)
        l2_v /= (np.sqrt(np.mean(ref_v_flat ** 2)) + 1e-12)
        l2_p /= (np.sqrt(np.mean(ref_p_flat ** 2)) + 1e-12)
    return {'u': l2_u, 'v': l2_v, 'p': l2_p, 'mean': (l2_u + l2_v + l2_p) / 3.0}