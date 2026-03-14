#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для работы с эталонными решениями (reference solutions) в проекте PINN Learning.
Поддерживает загрузку данных из файлов (например, экспорт из Ansys Fluent),
интерполяцию на произвольные координаты и вычисление ошибок L2.

Автор: Колмагоров И.Д.
Версия: 3.260226 (диагностическая)
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Tuple

# Попытка импорта scipy для интерполяции
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy не установлен. Интерполяция эталонных решений будет недоступна.")

logger = logging.getLogger(__name__)


def load_reference(filepath: str,
                   delimiter: str = ',',
                   skiprows: int = 1,
                   columns: Tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)) -> Dict[str, np.ndarray]:
    """
    Загружает эталонное решение из текстового файла (CSV или подобного).

    Ожидается, что файл содержит колонки: x, y, u, v, p (порядок может быть задан параметром columns).

    Args:
        filepath: путь к файлу
        delimiter: разделитель колонок
        skiprows: количество строк для пропуска (например, заголовок)
        columns: кортеж из пяти индексов колонок для (x, y, u, v, p)

    Returns:
        словарь с ключами 'x', 'y', 'u', 'v', 'p' и массивами numpy
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл эталонного решения не найден: {filepath}")

    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)
    except Exception as e:
        # Если не получается загрузить как числа, пробуем прочитать через genfromtxt (более толерантно)
        data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=skiprows, filling_values=np.nan)
        if np.any(np.isnan(data)):
            logger.warning("В данных обнаружены пропуски (NaN). Они будут проигнорированы.")
            data = data[~np.isnan(data).any(axis=1)]

    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError("Файл должен содержать как минимум 5 колонок (x, y, u, v, p)")

    x_col, y_col, u_col, v_col, p_col = columns

    ref = {
        'x': data[:, x_col].copy(),
        'y': data[:, y_col].copy(),
        'u': data[:, u_col].copy(),
        'v': data[:, v_col].copy(),
        'p': data[:, p_col].copy()
    }

    logger.info(f"Эталонное решение загружено из {filepath}: {len(ref['x'])} точек")
    return ref


def interpolate_reference(ref_data: Dict[str, np.ndarray],
                          points_x: np.ndarray,
                          points_y: np.ndarray,
                          method: str = 'linear') -> Dict[str, np.ndarray]:
    """
    Интерполирует эталонные поля на заданные координаты points_x, points_y.

    Args:
        ref_data: словарь с ключами 'x', 'y', 'u', 'v', 'p' (массивы)
        points_x, points_y: координаты целевых точек (одномерные массивы одинаковой длины)
        method: метод интерполяции ('linear', 'nearest', 'cubic') – передаётся в scipy.interpolate.griddata

    Returns:
        словарь с интерполированными 'u', 'v', 'p' (массивы той же длины, что и points_x)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy не установлен. Невозможно выполнить интерполяцию.")

    # Проверка размеров
    if points_x.shape != points_y.shape:
        raise ValueError("points_x и points_y должны иметь одинаковую форму")

    # Формируем точки-источники
    src_points = np.column_stack((ref_data['x'], ref_data['y']))

    # Интерполяция
    u_interp = griddata(src_points, ref_data['u'], (points_x, points_y), method=method)
    v_interp = griddata(src_points, ref_data['v'], (points_x, points_y), method=method)
    p_interp = griddata(src_points, ref_data['p'], (points_x, points_y), method=method)

    # Проверка на NaN (точки вне области источника)
    mask = np.isnan(u_interp)
    if np.any(mask):
        logger.warning(f"Обнаружено {np.sum(mask)} точек за пределами области эталонных данных. "
                       "Для них используется метод 'nearest' для заполнения.")
        # Заполняем пропуски методом ближайшего соседа
        u_interp[mask] = griddata(src_points, ref_data['u'], (points_x[mask], points_y[mask]), method='nearest')
        v_interp[mask] = griddata(src_points, ref_data['v'], (points_x[mask], points_y[mask]), method='nearest')
        p_interp[mask] = griddata(src_points, ref_data['p'], (points_x[mask], points_y[mask]), method='nearest')

    return {'u': u_interp, 'v': v_interp, 'p': p_interp}


def compute_l2_errors(pred_u: np.ndarray,
                      pred_v: np.ndarray,
                      pred_p: np.ndarray,
                      ref_u: np.ndarray,
                      ref_v: np.ndarray,
                      ref_p: np.ndarray,
                      norm: str = 'relative') -> Dict[str, float]:
    """
    Вычисляет L2-ошибки между предсказанными и эталонными полями.

    Args:
        pred_u, pred_v, pred_p: предсказанные поля (массивы)
        ref_u, ref_v, ref_p: эталонные поля (массивы той же формы)
        norm: 'relative' – относительная ошибка (норма разности / норма эталона),
              'absolute' – абсолютная норма разности

    Returns:
        словарь с ошибками для u, v, p и средней по всем
    """
    # Проверка форм
    if (pred_u.shape != ref_u.shape or
        pred_v.shape != ref_v.shape or
        pred_p.shape != ref_p.shape):
        raise ValueError("Формы предсказанных и эталонных массивов должны совпадать")

    # Преобразуем в плоские массивы, если нужно
    pred_u_flat = pred_u.flatten()
    pred_v_flat = pred_v.flatten()
    pred_p_flat = pred_p.flatten()
    ref_u_flat = ref_u.flatten()
    ref_v_flat = ref_v.flatten()
    ref_p_flat = ref_p.flatten()

    # Вычисляем нормы
    diff_u = pred_u_flat - ref_u_flat
    diff_v = pred_v_flat - ref_v_flat
    diff_p = pred_p_flat - ref_p_flat

    l2_u = np.sqrt(np.mean(diff_u ** 2))
    l2_v = np.sqrt(np.mean(diff_v ** 2))
    l2_p = np.sqrt(np.mean(diff_p ** 2))

    if norm == 'relative':
        norm_u = np.sqrt(np.mean(ref_u_flat ** 2)) + 1e-12
        norm_v = np.sqrt(np.mean(ref_v_flat ** 2)) + 1e-12
        norm_p = np.sqrt(np.mean(ref_p_flat ** 2)) + 1e-12
        l2_u /= norm_u
        l2_v /= norm_v
        l2_p /= norm_p

    mean_error = (l2_u + l2_v + l2_p) / 3.0

    return {
        'u': l2_u,
        'v': l2_v,
        'p': l2_p,
        'mean': mean_error
    }


def save_reference_as_npy(ref_data: Dict[str, np.ndarray], filepath: str) -> None:
    """
    Сохраняет эталонные данные в формате .npy (структурированный массив).
    """
    dtype = [('x', 'f8'), ('y', 'f8'), ('u', 'f8'), ('v', 'f8'), ('p', 'f8')]
    arr = np.zeros(len(ref_data['x']), dtype=dtype)
    arr['x'] = ref_data['x']
    arr['y'] = ref_data['y']
    arr['u'] = ref_data['u']
    arr['v'] = ref_data['v']
    arr['p'] = ref_data['p']
    np.save(filepath, arr)
    logger.info(f"Эталонное решение сохранено в {filepath}")


def load_reference_from_npy(filepath: str) -> Dict[str, np.ndarray]:
    """
    Загружает эталонные данные из .npy (структурированный массив).
    """
    arr = np.load(filepath, allow_pickle=True)
    ref = {
        'x': arr['x'],
        'y': arr['y'],
        'u': arr['u'],
        'v': arr['v'],
        'p': arr['p']
    }
    logger.info(f"Эталонное решение загружено из {filepath}")
    return ref