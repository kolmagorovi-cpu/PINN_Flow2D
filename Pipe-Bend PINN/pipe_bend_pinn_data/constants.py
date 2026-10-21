#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Константы и контракты данных пакета ``pipe_bend_pinn_data``.

Модуль не зависит от обучающего ядра ``pipe_bend_pinn`` и описывает только
формат CFD-экспорта, канонические имена колонок и dtype нормализованных точек.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SECTION_DATA",
    "SECTION_NAME",
    "COLUMN_ALIASES",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "DEFAULT_DELIMITER",
    "DATA_POINT_TYPE",
    "NORMALIZED_DTYPE",
    "UnsetColumnError",
]

# Маркеры секций экспорта Fluent/CFX.
SECTION_NAME = "[name]"
SECTION_DATA = "[data]"

# Каноническое имя -> допустимые написания в заголовке (после удаления единиц,
# приведения к нижнему регистру и нормализации пробелов).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "x": ("x", "x coord", "x coordinate"),
    "y": ("y", "y coord", "y coordinate"),
    "z": ("z", "z coord", "z coordinate"),
    "pressure": ("pressure", "p", "static pressure"),
    "velocity": ("velocity", "velocity magnitude", "speed"),
    "u": ("velocity u", "u", "u velocity", "velocity x", "x velocity"),
    "v": ("velocity v", "v", "v velocity", "velocity y", "y velocity"),
}

# Без этих колонок восстановить поле невозможно.
REQUIRED_COLUMNS: tuple[str, ...] = ("x", "y", "pressure", "u", "v")

# Колонки, которые читаются, но не обязательны.
OPTIONAL_COLUMNS: tuple[str, ...] = ("z", "velocity")

DEFAULT_DELIMITER = ","

# Метка типа точки, которую получают CFD-данные (в отличие от inlet/outlet/wall).
DATA_POINT_TYPE = "data"

# Структурированный массив нормализованных CFD-точек. Совместим по обязательным
# полям с ``pipe_bend_pinn/data/handler.py::PINNDataset`` (x_norm, y_norm,
# sdf_norm, p_norm, u_norm, v_norm) и дополнен параметризацией по Re.
NORMALIZED_DTYPE = np.dtype(
    [
        ("x_norm", "f8"),
        ("y_norm", "f8"),
        ("sdf_norm", "f8"),
        ("p_norm", "f8"),
        ("u_norm", "f8"),
        ("v_norm", "f8"),
        ("re", "f8"),
        ("re_norm", "f8"),
        ("U_in", "f8"),
        ("case_id", "i4"),
        ("boundary_type", "U20"),
        ("on_boundary", "bool"),
    ]
)


class UnsetColumnError(ValueError):
    """Заголовок CFD-экспорта не содержит обязательной колонки."""
