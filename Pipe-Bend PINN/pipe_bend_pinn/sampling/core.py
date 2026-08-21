#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дискретизация Pipe-Bend PINN: генерация точек коллокации.

Классифицирует точки по типам границ (inlet, outlet, wall) и сохраняет
структурированный массив точек в workspace/sampling/collocation/.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import numpy as np
import matplotlib.path as mplPath
from shapely.geometry import LineString

from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT,
    GEOMETRY_DOMAINS_DIR,
    SAMPLING_COLLOCATION_DIR,
)


def load_domain(filename: str = "pipe_bend_domain.npy") -> np.ndarray:
    """Загружает контур расчётной области из workspace/geometry/domains/."""
    domain_path = os.path.join(GEOMETRY_DOMAINS_DIR, filename)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Файл геометрии не найден: {domain_path}")
    domain = np.load(domain_path)
    if domain.ndim == 1 and domain.dtype.names is not None:
        domain = np.column_stack((domain["x"], domain["y"]))
    return domain


def generate_collocation(
    step_size: float = 0.02,
    num_boundary: int = 900,
    n_inlet: int = 100,
    n_outlet: int = 100,
    bend_radius: float = 1.0,
    pipe_width: float = 0.5,
    domain_filename: str = "pipe_bend_domain.npy",
    output_name: str = "collocation_points",
) -> dict:
    """
    Генерирует точки коллокации (interior + boundary) для области отвода.

    Inlet/outlet грани семплируются явно (n_inlet / n_outlet точек),
    остальной периметр — равномерно по длине (num_boundary точек, классифицируются).
    Это даёт плотную дискретизацию входных/выходных сечений (важно для BC-потерь).

    Returns:
        Словарь со статистикой: num_points, num_interior, num_boundary, counts.
    """
    domain = load_domain(domain_filename)
    R_inner = bend_radius - pipe_width / 2

    x_min, y_min = np.min(domain, axis=0)
    x_max, y_max = np.max(domain, axis=0)

    # Внутренние точки (регулярная сетка внутри контура)
    path = mplPath.Path(domain)
    x_coords = np.arange(x_min, x_max, step_size)
    y_coords = np.arange(y_min, y_max, step_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.ravel(), Y.ravel()))
    inside = path.contains_points(points)
    interior_points = points[inside]

    boundary_points = []
    boundary_types = []
    tol = max(step_size, 1e-6)

    # --- Inlet: вертикальная грань x == x_min, y in [y_min, y_min+pipe_width]
    y_in_lo = y_min
    y_in_hi = y_min + pipe_width
    for i in range(n_inlet):
        yb = y_in_lo + (y_in_hi - y_in_lo) * (i / max(1, n_inlet - 1))
        boundary_points.append((x_min, float(yb)))
        boundary_types.append("inlet")

    # --- Outlet: горизонтальная грань y == y_max, x in [x_max-pipe_width, x_max]
    x_out_lo = x_max - pipe_width
    x_out_hi = x_max
    for i in range(n_outlet):
        xb = x_out_lo + (x_out_hi - x_out_lo) * (i / max(1, n_outlet - 1))
        boundary_points.append((float(xb), y_max))
        boundary_types.append("outlet")

    # --- Стены: равномерно по периметру, точки inlet/outlet отбрасываем
    line = LineString(domain)
    total_length = line.length
    for i in range(num_boundary):
        distance = i * total_length / num_boundary
        pt = line.interpolate(distance)
        xb, yb = pt.x, pt.y
        if abs(xb - x_min) < tol and yb < y_in_hi + tol:
            continue  # уже покрыто явными inlet-точками
        if abs(yb - y_max) < tol and xb > x_out_lo - tol:
            continue  # уже покрыто явными outlet-точками
        boundary_points.append((xb, yb))
        boundary_types.append("wall")

    all_points_np = np.array(interior_points.tolist() + boundary_points)
    all_types = ["interior"] * len(interior_points) + boundary_types

    dtype = [("id", "i4"), ("x", "f8"), ("y", "f8"), ("boundary_type", "U20"), ("on_boundary", "bool")]
    structured = np.zeros(len(all_points_np), dtype=dtype)
    structured["id"] = np.arange(len(all_points_np))
    structured["x"] = all_points_np[:, 0]
    structured["y"] = all_points_np[:, 1]
    structured["boundary_type"] = all_types
    structured["on_boundary"] = [t != "interior" for t in all_types]

    os.makedirs(SAMPLING_COLLOCATION_DIR, exist_ok=True)
    np.save(os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}.npy"), structured)

    metadata = {
        "num_points": len(structured),
        "num_interior": len(interior_points),
        "num_boundary": len(boundary_points),
        "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "chord": pipe_width,
        "pipe_width": pipe_width,
    }
    np.save(os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}_metadata.npy"), metadata)

    counts = dict(Counter(all_types))
    return {
        "num_points": len(structured),
        "num_interior": len(interior_points),
        "num_boundary": len(boundary_points),
        "counts": counts,
        "path": os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}.npy"),
    }


def S2DR() -> None:
    """Интерактивный запуск генерации точек коллокации."""
    print("\n" + "=" * 60)
    print("ДИСКРЕТИЗАЦИЯ (КОЛЛОКАЦИЯ)".center(60))
    print("=" * 60)
    try:
        step = float(input("Шаг сетки [0.05]: ").strip() or 0.05)
        num_bnd = int(input("Число граничных точек [200]: ").strip() or 200)
        stat = generate_collocation(step_size=step, num_boundary=num_bnd)
        print(f"\n✅ Точки коллокации созданы:")
        print(f"   Всего: {stat['num_points']}")
        print(f"   Interior: {stat['num_interior']}")
        print(f"   Boundary: {stat['num_boundary']}")
        for bt, cnt in stat["counts"].items():
            print(f"   {bt}: {cnt}")
        print(f"   Файл: {stat['path']}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    input("\nНажмите Enter для возврата...")
