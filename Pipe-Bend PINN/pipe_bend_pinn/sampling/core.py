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


def _resample_contour_uniform(domain: np.ndarray, ds: float) -> np.ndarray:
    """Равномерно передискретизирует замкнутый контур с шагом дуги ≈ ds."""
    closed = np.vstack([domain, domain[0:1]])
    seg = np.diff(closed, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(length)])
    total = float(cum[-1])
    n = max(2, int(np.ceil(total / max(ds, 1e-9))))
    s = np.linspace(0.0, total, n + 1)[:-1]
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = (s - cum[idx]) / np.maximum(length[idx], 1e-12)
    return closed[idx] + frac[:, None] * seg[idx]


def _contour_curvature(pts: np.ndarray) -> np.ndarray:
    """Кривизна 1/м в точках контура по окружности через 3 соседние точки."""
    ring = np.vstack([pts[-1:], pts, pts[0:1]])
    a, b, c = ring[:-2], ring[1:-1], ring[2:]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    area2 = np.abs(cross)
    k = np.zeros(len(pts))
    m = (area2 > 1e-14) & (ab > 1e-12) & (bc > 1e-12) & (ca > 1e-12)
    k[m] = 2.0 * area2[m] / (ab[m] * bc[m] * ca[m])
    return k


def _refine_interior_near_curved_walls(
    domain, step_size, pipe_width, interior_base, boundary_np,
    curvature_lo=0.25, curvature_hi=6.0, station_frac=0.5,
    offsets_frac=(0.5, 1.0, 1.75, 2.75, 4.0, 5.75), min_clearance_frac=0.35,
):
    """Геометрия-агностичное пристеночное сгущение у криволинейных стенок.

    Для любого замкнутого контура: передискретизация → локальная кривизна κ;
    «криволинейный» участок = κ·pipe_width ∈ [lo, hi] (прямые стенки κ≈0 и
    острые углы κ·W >> hi исключаются вместе с их малой окрестностью);
    из узлов участка по внутренней нормали добавляются interior-точки на
    расстояниях offsets_frac·step_size. Плотность вдоль стенки повышена
    автоматически (station_frac·step_size), поэтому изгиб сгущается сам,
    без ручного выделения подобластей. Возвращает (m,2) доп. точек.
    """
    if not np.isfinite(curvature_lo) or curvature_lo <= 0.0:
        return np.empty((0, 2))
    from scipy.spatial import cKDTree

    ds_res = max(step_size * station_frac, 0.004)
    contour = _resample_contour_uniform(domain, ds_res)
    kappa_w = _contour_curvature(contour) * pipe_width
    curved = (kappa_w >= curvature_lo) & (kappa_w <= curvature_hi)
    spikes = kappa_w > curvature_hi
    if not np.any(curved):
        return np.empty((0, 2))
    if np.any(spikes):
        from scipy.ndimage import binary_dilation

        margin = max(2, int(round(0.02 / max(ds_res, 1e-9))))
        curved = curved & ~binary_dilation(spikes, iterations=margin)
    station_ids = np.flatnonzero(curved)
    if not len(station_ids):
        return np.empty((0, 2))
    station_step = max(1, int(round(step_size * station_frac / max(ds_res, 1e-9))))
    station_ids = station_ids[::station_step]

    ring_dists = np.asarray([f * step_size for f in offsets_frac])
    contour_tree = cKDTree(contour)
    base_all = np.vstack([interior_base, boundary_np]) if len(boundary_np) else interior_base
    path = mplPath.Path(domain)

    cand = []
    for sid in station_ids:
        p = contour[sid]
        t = contour[(sid + 1) % len(contour)] - contour[(sid - 1) % len(contour)]
        tn = float(np.linalg.norm(t))
        if tn < 1e-12:
            continue
        nrm = np.array([-t[1], t[0]]) / tn
        for d_off in ring_dists:
            for q in (p + nrm * d_off, p - nrm * d_off):
                if path.contains_point(q):
                    cand.append(q)
                    break
    if not cand:
        return np.empty((0, 2))
    cand = np.asarray(cand)
    cand = cand[path.contains_points(cand)]
    if not len(cand):
        return np.empty((0, 2))
    d_wall, _ = contour_tree.query(cand, k=1)
    keep = d_wall >= min_clearance_frac * step_size
    cand, d_wall = cand[keep], d_wall[keep]
    if not len(cand):
        return np.empty((0, 2))

    # Дедупликация: сначала ближние к стенке; не ближе min_dist друг к другу.
    order = np.argsort(d_wall, kind="stable")
    cand = cand[order]
    accepted = []
    acc_tree = cKDTree(base_all) if len(base_all) else None
    min_dist = max(0.30 * min_clearance_frac * step_size, 1e-6)
    for q in cand:
        if acc_tree is not None:
            if acc_tree.query(q, k=1)[0] < min_dist:
                continue
        accepted.append(q)
        if len(accepted) % 128 == 0:
            acc_tree = cKDTree(np.vstack([base_all, np.asarray(accepted)]))
    return np.asarray(accepted) if accepted else np.empty((0, 2))



def generate_collocation(
    step_size: float = 0.02,
    num_boundary: int = 900,
    n_inlet: int = 100,
    n_outlet: int = 100,
    bend_radius: float = 1.0,
    pipe_width: float = 0.5,
    domain_filename: str = "pipe_bend_domain.npy",
    output_name: str = "collocation_points",
    wall_refine: bool = False,
    wall_refine_curvature_lo: float = 0.25,
    wall_refine_curvature_hi: float = 6.0,
    wall_refine_station_frac: float = 0.5,
    wall_refine_offsets_frac=(0.5, 1.0, 1.75, 2.75, 4.0, 5.75),
    wall_refine_min_clearance_frac: float = 0.35,
) -> dict:
    """
    Генерирует точки коллокации (interior + boundary) для области отвода.

    Inlet/outlet грани семплируются явно (n_inlet / n_outlet точек),
    остальной периметр — равномерно по длине (num_boundary точек, классифицируются).
    Это даёт плотную дискретизацию входных/выходных сечений (важно для BC-потерь).
    При wall_refine=True автоматически (по кривизне стенок) сгущается интерьер
    вдоль криволинейных участков контура — без ручного выделения подобластей.

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

    # Пристеночное сгущение у криволинейных стенок (геометрия-агностично)
    if wall_refine:
        extra_interior = _refine_interior_near_curved_walls(
            domain, step_size, pipe_width,
            np.asarray(interior_points, dtype=np.float64),
            np.asarray(boundary_points, dtype=np.float64),
            curvature_lo=wall_refine_curvature_lo,
            curvature_hi=wall_refine_curvature_hi,
            station_frac=wall_refine_station_frac,
            offsets_frac=wall_refine_offsets_frac,
            min_clearance_frac=wall_refine_min_clearance_frac,
        )
        if len(extra_interior):
            interior_points = np.vstack([interior_points, extra_interior])
        print(f"   Пристеночное сгущение у криволинейных стенок: +{len(extra_interior)} interior-точек")

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
