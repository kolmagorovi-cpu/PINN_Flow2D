#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика расстояний до стенки для сетки и этапа инициализации.

SDF в проекте используется как unsigned distance до WALL-точек. Этот модуль
даёт общий расчёт статистики, чтобы генератор сетки и Initializer показывали
одинаковую картину разрешения пристенной области.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


DEFAULT_SDF_OVER_CHORD_LEVELS: Tuple[float, ...] = (
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
)

DEFAULT_SDF_OVER_DELTA_LEVELS: Tuple[float, ...] = (
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
)

_PERCENTILES: Tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)

__all__ = [
    "SCIPY_AVAILABLE",
    "compute_nearest_wall_distances",
    "compute_sdf_summary",
    "estimate_boundary_layer_delta",
    "format_sdf_summary",
    "resolve_boundary_layer_model",
]


def estimate_boundary_layer_delta(
    chord: Optional[float],
    re: Optional[float],
    model: str = "laminar",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Грубая оценка толщины пограничного слоя по хорде.

    Для laminar используется delta_99/c ~= 5/sqrt(Re), для turbulent —
    delta_99/c ~= 0.37/Re^0.2. Модель должна совпадать с моделью normal-layer.
    """
    if chord is None or re is None:
        return None, None
    chord_f = float(chord)
    re_f = float(re)
    if chord_f <= 0.0 or re_f <= 0.0 or not np.isfinite(re_f):
        return None, None
    resolved_model = resolve_boundary_layer_model(re_f, model)
    if resolved_model == "turbulent":
        delta_over_chord = 0.37 / (re_f ** 0.2)
    else:
        delta_over_chord = 5.0 / np.sqrt(re_f)
    return chord_f * delta_over_chord, float(delta_over_chord)


def resolve_boundary_layer_model(re: float, model: str = "auto") -> str:
    """Resolve the flat-plate delta estimate model used by all diagnostics."""
    model_l = str(model or "auto").lower()
    if model_l == "auto":
        return "laminar" if float(re) < 5.0e5 else "turbulent"
    return "turbulent" if model_l == "turbulent" else "laminar"


def compute_nearest_wall_distances(
    query_points: np.ndarray,
    wall_points: np.ndarray,
) -> np.ndarray:
    """Возвращает расстояния от query_points до ближайшей WALL-точки."""
    query = _as_xy(query_points)
    wall = _as_xy(wall_points)
    if len(query) == 0 or len(wall) == 0:
        return np.empty(0, dtype=np.float64)

    if SCIPY_AVAILABLE:
        tree = cKDTree(wall)
        dist, _ = tree.query(query, k=1)
        return np.asarray(dist, dtype=np.float64)

    # Fallback без SciPy: блочная векторизация, чтобы не раздувать память.
    max_pairs = 2_000_000
    block_size = max(1, min(len(query), max_pairs // max(1, len(wall))))
    sdf = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), block_size):
        stop = min(start + block_size, len(query))
        block = query[start:stop]
        delta = block[:, None, :] - wall[None, :, :]
        dist_sq = np.sum(delta * delta, axis=2)
        sdf[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return sdf


def compute_sdf_summary(
    sdf_m: np.ndarray,
    *,
    chord: Optional[float] = None,
    delta: Optional[float] = None,
    sdf_over_chord_levels: Sequence[float] = DEFAULT_SDF_OVER_CHORD_LEVELS,
    sdf_over_delta_levels: Sequence[float] = DEFAULT_SDF_OVER_DELTA_LEVELS,
) -> Dict[str, object]:
    """Собирает сериализуемую статистику по массиву SDF в метрах."""
    sdf = np.asarray(sdf_m, dtype=np.float64).reshape(-1)
    sdf = sdf[np.isfinite(sdf)]
    summary: Dict[str, object] = {
        "count": int(len(sdf)),
        "stats_m": _stats(sdf),
    }

    chord_f = _positive_float(chord)
    if chord_f is not None and len(sdf):
        sdf_over_chord = sdf / chord_f
        summary["chord"] = chord_f
        summary["stats_over_chord"] = _stats(sdf_over_chord)
        summary["counts_over_chord"] = _threshold_counts(
            sdf_over_chord,
            sdf_over_chord_levels,
        )

    delta_f = _positive_float(delta)
    if delta_f is not None and len(sdf):
        sdf_over_delta = sdf / delta_f
        summary["delta"] = delta_f
        summary["stats_over_delta"] = _stats(sdf_over_delta)
        summary["counts_over_delta"] = _threshold_counts(
            sdf_over_delta,
            sdf_over_delta_levels,
        )

    return summary


def format_sdf_summary(
    summary: Dict[str, object],
    *,
    title: str = "SDF-статистика",
) -> List[str]:
    """Форматирует summary в строки для print/logger."""
    lines: List[str] = [title]
    count = int(summary.get("count", 0))
    lines.append(f"  Выборка: {count} interior-точек")
    if count == 0:
        lines.append("  Нет точек для расчёта SDF-статистики.")
        return lines

    stats_m = summary.get("stats_m")
    if isinstance(stats_m, dict):
        lines.extend(_format_stats(stats_m, "SDF [м]"))

    stats_c = summary.get("stats_over_chord")
    if isinstance(stats_c, dict):
        lines.extend(_format_stats(stats_c, "SDF/c"))

    counts_c = summary.get("counts_over_chord")
    if isinstance(counts_c, list):
        lines.append("  Количество точек по уровням SDF/c (накопительно):")
        lines.extend(_format_counts(counts_c, indent="    "))

    delta = summary.get("delta")
    stats_d = summary.get("stats_over_delta")
    if isinstance(delta, float):
        lines.append(f"  Оценка delta_99: {delta:.6g} м")
    if isinstance(stats_d, dict):
        lines.extend(_format_stats(stats_d, "SDF/delta"))

    counts_d = summary.get("counts_over_delta")
    if isinstance(counts_d, list):
        lines.append("  Количество точек по уровням SDF/delta (накопительно):")
        lines.extend(_format_counts(counts_d, indent="    "))

    return lines


def _as_xy(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, 2)
    return arr


def _positive_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value_f = float(value)
    if value_f <= 0.0 or not np.isfinite(value_f):
        return None
    return value_f


def _stats(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    stats = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    perc = np.percentile(values, _PERCENTILES)
    for p, value in zip(_PERCENTILES, perc):
        stats[f"p{p:02d}"] = float(value)
    return stats


def _threshold_counts(
    values: np.ndarray,
    levels: Sequence[float],
) -> List[Dict[str, float]]:
    n = int(len(values))
    result: List[Dict[str, float]] = []
    for level in levels:
        level_f = float(level)
        count = int(np.count_nonzero(values <= level_f))
        result.append({
            "level": level_f,
            "count": count,
            "fraction": float(count / n) if n else 0.0,
        })
    return result


def _format_stats(stats: Dict[str, float], label: str) -> List[str]:
    return [
        (
            f"  {label}: min={_fmt(stats.get('min'))}, "
            f"p50={_fmt(stats.get('p50'))}, "
            f"p95={_fmt(stats.get('p95'))}, "
            f"p99={_fmt(stats.get('p99'))}, "
            f"max={_fmt(stats.get('max'))}, "
            f"mean={_fmt(stats.get('mean'))}"
        ),
        (
            f"  {label} percentiles: "
            f"p01={_fmt(stats.get('p01'))}, "
            f"p05={_fmt(stats.get('p05'))}, "
            f"p10={_fmt(stats.get('p10'))}, "
            f"p25={_fmt(stats.get('p25'))}, "
            f"p75={_fmt(stats.get('p75'))}, "
            f"p90={_fmt(stats.get('p90'))}"
        ),
    ]


def _format_counts(
    counts: List[Dict[str, float]],
    *,
    indent: str,
) -> List[str]:
    lines = []
    for item in counts:
        level = float(item["level"])
        count = int(item["count"])
        fraction = float(item["fraction"])
        lines.append(f"{indent}<= {level:g}: {count} ({fraction:.1%})")
    return lines


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"
