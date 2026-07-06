#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal-layer point generation near an airfoil profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree


DEFAULT_ETA_LEVELS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
)


@dataclass
class NormalLayerConfig:
    """Settings for normal-layer sampling in physical units."""

    mode: str = "manual"  # manual | y_plus
    profile_step: float = 0.005
    first_layer_height: float = 1.0e-4
    y_plus: float = 1.0
    re_chord: Optional[float] = None
    cf_model: str = "auto"  # auto | laminar | turbulent
    layer_count: int = 14
    growth_mode: str = "manual"  # manual | boundary_layer_estimate
    growth_ratio: float = 1.25
    delta_mode: str = "estimated"  # estimated | manual
    delta: Optional[float] = None
    eta_levels: Optional[Tuple[float, ...]] = None
    open_profile_sides: str = "both"  # both | auto | upper | lower | positive | negative
    endpoint_skip: float = 0.0
    min_point_distance_factor: float = 0.0
    max_points: Optional[int] = None


@dataclass
class NormalLayerResult:
    """Generated layer points and diagnostic metadata."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    surface_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    offsets: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    stats: Dict[str, Any] = field(default_factory=dict)


def generate_normal_layer_points(
    domain: np.ndarray,
    foil: np.ndarray,
    config: NormalLayerConfig,
) -> NormalLayerResult:
    """Generate interior points along normals to the airfoil surface."""

    domain = _as_xy(domain)
    foil = _remove_adjacent_duplicates(_as_xy(foil))
    if len(domain) < 3 or len(foil) < 2:
        return NormalLayerResult(stats={"reason": "empty geometry"})

    closed = _is_closed(foil)
    chord = _chord(foil)
    offsets, offset_stats = _build_offsets(config, chord)
    if len(offsets) == 0:
        return NormalLayerResult(stats={"reason": "empty offsets", **offset_stats})

    base_points, tangents = resample_profile_by_step(
        foil,
        config.profile_step,
        closed=closed,
        endpoint_skip=0.0 if closed else max(float(config.endpoint_skip), 0.0),
    )
    if len(base_points) == 0:
        return NormalLayerResult(stats={"reason": "empty profile samples", **offset_stats})

    normals_a = _normalize(np.column_stack([-tangents[:, 1], tangents[:, 0]]))
    normals_b = -normals_a
    domain_path = Path(domain)
    foil_path = Path(foil) if closed and len(foil) >= 4 else None

    points = []
    columns = 0
    accepted_normals = {"positive": 0, "negative": 0}
    for base, normal_a, normal_b in zip(base_points, normals_a, normals_b):
        for side_name, normal in _candidate_normals(
            base,
            normal_a,
            normal_b,
            offsets[0],
            closed,
            config.open_profile_sides,
            domain_path,
            foil_path,
        ):
            column = base + offsets[:, None] * normal[None, :]
            valid = _valid_flow_points(column, domain_path, foil_path)
            if np.any(valid):
                points.append(column[valid])
                columns += 1
                accepted_normals[side_name] += 1

    if not points:
        return NormalLayerResult(
            surface_points=base_points.astype(np.float64, copy=False),
            offsets=offsets.astype(np.float64, copy=False),
            stats={
                "reason": "all candidates filtered",
                "closed_profile": closed,
                "profile_samples": int(len(base_points)),
                **offset_stats,
            }
        )

    arr = np.vstack(points).astype(np.float64, copy=False)
    before_unique = len(arr)
    arr = _deduplicate(arr)
    before_spacing = len(arr)
    arr = _filter_min_spacing(arr, offsets, config)
    before_limit = len(arr)
    arr = _limit_points(arr, int(config.max_points) if config.max_points else None)

    stats = {
        "closed_profile": bool(closed),
        "profile_samples": int(len(base_points)),
        "accepted_normal_columns": int(columns),
        "accepted_positive_normals": int(accepted_normals["positive"]),
        "accepted_negative_normals": int(accepted_normals["negative"]),
        "raw_points": int(before_unique),
        "unique_points": int(before_spacing),
        "spacing_filtered_points": int(before_limit),
        "points": int(len(arr)),
        "profile_step": float(config.profile_step),
        "open_profile_sides": str(config.open_profile_sides),
        **offset_stats,
    }
    return NormalLayerResult(
        points=arr,
        surface_points=base_points.astype(np.float64, copy=False),
        offsets=offsets.astype(np.float64, copy=False),
        stats=stats,
    )


def resample_profile_by_step(
    points: np.ndarray,
    step: float,
    *,
    closed: Optional[bool] = None,
    endpoint_skip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline by arc length and return points and tangents."""

    pts = _remove_adjacent_duplicates(_as_xy(points))
    if len(pts) < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    is_closed = _is_closed(pts) if closed is None else bool(closed)
    if is_closed:
        work = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
        seg_start = work
        seg_end = np.roll(work, -1, axis=0)
    else:
        work = pts
        seg_start = work[:-1]
        seg_end = work[1:]

    seg = seg_end - seg_start
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 1e-14
    seg_start = seg_start[valid]
    seg = seg[valid]
    seg_len = seg_len[valid]
    if len(seg_len) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    total = float(seg_len.sum())
    step = max(float(step), total / 10000.0, 1e-12)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])

    if is_closed:
        n_samples = max(3, int(np.ceil(total / step)))
        sample_s = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples * total
    else:
        start = min(max(float(endpoint_skip), 0.0), total * 0.49)
        stop = max(start, total - start)
        if stop <= start:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        sample_s = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
        sample_s = np.clip(sample_s, 0.0, np.nextafter(total, 0.0))

    seg_idx = np.searchsorted(cumulative, sample_s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    local = (sample_s - cumulative[seg_idx]) / seg_len[seg_idx]
    sampled = seg_start[seg_idx] + local[:, None] * seg[seg_idx]
    tangents = seg[seg_idx] / seg_len[seg_idx, None]
    return sampled, tangents


def estimate_first_layer_height_y_plus(
    y_plus: float,
    re_chord: float,
    chord: float,
    cf_model: str = "auto",
) -> Tuple[float, Dict[str, Any]]:
    """Estimate first-layer height from y+ using a chord Reynolds number."""

    re = float(re_chord)
    if re <= 0.0 or not np.isfinite(re):
        raise ValueError("re_chord must be positive for y_plus mode")
    cf, model = estimate_skin_friction_coefficient(re, cf_model)
    u_tau_over_u = float(np.sqrt(max(cf, 1e-30) / 2.0))
    y1 = float(chord) * float(y_plus) / (re * u_tau_over_u)
    return y1, {
        "y_plus": float(y_plus),
        "re_chord": re,
        "cf": float(cf),
        "cf_model": model,
        "u_tau_over_u": u_tau_over_u,
    }


def estimate_skin_friction_coefficient(re_chord: float, model: str = "auto") -> Tuple[float, str]:
    """Return a simple flat-plate skin-friction estimate."""

    re = max(float(re_chord), 1.0)
    model_l = str(model or "auto").lower()
    if model_l == "auto":
        model_l = "laminar" if re < 5.0e5 else "turbulent"
    if model_l == "turbulent":
        return 0.0592 / (re ** 0.2), "turbulent"
    return 0.664 / np.sqrt(re), "laminar"


def estimate_boundary_layer_thickness(
    chord: float,
    re_chord: Optional[float],
    cf_model: str = "auto",
) -> Optional[float]:
    """Estimate a boundary-layer thickness in physical units."""

    if re_chord is None:
        return None
    re = float(re_chord)
    if re <= 0.0 or not np.isfinite(re):
        return None
    _, model = estimate_skin_friction_coefficient(re, cf_model)
    if model == "turbulent":
        return float(0.37 * chord / (re ** 0.2))
    return float(5.0 * chord / np.sqrt(re))


def _build_offsets(config: NormalLayerConfig, chord: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode = str(config.mode or "manual").lower()
    stats: Dict[str, Any] = {"mode": mode, "chord": float(chord)}
    if mode == "y_plus":
        if config.re_chord is None:
            raise ValueError("normal_layer_re_chord is required for y_plus mode")
        first, yplus_stats = estimate_first_layer_height_y_plus(
            config.y_plus,
            config.re_chord,
            chord,
            config.cf_model,
        )
        stats.update(yplus_stats)
    else:
        first = float(config.first_layer_height)

    if not np.isfinite(first) or first <= 0.0:
        raise ValueError("first layer height must be positive")

    eta_levels = _positive_array(config.eta_levels)
    if eta_levels is not None:
        offsets = first * eta_levels / float(eta_levels[0])
        stats.update({
            "spacing_mode": "eta_levels",
            "first_layer_height": float(first),
            "eta_levels": [float(v) for v in eta_levels],
            "layer_count": int(len(offsets)),
            "growth_ratio": None,
            "delta": float(offsets[-1]),
        })
        return offsets, stats

    n_layers = max(1, int(config.layer_count))
    growth_mode = str(config.growth_mode or "manual").lower()
    delta = _configured_delta(config, chord)
    ratio = float(config.growth_ratio)
    if growth_mode == "boundary_layer_estimate":
        if delta is None:
            delta = estimate_boundary_layer_thickness(chord, config.re_chord, config.cf_model)
        if delta is not None and delta > first and n_layers > 1:
            ratio = float((delta / first) ** (1.0 / (n_layers - 1)))
            ratio = float(np.clip(ratio, 1.01, 1.8))
        else:
            ratio = 1.0
    ratio = max(float(ratio), 1.0)
    offsets = first * (ratio ** np.arange(n_layers, dtype=np.float64))
    stats.update({
        "spacing_mode": "geometric",
        "first_layer_height": float(first),
        "layer_count": int(n_layers),
        "growth_mode": growth_mode,
        "growth_ratio": float(ratio),
        "delta": float(delta) if delta is not None else None,
    })
    return offsets, stats


def _configured_delta(config: NormalLayerConfig, chord: float) -> Optional[float]:
    if str(config.delta_mode or "estimated").lower() == "manual":
        if config.delta is not None and float(config.delta) > 0.0:
            return float(config.delta)
        return None
    if config.delta is not None and float(config.delta) > 0.0:
        return float(config.delta)
    return estimate_boundary_layer_thickness(chord, config.re_chord, config.cf_model)


def _candidate_normals(
    base: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
    first_offset: float,
    closed: bool,
    sides: str,
    domain_path: Path,
    foil_path: Optional[Path],
):
    side_l = str(sides or "both").lower()
    candidates = [("positive", normal_a), ("negative", normal_b)]
    if closed:
        valid = [
            (name, normal)
            for name, normal in candidates
            if _valid_flow_points((base + normal * first_offset).reshape(1, 2), domain_path, foil_path)[0]
        ]
        if len(valid) == 1:
            return valid
        if len(valid) > 1:
            centroid = np.mean(foil_path.vertices, axis=0) if foil_path is not None else base
            away = [
                (name, normal, float(np.dot(normal, base - centroid)))
                for name, normal in valid
            ]
            best = max(away, key=lambda item: item[2])
            return [(best[0], best[1])]
        return []

    if side_l in {"positive", "normal_a"}:
        candidates = [("positive", normal_a)]
    elif side_l in {"negative", "normal_b"}:
        candidates = [("negative", normal_b)]
    elif side_l == "upper":
        candidates = [("positive", normal_a)] if normal_a[1] >= normal_b[1] else [("negative", normal_b)]
    elif side_l == "lower":
        candidates = [("positive", normal_a)] if normal_a[1] < normal_b[1] else [("negative", normal_b)]

    return [
        (name, normal)
        for name, normal in candidates
        if _valid_flow_points((base + normal * first_offset).reshape(1, 2), domain_path, foil_path)[0]
    ]


def _valid_flow_points(
    points: np.ndarray,
    domain_path: Path,
    foil_path: Optional[Path],
) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(xy).all(axis=1)
    valid &= domain_path.contains_points(xy)
    if foil_path is not None:
        valid &= ~foil_path.contains_points(xy)
    return valid


def _filter_min_spacing(
    points: np.ndarray,
    offsets: np.ndarray,
    config: NormalLayerConfig,
) -> np.ndarray:
    factor = float(config.min_point_distance_factor)
    if len(points) <= 1 or factor <= 0.0:
        return points
    min_dist = max(float(np.min(offsets)) * factor, 1e-14)
    tree = cKDTree(points)
    keep = np.ones(len(points), dtype=bool)
    for idx in range(len(points)):
        if not keep[idx]:
            continue
        neighbors = tree.query_ball_point(points[idx], r=min_dist)
        for nb in neighbors:
            if nb > idx:
                keep[nb] = False
    return points[keep]


def _limit_points(points: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def _deduplicate(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    _, idx = np.unique(np.round(points, decimals=12), axis=0, return_index=True)
    return points[np.sort(idx)]


def _positive_array(values: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) == 0:
        return None
    return np.unique(np.sort(arr))


def _remove_adjacent_duplicates(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-14
    return points[keep]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1)
    norm = np.where(norm > 1e-14, norm, 1.0)
    return vectors / norm[:, None]


def _as_xy(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("geometry coordinates must have shape (N, 2)")
    return arr[:, :2]


def _is_closed(points: np.ndarray) -> bool:
    return len(points) >= 3 and np.allclose(points[0], points[-1], atol=1e-9)


def _chord(foil: np.ndarray) -> float:
    chord = float(np.max(foil[:, 0]) - np.min(foil[:, 0]))
    if chord <= 0.0 or not np.isfinite(chord):
        chord = float(np.linalg.norm(np.ptp(foil, axis=0)))
    return max(chord, 1e-12)


__all__ = [
    "DEFAULT_ETA_LEVELS",
    "NormalLayerConfig",
    "NormalLayerResult",
    "estimate_boundary_layer_thickness",
    "estimate_first_layer_height_y_plus",
    "estimate_skin_friction_coefficient",
    "generate_normal_layer_points",
    "resample_profile_by_step",
]
