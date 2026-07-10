#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary-layer diagnostics for airfoil PINN runs.

The routines in this module do not change the loss function.  They only sample
the trained PINN along wall-normal columns, compute profile metrics, and save
plots/tables that make a thin boundary layer visible and measurable.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from airfoil_pinn.sampling.normal_layer import (
    NormalLayerColumnsResult,
    NormalLayerConfig,
    generate_normal_layer_columns,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundaryLayerDiagnosticsConfig",
    "BoundaryLayerDiagnosticsResult",
    "run_boundary_layer_diagnostics",
]


@dataclass
class BoundaryLayerDiagnosticsConfig:
    """Configuration for wall-normal boundary-layer post-processing."""

    enabled: bool = False
    interval: int = 50
    sections: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95)
    sides: Tuple[str, ...] = ("upper", "lower")
    profile_step: Optional[float] = None
    profile_step_fraction: float = 0.01
    first_layer_height: Optional[float] = None
    first_layer_height_fraction: float = 1.0e-5
    layer_count: int = 80
    growth_ratio: float = 1.12
    max_normal_distance: Optional[float] = None
    max_normal_distance_fraction: float = 0.08
    edge_threshold: float = 0.99
    fit_points: int = 8
    x_min_c: float = 0.02
    x_max_c: float = 0.98
    max_surface_points_per_side: int = 120
    unresolved_threshold: float = 0.80
    eval_batch_size: int = 8192
    save_profiles_npz: bool = True
    save_metrics_csv: bool = True
    plot_profiles: bool = True
    plot_cp: bool = True
    plot_cf: bool = True
    plot_delta99: bool = True
    plot_deficit: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "BoundaryLayerDiagnosticsConfig":
        """Build a typed config from a nested YAML dictionary."""

        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            interval=max(1, int(data.get("interval", data.get("diagnostics_interval", 50)))),
            sections=_tuple_float(data.get("sections", cls.sections)),
            sides=_tuple_str(data.get("sides", cls.sides)),
            profile_step=_optional_float(data.get("profile_step")),
            profile_step_fraction=float(data.get("profile_step_fraction", cls.profile_step_fraction)),
            first_layer_height=_optional_float(data.get("first_layer_height")),
            first_layer_height_fraction=float(
                data.get("first_layer_height_fraction", cls.first_layer_height_fraction)
            ),
            layer_count=max(2, int(data.get("layer_count", cls.layer_count))),
            growth_ratio=max(1.0, float(data.get("growth_ratio", cls.growth_ratio))),
            max_normal_distance=_optional_float(data.get("max_normal_distance")),
            max_normal_distance_fraction=float(
                data.get("max_normal_distance_fraction", cls.max_normal_distance_fraction)
            ),
            edge_threshold=float(data.get("edge_threshold", cls.edge_threshold)),
            fit_points=max(2, int(data.get("fit_points", cls.fit_points))),
            x_min_c=float(data.get("x_min_c", cls.x_min_c)),
            x_max_c=float(data.get("x_max_c", cls.x_max_c)),
            max_surface_points_per_side=max(
                1,
                int(data.get("max_surface_points_per_side", cls.max_surface_points_per_side)),
            ),
            unresolved_threshold=float(data.get("unresolved_threshold", cls.unresolved_threshold)),
            eval_batch_size=max(128, int(data.get("eval_batch_size", cls.eval_batch_size))),
            save_profiles_npz=bool(data.get("save_profiles_npz", cls.save_profiles_npz)),
            save_metrics_csv=bool(data.get("save_metrics_csv", cls.save_metrics_csv)),
            plot_profiles=bool(data.get("plot_profiles", cls.plot_profiles)),
            plot_cp=bool(data.get("plot_cp", cls.plot_cp)),
            plot_cf=bool(data.get("plot_cf", cls.plot_cf)),
            plot_delta99=bool(data.get("plot_delta99", cls.plot_delta99)),
            plot_deficit=bool(data.get("plot_deficit", cls.plot_deficit)),
        )

    def normal_layer_config(self, chord: float, re_chord: Optional[float]) -> NormalLayerConfig:
        """Create the normal-layer sampling config used for diagnostics."""

        chord = max(float(chord), 1.0e-12)
        profile_step = self.profile_step
        if profile_step is None:
            profile_step = chord * max(float(self.profile_step_fraction), 1.0e-6)
        first_height = self.first_layer_height
        if first_height is None:
            first_height = chord * max(float(self.first_layer_height_fraction), 1.0e-9)

        return NormalLayerConfig(
            mode="manual",
            profile_step=float(profile_step),
            first_layer_height=float(first_height),
            re_chord=re_chord,
            layer_count=int(self.layer_count),
            growth_mode="manual",
            growth_ratio=float(self.growth_ratio),
            open_profile_sides="both",
        )

    def max_distance(self, chord: float) -> float:
        """Return the farthest wall-normal distance used in plots and metrics."""

        if self.max_normal_distance is not None and self.max_normal_distance > 0.0:
            return float(self.max_normal_distance)
        return float(chord) * max(float(self.max_normal_distance_fraction), 1.0e-6)


@dataclass
class BoundaryLayerDiagnosticsResult:
    """Files and aggregate values produced by one diagnostics call."""

    summary: Dict[str, float]
    metrics_rows: List[Dict[str, float]]
    files: Dict[str, str]


def run_boundary_layer_diagnostics(
    *,
    model: torch.nn.Module,
    loss_computer: Any,
    data_handler: Any,
    device: torch.device,
    dtype: torch.dtype,
    savedir: str,
    epoch: int,
    config: Optional[Dict[str, Any]],
) -> Optional[BoundaryLayerDiagnosticsResult]:
    """Run the complete boundary-layer diagnostics pipeline for one epoch."""

    cfg = BoundaryLayerDiagnosticsConfig.from_dict(config)
    if not cfg.enabled or int(epoch) % cfg.interval != 0:
        return None

    domain, foil = _load_geometry(data_handler)
    if domain is None or foil is None:
        logger.warning("Boundary-layer diagnostics skipped: airfoil/domain geometry is unavailable.")
        return None

    chord = float(getattr(data_handler, "chord", 1.0) or 1.0)
    re_chord = float(getattr(data_handler, "Re", 0.0) or 0.0)
    layer_cfg = cfg.normal_layer_config(chord=chord, re_chord=re_chord)
    columns = generate_normal_layer_columns(domain, foil, layer_cfg)
    columns = _filter_columns(columns, cfg)
    if len(columns.surface_points) == 0:
        logger.warning("Boundary-layer diagnostics skipped: no valid wall-normal columns.")
        return None

    profiles = evaluate_wall_normal_profiles(
        model=model,
        loss_computer=loss_computer,
        data_handler=data_handler,
        columns=columns,
        cfg=cfg,
        chord=chord,
        device=device,
        dtype=dtype,
    )
    metrics_rows = compute_boundary_layer_metrics(
        profiles=profiles,
        cfg=cfg,
        re_chord=max(re_chord, 1.0e-12),
    )
    summary = summarize_boundary_layer_metrics(metrics_rows, cfg)
    files = save_boundary_layer_outputs(
        profiles=profiles,
        metrics_rows=metrics_rows,
        cfg=cfg,
        savedir=savedir,
        epoch=epoch,
        foil=foil,
        chord=chord,
    )

    logger.info(
        "BL diagnostics epoch %s: profiles=%d, unresolved=%d/%d, "
        "u_first/Uedge max=%.3g, delta99 mean=%.3e, Cf rms=%.3e",
        epoch,
        len(metrics_rows),
        int(summary.get("unresolved_profiles", 0.0)),
        int(summary.get("profile_count", 0.0)),
        float(summary.get("u_first_over_Uedge_max", np.nan)),
        float(summary.get("delta99_mean", np.nan)),
        float(summary.get("Cf_rms", np.nan)),
    )
    return BoundaryLayerDiagnosticsResult(summary=summary, metrics_rows=metrics_rows, files=files)


def evaluate_wall_normal_profiles(
    *,
    model: torch.nn.Module,
    loss_computer: Any,
    data_handler: Any,
    columns: NormalLayerColumnsResult,
    cfg: BoundaryLayerDiagnosticsConfig,
    chord: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    """Evaluate the PINN on wall-normal columns and project velocity to t/n axes."""

    offsets = np.asarray(columns.offsets, dtype=np.float64)
    active_offsets = offsets <= cfg.max_distance(chord)
    valid_mask = np.asarray(columns.valid_mask, dtype=bool) & active_offsets[None, :]
    n_cols = int(len(columns.surface_points))
    n_layers = int(len(offsets))

    shape = (n_cols, n_layers + 1)
    x_norm = np.full(shape, np.nan, dtype=np.float64)
    y_norm = np.full(shape, np.nan, dtype=np.float64)
    n_norm = np.full(shape, np.nan, dtype=np.float64)
    u_norm = np.full(shape, np.nan, dtype=np.float64)
    v_norm = np.full(shape, np.nan, dtype=np.float64)
    p_norm = np.full(shape, np.nan, dtype=np.float64)

    flat_points = []
    flat_sdf = []
    flat_labels = []
    flat_indices = []
    for col_idx in range(n_cols):
        wall = columns.surface_points[col_idx]
        flat_points.append(wall)
        flat_sdf.append(0.0)
        flat_labels.append("wall")
        flat_indices.append((col_idx, 0))
        x_norm[col_idx, 0] = wall[0] / chord
        y_norm[col_idx, 0] = wall[1] / chord
        n_norm[col_idx, 0] = 0.0

        for layer_idx in np.where(valid_mask[col_idx])[0]:
            pt = columns.points_by_column[col_idx, layer_idx]
            flat_points.append(pt)
            flat_sdf.append(offsets[layer_idx] / chord)
            flat_labels.append("interior")
            flat_indices.append((col_idx, int(layer_idx) + 1))
            x_norm[col_idx, layer_idx + 1] = pt[0] / chord
            y_norm[col_idx, layer_idx + 1] = pt[1] / chord
            n_norm[col_idx, layer_idx + 1] = offsets[layer_idx] / chord

    if not flat_points:
        raise ValueError("evaluate_wall_normal_profiles: no points to evaluate")

    points_np = np.asarray(flat_points, dtype=np.float64)
    sdf_np = np.asarray(flat_sdf, dtype=np.float64)
    bt_np = _boundary_type_matrix(flat_labels, getattr(data_handler, "boundary_type_names", []))
    pred = _predict_model_on_points(
        model=model,
        loss_computer=loss_computer,
        x_norm_np=points_np[:, 0] / chord,
        y_norm_np=points_np[:, 1] / chord,
        sdf_norm_np=sdf_np,
        boundary_types_np=bt_np,
        device=device,
        dtype=dtype,
        batch_size=cfg.eval_batch_size,
    )

    for flat_idx, (col_idx, layer_slot) in enumerate(flat_indices):
        u_norm[col_idx, layer_slot] = pred["u"][flat_idx]
        v_norm[col_idx, layer_slot] = pred["v"][flat_idx]
        p_norm[col_idx, layer_slot] = pred["p"][flat_idx]

    tangents = np.asarray(columns.tangents, dtype=np.float64)
    normals = np.asarray(columns.normals, dtype=np.float64)
    u_t = u_norm * tangents[:, 0:1] + v_norm * tangents[:, 1:2]
    u_n = u_norm * normals[:, 0:1] + v_norm * normals[:, 1:2]

    return {
        "x_norm": x_norm,
        "y_norm": y_norm,
        "n_norm": n_norm,
        "u_norm": u_norm,
        "v_norm": v_norm,
        "p_norm": p_norm,
        "u_t_norm": u_t,
        "u_n_norm": u_n,
        "surface_x_norm": columns.surface_points[:, 0] / chord,
        "surface_y_norm": columns.surface_points[:, 1] / chord,
        "normal_x": normals[:, 0],
        "normal_y": normals[:, 1],
        "tangent_x": tangents[:, 0],
        "tangent_y": tangents[:, 1],
        "x_c": np.asarray(columns.x_c, dtype=np.float64),
        "side": np.asarray(columns.side_names, dtype=object),
        "offsets_norm": np.concatenate([[0.0], offsets / chord]),
    }


def compute_boundary_layer_metrics(
    *,
    profiles: Dict[str, np.ndarray],
    cfg: BoundaryLayerDiagnosticsConfig,
    re_chord: float,
) -> List[Dict[str, float]]:
    """Compute Cp, Cf and integral boundary-layer metrics for each column."""

    rows: List[Dict[str, float]] = []
    n_cols = int(len(profiles["x_c"]))
    for col_idx in range(n_cols):
        n = profiles["n_norm"][col_idx]
        u_t = profiles["u_t_norm"][col_idx]
        p = profiles["p_norm"][col_idx]
        finite = np.isfinite(n) & np.isfinite(u_t)
        if finite.sum() < 2:
            continue

        n_f = n[finite]
        u_f = u_t[finite]
        p_f = p[finite]
        order = np.argsort(n_f)
        n_f = n_f[order]
        u_f = u_f[order]
        p_f = p_f[order]

        u_edge = _edge_velocity(u_f)
        ratio = u_f / u_edge if np.isfinite(u_edge) and abs(u_edge) > 1.0e-12 else np.full_like(u_f, np.nan)
        delta99 = _first_threshold_crossing(n_f, ratio, cfg.edge_threshold)
        u_first = float(u_f[1]) if len(u_f) > 1 else np.nan
        u_first_ratio = u_first / u_edge if np.isfinite(u_edge) and abs(u_edge) > 1.0e-12 else np.nan
        n_first = float(n_f[1]) if len(n_f) > 1 else np.nan
        n_first_over_delta99 = n_first / delta99 if np.isfinite(delta99) and delta99 > 0.0 else np.nan
        du_dn = _wall_slope(n_f, u_f, cfg.fit_points)
        cf = 2.0 * du_dn / re_chord if np.isfinite(du_dn) else np.nan
        delta_star, theta, shape_factor = _integral_thicknesses(n_f, ratio)

        rows.append({
            "column": float(col_idx),
            "x_c": float(profiles["x_c"][col_idx]),
            "side": profiles["side"][col_idx],
            "x_norm": float(profiles["surface_x_norm"][col_idx]),
            "y_norm": float(profiles["surface_y_norm"][col_idx]),
            "normal_x": float(profiles["normal_x"][col_idx]),
            "normal_y": float(profiles["normal_y"][col_idx]),
            "tangent_x": float(profiles["tangent_x"][col_idx]),
            "tangent_y": float(profiles["tangent_y"][col_idx]),
            "Cp": float(p_f[0]) if len(p_f) else np.nan,
            "Cf": float(cf),
            "du_t_dn_wall": float(du_dn),
            "U_edge": float(u_edge),
            "delta99": float(delta99),
            "delta_star": float(delta_star),
            "theta": float(theta),
            "H": float(shape_factor),
            "u_wall": float(u_f[0]),
            "u_first": float(u_first),
            "u_first_over_Uedge": float(u_first_ratio),
            "n_first": float(n_first),
            "n_first_over_delta99": float(n_first_over_delta99),
            "resolved_flag": float(
                np.isfinite(u_first_ratio) and u_first_ratio <= cfg.unresolved_threshold
            ),
        })
    return rows


def summarize_boundary_layer_metrics(
    rows: Sequence[Dict[str, float]],
    cfg: BoundaryLayerDiagnosticsConfig,
) -> Dict[str, float]:
    """Reduce per-column metrics to a compact log-friendly summary."""

    if not rows:
        return {
            "profile_count": 0.0,
            "unresolved_profiles": 0.0,
            "u_first_over_Uedge_max": np.nan,
            "delta99_mean": np.nan,
            "Cf_rms": np.nan,
        }

    u_first = _finite_values(row.get("u_first_over_Uedge", np.nan) for row in rows)
    delta99 = _finite_values(row.get("delta99", np.nan) for row in rows)
    cf = _finite_values(row.get("Cf", np.nan) for row in rows)
    cp = _finite_values(row.get("Cp", np.nan) for row in rows)
    unresolved = sum(
        1
        for row in rows
        if np.isfinite(row.get("u_first_over_Uedge", np.nan))
        and row["u_first_over_Uedge"] > cfg.unresolved_threshold
    )
    return {
        "profile_count": float(len(rows)),
        "unresolved_profiles": float(unresolved),
        "u_first_over_Uedge_mean": _safe_mean(u_first),
        "u_first_over_Uedge_max": _safe_max(u_first),
        "delta99_mean": _safe_mean(delta99),
        "delta99_min": _safe_min(delta99),
        "delta99_max": _safe_max(delta99),
        "Cf_rms": float(np.sqrt(np.mean(cf * cf))) if len(cf) else np.nan,
        "Cf_max_abs": float(np.max(np.abs(cf))) if len(cf) else np.nan,
        "Cp_min": _safe_min(cp),
        "Cp_max": _safe_max(cp),
    }


def save_boundary_layer_outputs(
    *,
    profiles: Dict[str, np.ndarray],
    metrics_rows: Sequence[Dict[str, float]],
    cfg: BoundaryLayerDiagnosticsConfig,
    savedir: str,
    epoch: int,
    foil: np.ndarray,
    chord: float,
) -> Dict[str, str]:
    """Save boundary-layer profiles, coefficient tables and diagnostic plots."""

    os.makedirs(savedir, exist_ok=True)
    files: Dict[str, str] = {}
    suffix = f"_epoch_{epoch}"
    if cfg.save_profiles_npz:
        path = os.path.join(savedir, f"boundary_layer_profiles{suffix}.npz")
        np.savez_compressed(path, **profiles)
        files["profiles_npz"] = path
    if cfg.save_metrics_csv:
        path = os.path.join(savedir, f"boundary_layer_metrics{suffix}.csv")
        _write_metrics_csv(path, metrics_rows)
        files["metrics_csv"] = path
    if cfg.plot_profiles:
        path = os.path.join(savedir, f"bl_profiles{suffix}.png")
        plot_wall_normal_profiles(path, profiles, metrics_rows, cfg)
        files["profiles_plot"] = path
    if cfg.plot_cp:
        path = os.path.join(savedir, f"cp_profile{suffix}.png")
        plot_surface_coefficient(path, metrics_rows, key="Cp", invert_y=True)
        files["cp_plot"] = path
    if cfg.plot_cf:
        path = os.path.join(savedir, f"cf_profile{suffix}.png")
        plot_surface_coefficient(path, metrics_rows, key="Cf", invert_y=False)
        files["cf_plot"] = path
    if cfg.plot_delta99:
        path = os.path.join(savedir, f"delta99_profile{suffix}.png")
        plot_surface_coefficient(path, metrics_rows, key="delta99", invert_y=False)
        files["delta99_plot"] = path
    if cfg.plot_deficit:
        path = os.path.join(savedir, f"bl_deficit_columns{suffix}.png")
        plot_wall_normal_deficit(path, profiles, foil / chord)
        files["deficit_plot"] = path
    return files


def plot_wall_normal_profiles(
    path: str,
    profiles: Dict[str, np.ndarray],
    metrics_rows: Sequence[Dict[str, float]],
    cfg: BoundaryLayerDiagnosticsConfig,
) -> None:
    """Plot selected wall-normal velocity profiles for upper and lower sides."""

    row_by_column = {int(row["column"]): row for row in metrics_rows}
    selected = _select_section_columns(profiles, cfg.sections, cfg.sides)
    if not selected:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    side_to_ax = {"upper": axes[0], "lower": axes[1]}
    for col_idx in selected:
        side = str(profiles["side"][col_idx])
        ax = side_to_ax.get(side, axes[0])
        row = row_by_column.get(int(col_idx), {})
        n = profiles["n_norm"][col_idx]
        u_t = profiles["u_t_norm"][col_idx]
        u_edge = float(row.get("U_edge", np.nan))
        finite = np.isfinite(n) & np.isfinite(u_t)
        if finite.sum() < 2 or not np.isfinite(u_edge) or abs(u_edge) < 1.0e-12:
            continue
        label = f"x/c={profiles['x_c'][col_idx]:.2f}"
        ax.plot(u_t[finite] / u_edge, n[finite], marker=".", linewidth=1.2, label=label)

    for side, ax in side_to_ax.items():
        ax.set_title(f"{side.title()} wall-normal profiles")
        ax.set_xlabel("u_t / U_edge")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("n / chord")
    fig.suptitle("Boundary-layer profiles")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_surface_coefficient(
    path: str,
    metrics_rows: Sequence[Dict[str, float]],
    *,
    key: str,
    invert_y: bool,
) -> None:
    """Plot a surface coefficient against x/c for upper and lower surfaces."""

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for side in ("upper", "lower"):
        rows = [
            row
            for row in metrics_rows
            if row.get("side") == side
            and np.isfinite(float(row.get("x_c", np.nan)))
            and np.isfinite(float(row.get(key, np.nan)))
        ]
        rows = sorted(rows, key=lambda item: item.get("x_c", np.nan))
        if not rows:
            continue
        x = np.asarray([float(row["x_c"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker=".", linewidth=1.2, label=side)
    ax.set_xlabel("x/c")
    ax.set_ylabel(key)
    ax.set_title(f"{key} along airfoil surface")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if invert_y:
        ax.invert_yaxis()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wall_normal_deficit(path: str, profiles: Dict[str, np.ndarray], foil_norm: np.ndarray) -> None:
    """Scatter evaluated normal columns colored by the tangential velocity deficit."""

    x = profiles["x_norm"]
    y = profiles["y_norm"]
    u_t = profiles["u_t_norm"]
    x_c = profiles["x_c"]
    deficit = np.full_like(u_t, np.nan, dtype=np.float64)
    for col_idx in range(u_t.shape[0]):
        finite = np.isfinite(u_t[col_idx])
        if finite.sum() < 2:
            continue
        edge = _edge_velocity(u_t[col_idx, finite])
        if np.isfinite(edge) and abs(edge) > 1.0e-12:
            deficit[col_idx, finite] = np.clip(1.0 - u_t[col_idx, finite] / edge, 0.0, 1.0)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(deficit)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    if np.any(finite):
        sc = ax.scatter(x[finite], y[finite], c=deficit[finite], s=8, cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(sc, ax=ax, label="1 - u_t/U_edge")
    if foil_norm is not None and len(foil_norm) >= 2:
        ax.plot(foil_norm[:, 0], foil_norm[:, 1], color="cyan", linewidth=1.0)
    ax.set_xlim(float(np.nanmin(x_c)) - 0.05, float(np.nanmax(x_c)) + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/chord")
    ax.set_ylabel("y/chord")
    ax.set_title("Wall-normal columns colored by velocity deficit")
    ax.grid(True, alpha=0.15)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_geometry(data_handler: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return domain and airfoil geometry, loading metadata if the handler can."""

    if getattr(data_handler, "mesh_domain_coords", None) is None or getattr(data_handler, "mesh_airfoil_coords", None) is None:
        loader = getattr(data_handler, "load_mesh_geometry_from_geom_config", None)
        if callable(loader):
            loader()
    domain = getattr(data_handler, "mesh_domain_coords", None)
    foil = getattr(data_handler, "mesh_airfoil_coords", None)
    if domain is None or foil is None:
        return None, None
    domain = np.asarray(domain, dtype=np.float64).reshape(-1, 2)
    foil = np.asarray(foil, dtype=np.float64).reshape(-1, 2)
    return domain, foil


def _filter_columns(
    columns: NormalLayerColumnsResult,
    cfg: BoundaryLayerDiagnosticsConfig,
) -> NormalLayerColumnsResult:
    """Keep requested sides and useful x/c range, with optional per-side decimation."""

    if len(columns.surface_points) == 0:
        return columns
    sides = {side.lower() for side in cfg.sides}
    keep = (
        np.isfinite(columns.x_c)
        & (columns.x_c >= cfg.x_min_c)
        & (columns.x_c <= cfg.x_max_c)
        & np.asarray([str(side).lower() in sides for side in columns.side_names], dtype=bool)
    )
    indices = np.where(keep)[0]
    indices = _limit_indices_per_side(indices, columns.side_names, cfg.max_surface_points_per_side)
    if len(indices) == len(columns.surface_points):
        return columns
    return NormalLayerColumnsResult(
        points=columns.points,
        surface_points=columns.surface_points[indices],
        tangents=columns.tangents[indices],
        normals=columns.normals[indices],
        offsets=columns.offsets,
        points_by_column=columns.points_by_column[indices],
        valid_mask=columns.valid_mask[indices],
        side_names=columns.side_names[indices],
        x_c=columns.x_c[indices],
        stats=dict(columns.stats, filtered_columns=int(len(indices))),
    )


def _predict_model_on_points(
    *,
    model: torch.nn.Module,
    loss_computer: Any,
    x_norm_np: np.ndarray,
    y_norm_np: np.ndarray,
    sdf_norm_np: np.ndarray,
    boundary_types_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Run the PINN on arbitrary normalized points in evaluation batches."""

    was_training = model.training
    model.eval()
    all_u: List[np.ndarray] = []
    all_v: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(x_norm_np), batch_size):
                stop = min(start + batch_size, len(x_norm_np))
                x = torch.as_tensor(x_norm_np[start:stop], device=device, dtype=dtype).view(-1, 1)
                y = torch.as_tensor(y_norm_np[start:stop], device=device, dtype=dtype).view(-1, 1)
                sdf = torch.as_tensor(sdf_norm_np[start:stop], device=device, dtype=dtype).view(-1, 1)
                bt = torch.as_tensor(boundary_types_np[start:stop], device=device, dtype=dtype)
                if bt.shape[1] == 0:
                    bt = None
                inp = loss_computer.build_input_tensor(x, y, sdf, bt)
                out = model(inp)
                u = out[:, 0:1]
                v = out[:, 1:2]
                p = out[:, 2:3]
                if hasattr(loss_computer, "apply_velocity_dirichlet_constraint"):
                    u, v = loss_computer.apply_velocity_dirichlet_constraint(u, v, bt)
                all_u.append(u.detach().cpu().numpy().reshape(-1))
                all_v.append(v.detach().cpu().numpy().reshape(-1))
                all_p.append(p.detach().cpu().numpy().reshape(-1))
    finally:
        if was_training:
            model.train()
    return {
        "u": np.concatenate(all_u),
        "v": np.concatenate(all_v),
        "p": np.concatenate(all_p),
    }


def _boundary_type_matrix(labels: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Create a one-hot boundary-type matrix for synthetic diagnostic points."""

    names_l = [str(name).lower() for name in names]
    arr = np.zeros((len(labels), len(names_l)), dtype=np.float32)
    for row, label in enumerate(labels):
        label_l = str(label).lower()
        if label_l in names_l:
            arr[row, names_l.index(label_l)] = 1.0
    return arr


def _write_metrics_csv(path: str, rows: Sequence[Dict[str, float]]) -> None:
    """Write per-column boundary-layer metrics to CSV."""

    fieldnames = [
        "column",
        "x_c",
        "side",
        "x_norm",
        "y_norm",
        "normal_x",
        "normal_y",
        "tangent_x",
        "tangent_y",
        "Cp",
        "Cf",
        "du_t_dn_wall",
        "U_edge",
        "delta99",
        "delta_star",
        "theta",
        "H",
        "u_wall",
        "u_first",
        "u_first_over_Uedge",
        "n_first",
        "n_first_over_delta99",
        "resolved_flag",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _select_section_columns(
    profiles: Dict[str, np.ndarray],
    sections: Sequence[float],
    sides: Sequence[str],
) -> List[int]:
    """Pick one closest normal column for every requested x/c section and side."""

    selected: List[int] = []
    x_c = np.asarray(profiles["x_c"], dtype=np.float64)
    side_arr = np.asarray(profiles["side"], dtype=object)
    for side in sides:
        side_mask = np.asarray([str(value).lower() == str(side).lower() for value in side_arr])
        if not np.any(side_mask):
            continue
        side_indices = np.where(side_mask)[0]
        for section in sections:
            local = side_indices[np.argmin(np.abs(x_c[side_indices] - float(section)))]
            if int(local) not in selected:
                selected.append(int(local))
    return selected


def _limit_indices_per_side(indices: np.ndarray, side_names: np.ndarray, max_per_side: int) -> np.ndarray:
    """Decimate dense surface columns while keeping upper/lower coverage."""

    kept: List[int] = []
    for side in ("upper", "lower"):
        side_idx = [int(idx) for idx in indices if str(side_names[idx]).lower() == side]
        if len(side_idx) <= max_per_side:
            kept.extend(side_idx)
        else:
            pick = np.linspace(0, len(side_idx) - 1, max_per_side, dtype=np.int64)
            kept.extend([side_idx[int(i)] for i in pick])
    return np.asarray(sorted(kept), dtype=np.int64)


def _edge_velocity(u_t: np.ndarray) -> float:
    """Estimate edge velocity from the last part of one wall-normal profile."""

    finite = np.asarray(u_t, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    tail_count = max(1, min(len(finite), max(3, len(finite) // 5)))
    edge = float(np.nanmedian(finite[-tail_count:]))
    if abs(edge) < 1.0e-12:
        edge = float(finite[-1])
    return edge


def _first_threshold_crossing(n: np.ndarray, ratio: np.ndarray, threshold: float) -> float:
    """Return the first wall-normal coordinate where u/Uedge reaches a threshold."""

    finite = np.isfinite(n) & np.isfinite(ratio)
    n_f = n[finite]
    r_f = ratio[finite]
    if len(n_f) == 0:
        return np.nan
    hit = np.where(r_f >= threshold)[0]
    if len(hit) == 0:
        return np.nan
    idx = int(hit[0])
    if idx == 0:
        return float(n_f[0])
    n0, n1 = float(n_f[idx - 1]), float(n_f[idx])
    r0, r1 = float(r_f[idx - 1]), float(r_f[idx])
    if abs(r1 - r0) < 1.0e-12:
        return n1
    alpha = np.clip((threshold - r0) / (r1 - r0), 0.0, 1.0)
    return float(n0 + alpha * (n1 - n0))


def _wall_slope(n: np.ndarray, u_t: np.ndarray, fit_points: int) -> float:
    """Estimate du_t/dn at the wall with a small least-squares line fit."""

    finite = np.isfinite(n) & np.isfinite(u_t)
    n_f = n[finite]
    u_f = u_t[finite]
    if len(n_f) < 2:
        return np.nan
    count = min(max(2, int(fit_points)), len(n_f))
    try:
        slope, _ = np.polyfit(n_f[:count], u_f[:count], deg=1)
    except Exception:
        return np.nan
    return float(slope)


def _integral_thicknesses(n: np.ndarray, ratio: np.ndarray) -> Tuple[float, float, float]:
    """Compute displacement thickness, momentum thickness and shape factor."""

    finite = np.isfinite(n) & np.isfinite(ratio)
    n_f = n[finite]
    r_f = np.clip(ratio[finite], 0.0, 1.5)
    if len(n_f) < 2:
        return np.nan, np.nan, np.nan
    delta_star = float(np.trapezoid(1.0 - np.clip(r_f, 0.0, 1.0), n_f))
    theta = float(np.trapezoid(np.clip(r_f, 0.0, 1.0) * (1.0 - np.clip(r_f, 0.0, 1.0)), n_f))
    shape = delta_star / theta if theta > 1.0e-14 else np.nan
    return delta_star, theta, shape


def _tuple_float(values: Any) -> Tuple[float, ...]:
    """Convert a config value to a tuple of finite floats."""

    if values is None:
        return tuple()
    if isinstance(values, (int, float)):
        values = [values]
    return tuple(float(v) for v in values if np.isfinite(float(v)))


def _tuple_str(values: Any) -> Tuple[str, ...]:
    """Convert a config value to a tuple of lowercase strings."""

    if values is None:
        return tuple()
    if isinstance(values, str):
        values = [values]
    return tuple(str(v).lower() for v in values)


def _optional_float(value: Any) -> Optional[float]:
    """Parse optional float values from YAML-like dictionaries."""

    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _finite_values(values: Any) -> np.ndarray:
    """Return finite values from an iterable as a numpy array."""

    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_mean(values: np.ndarray) -> float:
    """Return a finite mean or NaN for empty arrays."""

    return float(np.mean(values)) if len(values) else np.nan


def _safe_min(values: np.ndarray) -> float:
    """Return a finite minimum or NaN for empty arrays."""

    return float(np.min(values)) if len(values) else np.nan


def _safe_max(values: np.ndarray) -> float:
    """Return a finite maximum or NaN for empty arrays."""

    return float(np.max(values)) if len(values) else np.nan
