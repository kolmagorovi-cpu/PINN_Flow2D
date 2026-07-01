#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Adaptive Refinement для PINN.

Модуль разделяет адаптивное добавление точек на два независимых канала:
- FlowRAR: прежний residual-based RAR для свободного потока.
- BoundaryLayerRAR: отдельный пристенный RAR, строящий кандидаты по нормалям
  к профилю на уровнях eta = SDF / delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


DEFAULT_ETA_LEVELS: Tuple[float, ...] = (
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
class RARGenerationResult:
    """Новые точки RAR и диагностическая статистика."""

    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sdf: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(len(self.x))


class RARManager:
    """Координирует FlowRAR и BoundaryLayerRAR."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.config = trainer.config
        self.flow = FlowRAR(trainer)
        self.boundary = BoundaryLayerRAR(trainer)

    def generate(self, total_points: int) -> Optional[RARGenerationResult]:
        if total_points <= 0:
            return None

        flow_cfg = self.config.get("flow_rar", {}) or {}
        boundary_cfg = self.config.get("boundary_rar", {}) or {}

        boundary_enabled = bool(boundary_cfg.get("enabled", False))
        flow_enabled = bool(flow_cfg.get("enabled", True))

        boundary_n = 0
        if boundary_enabled:
            boundary_num_points = boundary_cfg.get("num_points", None)
            if boundary_num_points is not None:
                boundary_n = max(0, int(boundary_num_points))
            else:
                fraction = float(boundary_cfg.get("fraction", boundary_cfg.get("boundary_rar_fraction", 0.4)))
                boundary_n = int(round(total_points * min(max(fraction, 0.0), 1.0)))

        flow_num_points = flow_cfg.get("num_points", None)
        if flow_num_points is not None:
            flow_n = max(0, int(flow_num_points))
        else:
            flow_n = max(0, total_points - boundary_n) if boundary_enabled else total_points

        parts: List[Tuple[str, RARGenerationResult]] = []
        boundary_added = 0
        if boundary_enabled and boundary_n > 0:
            logger.info("Boundary-Layer RAR: запрос %d точек.", boundary_n)
            boundary_result = self.boundary.generate(boundary_n)
            if boundary_result is not None and boundary_result.n_points > 0:
                parts.append(("boundary", boundary_result))
                boundary_added = boundary_result.n_points

        if flow_enabled and boundary_enabled and boundary_added < boundary_n and flow_num_points is None:
            flow_n += boundary_n - boundary_added

        if flow_enabled and flow_n > 0:
            logger.info("Flow RAR: запрос %d точек.", flow_n)
            flow_result = self.flow.generate(flow_n)
            if flow_result is not None and flow_result.n_points > 0:
                parts.append(("flow", flow_result))

        if not parts:
            return None

        x = np.concatenate([part.x for _, part in parts]).astype(np.float64, copy=False)
        y = np.concatenate([part.y for _, part in parts]).astype(np.float64, copy=False)
        sdf = np.concatenate([part.sdf for _, part in parts]).astype(np.float64, copy=False)
        x, y, sdf = self._deduplicate(x, y, sdf)

        stats = {
            "requested_total": int(total_points),
            "requested_boundary": int(boundary_n),
            "requested_flow": int(flow_n),
            "generated_total": int(len(x)),
            "parts": {name: part.stats for name, part in parts},
        }
        return RARGenerationResult(x=x, y=y, sdf=sdf, stats=stats)

    @staticmethod
    def _deduplicate(
        x: np.ndarray,
        y: np.ndarray,
        sdf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x) == 0:
            return x, y, sdf
        coords = np.column_stack([x, y])
        _, unique_idx = np.unique(np.round(coords, decimals=12), axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        return x[unique_idx], y[unique_idx], sdf[unique_idx]


class FlowRAR:
    """Прежний residual-based RAR, опционально исключающий пристенную зону."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def generate(self, n_points: int) -> Optional[RARGenerationResult]:
        tr = self.trainer
        if n_points <= 0 or len(tr.point_residual) == 0:
            return None

        dh = tr.data_handler
        ds = dh.dataset
        coords = _dataset_xy(ds)
        residual = tr.point_residual.detach().cpu().numpy().reshape(-1).astype(np.float64)

        if len(residual) != len(coords):
            logger.warning(
                "Flow RAR: размер residual (%d) не совпадает с датасетом (%d)",
                len(residual),
                len(coords),
            )
            return None

        if ds.boundary_types is not None and "interior" in ds.type_to_idx:
            interior_mask = (ds.boundary_types[:, ds.type_to_idx["interior"]] > 0.5).cpu().numpy()
        else:
            interior_mask = ~getattr(ds, "on_boundary", torch.zeros(ds.n_points, dtype=torch.bool)).cpu().numpy()

        finite = np.isfinite(residual) & (residual > 0.0) & interior_mask

        delta_norm = _estimate_delta_norm(dh)
        exclude_sdf_delta = self._exclude_sdf_delta(delta_norm)
        if exclude_sdf_delta is not None:
            sdf_existing = ds.sdf.cpu().numpy().reshape(-1).astype(np.float64)
            finite &= sdf_existing >= exclude_sdf_delta * float(delta_norm)

        if not np.any(finite):
            logger.info("Flow RAR: нет положительной невязки в допустимой flow-зоне.")
            return None

        source_idx_all = np.where(finite)[0]
        source_res_all = residual[source_idx_all]
        q = min(max(float(tr.rar_error_quantile), 0.0), 0.99)
        threshold = float(np.quantile(source_res_all, q))
        source_idx = source_idx_all[source_res_all >= threshold]
        source_res = residual[source_idx]
        if len(source_idx) == 0:
            return None

        weights = np.maximum(source_res, 0.0) + 1e-30
        weights = np.power(weights, max(float(tr.rar_residual_power), 1e-6))
        weights = weights / weights.sum()

        existing_tree = cKDTree(coords)
        k = min(2, len(coords))
        nn_dist, _ = existing_tree.query(coords, k=k)
        if np.ndim(nn_dist) > 1:
            local_h = nn_dist[:, 1]
        else:
            local_h = np.full(len(coords), float(np.median(nn_dist)))
        positive_h = local_h[local_h > 1e-12]
        median_h = float(np.median(positive_h)) if len(positive_h) else 1e-3
        local_h = np.where(local_h > 1e-12, local_h, median_h)

        domain_norm = _domain_norm(dh, coords)
        foil_norm = _foil_norm(dh)
        wall_tree = _build_wall_tree(coords, ds)

        accepted: List[np.ndarray] = []
        accepted_sdf: List[float] = []
        accepted_arr = np.empty((0, 2), dtype=np.float64)
        max_attempts = max(n_points * int(tr.rar_candidate_multiplier) * 20, n_points)
        attempts = 0

        while len(accepted) < n_points and attempts < max_attempts:
            need = n_points - len(accepted)
            batch_n = min(max(need * int(tr.rar_candidate_multiplier), 64), 4096)
            attempts += batch_n

            src = np.random.choice(source_idx, size=batch_n, replace=True, p=weights)
            radius = float(tr.rar_jitter_scale) * local_h[src] * np.sqrt(np.random.random(batch_n))
            theta = 2.0 * np.pi * np.random.random(batch_n)
            cand = coords[src] + np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

            valid = valid_interior_mask(cand, domain_norm, foil_norm)
            if not np.any(valid):
                continue
            cand = cand[valid]
            src = src[valid]

            min_dist = float(tr.rar_min_distance_factor) * local_h[src]
            dist_existing, _ = existing_tree.query(cand, k=1, workers=-1)
            valid = dist_existing >= min_dist
            if not np.any(valid):
                continue
            cand = cand[valid]
            min_dist = min_dist[valid]

            if len(accepted_arr) > 0:
                dist_accepted, _ = cKDTree(accepted_arr).query(cand, k=1, workers=-1)
                valid = dist_accepted >= min_dist
                if not np.any(valid):
                    continue
                cand = cand[valid]
                min_dist = min_dist[valid]

            if wall_tree is not None:
                sdf_vals, _ = wall_tree.query(cand, k=1, workers=-1)
                min_sdf = float(tr.rar_min_boundary_distance_factor) * median_h
                if exclude_sdf_delta is not None and delta_norm is not None:
                    min_sdf = max(min_sdf, exclude_sdf_delta * float(delta_norm))
                valid = sdf_vals >= min_sdf
                if not np.any(valid):
                    continue
                cand = cand[valid]
                min_dist = min_dist[valid]
                sdf_vals = sdf_vals[valid]
            else:
                sdf_vals = np.zeros(len(cand), dtype=np.float64)

            batch_pts: List[np.ndarray] = []
            batch_sdf: List[float] = []
            for xy, sdf, d_min in zip(cand, sdf_vals, min_dist):
                if len(accepted) + len(batch_pts) >= n_points:
                    break
                if batch_pts:
                    batch_arr = np.asarray(batch_pts, dtype=np.float64)
                    if float(np.linalg.norm(batch_arr - xy, axis=1).min()) < float(d_min):
                        continue
                batch_pts.append(np.asarray(xy, dtype=np.float64))
                batch_sdf.append(float(sdf))

            if batch_pts:
                accepted.extend(batch_pts)
                accepted_sdf.extend(batch_sdf)
                accepted_arr = np.vstack([accepted_arr, np.asarray(batch_pts, dtype=np.float64)])

        if not accepted:
            logger.warning("Flow RAR: не удалось сгенерировать допустимые новые точки.")
            return None

        pts = np.vstack(accepted)
        sdf = np.asarray(accepted_sdf, dtype=np.float64)
        stats = {
            "requested": int(n_points),
            "accepted": int(len(pts)),
            "source_count": int(len(source_idx)),
            "quantile": float(q),
            "source_residual_min": float(source_res.min()),
            "source_residual_max": float(source_res.max()),
            "exclude_sdf_delta": exclude_sdf_delta,
        }
        logger.info(
            "Flow RAR: source=%d выше %.0f%% квантиля, residual=[%.3e, %.3e], accepted=%d",
            len(source_idx),
            q * 100.0,
            float(source_res.min()),
            float(source_res.max()),
            len(pts),
        )
        return RARGenerationResult(x=pts[:, 0], y=pts[:, 1], sdf=sdf, stats=stats)

    def _exclude_sdf_delta(self, delta_norm: Optional[float]) -> Optional[float]:
        cfg = self.trainer.config.get("flow_rar", {}) or {}
        value = cfg.get("exclude_sdf_delta", None)
        if value is None and bool((self.trainer.config.get("boundary_rar", {}) or {}).get("enabled", False)):
            value = 3.0
        if value is None or delta_norm is None:
            return None
        value_f = float(value)
        return value_f if value_f > 0.0 else None


class BoundaryLayerRAR:
    """RAR, который строит точки в пограничном слое по eta=SDF/delta."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def generate(self, n_points: int) -> Optional[RARGenerationResult]:
        tr = self.trainer
        cfg = tr.config.get("boundary_rar", {}) or {}
        if n_points <= 0:
            return None

        dh = tr.data_handler
        delta_norm = self._delta_norm(cfg)
        if delta_norm is None:
            logger.warning("Boundary-Layer RAR: не удалось определить delta/c.")
            return None

        airfoil = _foil_norm(dh)
        if airfoil is None or len(airfoil) < 2:
            logger.warning("Boundary-Layer RAR: нет координат профиля для нормалей.")
            return None

        eta_levels = _positive_array(cfg.get("eta_levels", DEFAULT_ETA_LEVELS))
        eta_max = float(cfg.get("delta_max", cfg.get("eta_max", max(eta_levels))))
        eta_levels = eta_levels[eta_levels <= eta_max + 1e-12]
        if len(eta_levels) == 0:
            logger.warning("Boundary-Layer RAR: eta_levels пуст после фильтра delta_max.")
            return None

        contour_samples = max(8, int(cfg.get("contour_samples", 400)))
        points_wall, tangents = resample_polyline_by_arclength(
            airfoil,
            contour_samples,
            random_offset=bool(cfg.get("randomize_contour_offset", True)),
        )
        normals = choose_outward_normals(points_wall, tangents, dh, delta_norm, eta_levels)

        candidates, contour_idx, eta_idx = self._build_candidates(
            points_wall,
            normals,
            eta_levels,
            delta_norm,
        )
        if len(candidates) == 0:
            return None

        candidates, contour_idx, eta_idx, sdf_vals = self._filter_candidates(
            candidates,
            contour_idx,
            eta_idx,
            eta_levels,
            delta_norm,
            cfg,
        )
        if len(candidates) == 0:
            logger.warning("Boundary-Layer RAR: все кандидаты отфильтрованы.")
            return None

        scores = self._score_candidates(candidates, sdf_vals, cfg)
        if len(scores) != len(candidates):
            logger.warning("Boundary-Layer RAR: score size mismatch.")
            return None

        selected = self._sample_candidates(
            scores,
            contour_idx,
            eta_idx,
            n_points,
            cfg,
        )
        if len(selected) == 0:
            logger.warning("Boundary-Layer RAR: вероятностный выбор не дал точек.")
            return None

        pts = candidates[selected]
        sdf = sdf_vals[selected]
        eta_selected = sdf / float(delta_norm)
        stats = {
            "requested": int(n_points),
            "candidate_count": int(len(candidates)),
            "accepted": int(len(pts)),
            "delta_norm": float(delta_norm),
            "delta_m": float(delta_norm * float(getattr(dh, "chord", 1.0))),
            "eta_min": float(np.min(eta_selected)),
            "eta_median": float(np.median(eta_selected)),
            "eta_max": float(np.max(eta_selected)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "eta_levels": [float(v) for v in eta_levels],
        }
        logger.info(
            "Boundary-Layer RAR: candidates=%d, accepted=%d, eta=[%.3g, %.3g, %.3g], score=[%.3e, %.3e]",
            len(candidates),
            len(pts),
            stats["eta_min"],
            stats["eta_median"],
            stats["eta_max"],
            stats["score_min"],
            stats["score_max"],
        )
        return RARGenerationResult(x=pts[:, 0], y=pts[:, 1], sdf=sdf, stats=stats)

    def _delta_norm(self, cfg: Dict[str, Any]) -> Optional[float]:
        if "delta_norm" in cfg:
            value = float(cfg["delta_norm"])
            return value if value > 0.0 else None
        if "delta_over_chord" in cfg:
            value = float(cfg["delta_over_chord"])
            return value if value > 0.0 else None
        delta_mode = str(cfg.get("delta_mode", "estimated")).lower()
        if delta_mode == "manual":
            value = float(cfg.get("delta", cfg.get("delta_m", 0.0)))
            chord = float(getattr(self.trainer.data_handler, "chord", 1.0))
            if value > 0.0 and chord > 0.0:
                return value / chord
            return None
        return _estimate_delta_norm(self.trainer.data_handler)

    def _build_candidates(
        self,
        wall_points: np.ndarray,
        normals: np.ndarray,
        eta_levels: np.ndarray,
        delta_norm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = eta_levels.reshape(1, -1, 1) * float(delta_norm)
        cand = wall_points[:, None, :] + normals[:, None, :] * offsets
        n_contour = len(wall_points)
        n_eta = len(eta_levels)
        contour_idx = np.repeat(np.arange(n_contour), n_eta)
        eta_idx = np.tile(np.arange(n_eta), n_contour)
        return cand.reshape(-1, 2), contour_idx, eta_idx

    def _filter_candidates(
        self,
        candidates: np.ndarray,
        contour_idx: np.ndarray,
        eta_idx: np.ndarray,
        eta_levels: np.ndarray,
        delta_norm: float,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dh = self.trainer.data_handler
        ds = dh.dataset
        coords = _dataset_xy(ds)
        domain_norm = _domain_norm(dh, coords)
        foil_norm = _foil_norm(dh)
        raw_count = int(len(candidates))
        valid = valid_interior_mask(candidates, domain_norm, foil_norm)
        domain_count = int(np.count_nonzero(valid))
        if not np.any(valid):
            logger.info(
                "Boundary-Layer RAR filter: raw=%d, geometry=0",
                raw_count,
            )
            return candidates[:0], contour_idx[:0], eta_idx[:0], np.empty(0, dtype=np.float64)

        candidates = candidates[valid]
        contour_idx = contour_idx[valid]
        eta_idx = eta_idx[valid]

        wall_tree = _build_wall_tree(coords, ds)
        if wall_tree is not None:
            sdf_vals, _ = wall_tree.query(candidates, k=1, workers=-1)
        else:
            sdf_vals = eta_levels[eta_idx] * float(delta_norm)

        eta_actual = sdf_vals / float(delta_norm)
        eta_max = float(cfg.get("delta_max", cfg.get("eta_max", max(eta_levels))))
        eta_min = float(cfg.get("eta_min", 0.0))
        valid = (eta_actual >= eta_min) & (eta_actual <= eta_max)
        eta_count = int(np.count_nonzero(valid))
        if not np.any(valid):
            logger.info(
                "Boundary-Layer RAR filter: raw=%d, geometry=%d, eta=0",
                raw_count,
                domain_count,
            )
            return candidates[:0], contour_idx[:0], eta_idx[:0], np.empty(0, dtype=np.float64)

        candidates = candidates[valid]
        contour_idx = contour_idx[valid]
        eta_idx = eta_idx[valid]
        sdf_vals = sdf_vals[valid]

        interior_coords = _interior_xy(ds, coords)
        existing_tree = cKDTree(interior_coords) if len(interior_coords) else None
        min_dist = float(cfg.get("min_distance_factor", 0.02)) * float(delta_norm)
        if min_dist > 0.0:
            if existing_tree is not None:
                dist_existing, _ = existing_tree.query(candidates, k=1, workers=-1)
                valid = dist_existing >= min_dist
                if not np.any(valid):
                    logger.info(
                        "Boundary-Layer RAR filter: raw=%d, geometry=%d, eta=%d, min_distance=0 "
                        "(min_distance_factor=%.3g)",
                        raw_count,
                        domain_count,
                        eta_count,
                        min_dist / float(delta_norm) if delta_norm > 0.0 else 0.0,
                    )
                    return candidates[:0], contour_idx[:0], eta_idx[:0], np.empty(0, dtype=np.float64)
                candidates = candidates[valid]
                contour_idx = contour_idx[valid]
                eta_idx = eta_idx[valid]
                sdf_vals = sdf_vals[valid]

        logger.info(
            "Boundary-Layer RAR filter: raw=%d, geometry=%d, eta=%d, min_distance=%d "
            "(min_distance_factor=%.3g)",
            raw_count,
            domain_count,
            eta_count,
            len(candidates),
            min_dist / float(delta_norm) if delta_norm > 0.0 else 0.0,
        )
        logger.info(
            "Boundary-Layer RAR eta candidates after filter: %s",
            _format_eta_counts(eta_idx, eta_levels),
        )
        return candidates, contour_idx, eta_idx, sdf_vals

    def _score_candidates(
        self,
        candidates: np.ndarray,
        sdf_vals: np.ndarray,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        tr = self.trainer
        model = tr.model
        loss_computer = tr.loss_computer
        weights = cfg.get("error_weights", {}) or {}
        pde_components = loss_computer.pde_components
        w_cont = float(weights.get("cont", 1.0))
        w_1 = float(weights.get(pde_components[1], weights.get("mom_x", weights.get("mom_r", 3.0))))
        w_2 = float(weights.get(pde_components[2], weights.get("mom_y", weights.get("mom_z", 3.0))))
        chunk_size = max(16, int(cfg.get("score_batch_size", 2048)))

        was_training = model.training
        requires_grad_state = [p.requires_grad for p in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        scores: List[np.ndarray] = []
        try:
            for start in range(0, len(candidates), chunk_size):
                stop = min(start + chunk_size, len(candidates))
                xy = candidates[start:stop]
                sdf = sdf_vals[start:stop]
                with torch.enable_grad():
                    x = torch.tensor(
                        xy[:, 0],
                        device=tr.device,
                        dtype=tr.dtype,
                    ).view(-1, 1).requires_grad_(True)
                    y = torch.tensor(
                        xy[:, 1],
                        device=tr.device,
                        dtype=tr.dtype,
                    ).view(-1, 1).requires_grad_(True)
                    sdf_t = torch.tensor(
                        sdf,
                        device=tr.device,
                        dtype=tr.dtype,
                    ).view(-1, 1)
                    bt = _interior_boundary_types(tr, len(x))
                    inp = loss_computer.build_input_tensor(x, y, sdf_t, bt)
                    out = model(inp)
                    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                    res = loss_computer.compute_pde_residuals(u, v, p, x, y)
                    score = (
                        w_cont * res["cont"]
                        + w_1 * res[pde_components[1]]
                        + w_2 * res[pde_components[2]]
                    )
                    scores.append(
                        torch.nan_to_num(
                            score.detach().flatten(),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).cpu().numpy().astype(np.float64)
                    )
        finally:
            for param, requires_grad in zip(model.parameters(), requires_grad_state):
                param.requires_grad_(requires_grad)
            if was_training:
                model.train()

        if not scores:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(scores)

    def _sample_candidates(
        self,
        scores: np.ndarray,
        contour_idx: np.ndarray,
        eta_idx: np.ndarray,
        n_points: int,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        if len(scores) == 0:
            return np.empty(0, dtype=np.int64)

        alpha = max(float(cfg.get("alpha", 1.5)), 1e-6)
        eps = float(cfg.get("probability_eps", 1e-12))
        uniform_contour_weight = min(max(float(cfg.get("uniform_contour_weight", 0.5)), 0.0), 1.0)
        powered = np.power(np.maximum(scores, 0.0) + eps, alpha)

        unique_contours, inverse = np.unique(contour_idx, return_inverse=True)
        contour_sum = np.bincount(inverse, weights=powered, minlength=len(unique_contours))
        contour_count = np.bincount(inverse, minlength=len(unique_contours))
        contour_mean = contour_sum / np.maximum(contour_count, 1)

        if np.isfinite(contour_mean).all() and contour_mean.sum() > 0.0:
            contour_error_prob = contour_mean / contour_mean.sum()
        else:
            contour_error_prob = np.full(len(unique_contours), 1.0 / len(unique_contours))
        contour_uniform = np.full(len(unique_contours), 1.0 / len(unique_contours))
        contour_prob = (
            uniform_contour_weight * contour_uniform
            + (1.0 - uniform_contour_weight) * contour_error_prob
        )

        size = min(int(n_points), len(scores))
        available: Dict[int, List[int]] = {
            ci: np.where(inverse == ci)[0].astype(np.int64).tolist()
            for ci in range(len(unique_contours))
        }
        selected: List[int] = []

        while len(selected) < size and available:
            active_contours = np.array(sorted(available.keys()), dtype=np.int64)
            p_contour = contour_prob[active_contours].astype(np.float64, copy=True)
            if not np.isfinite(p_contour).all() or p_contour.sum() <= 0.0:
                p_contour = np.full(len(active_contours), 1.0 / len(active_contours))
            else:
                p_contour /= p_contour.sum()

            ci = int(np.random.choice(active_contours, p=p_contour))
            local_idx = np.asarray(available[ci], dtype=np.int64)
            p_height = powered[local_idx].astype(np.float64, copy=True)
            if not np.isfinite(p_height).all() or p_height.sum() <= 0.0:
                p_height = np.full(len(local_idx), 1.0 / len(local_idx))
            else:
                p_height /= p_height.sum()

            chosen = int(np.random.choice(local_idx, p=p_height))
            selected.append(chosen)
            available[ci].remove(chosen)
            if not available[ci]:
                del available[ci]

        return np.asarray(selected, dtype=np.int64)


def _estimate_delta_norm(data_handler: Any) -> Optional[float]:
    re = float(getattr(data_handler, "Re", 0.0))
    if re <= 0.0 or not np.isfinite(re):
        return None
    return float(5.0 / np.sqrt(re))


def _dataset_xy(dataset: Any) -> np.ndarray:
    return np.column_stack([
        dataset.x.cpu().numpy().reshape(-1),
        dataset.y.cpu().numpy().reshape(-1),
    ]).astype(np.float64)


def _interior_xy(dataset: Any, coords: np.ndarray) -> np.ndarray:
    if dataset.boundary_types is not None and "interior" in dataset.type_to_idx:
        mask = (dataset.boundary_types[:, dataset.type_to_idx["interior"]] > 0.5).cpu().numpy()
    else:
        mask = ~getattr(dataset, "on_boundary", torch.zeros(dataset.n_points, dtype=torch.bool)).cpu().numpy()
    return coords[mask]


def _domain_norm(data_handler: Any, coords: np.ndarray) -> np.ndarray:
    chord = float(getattr(data_handler, "chord", 1.0))
    domain = getattr(data_handler, "mesh_domain_coords", None)
    if domain is not None and len(domain) >= 3 and chord > 0.0:
        return np.asarray(domain, dtype=np.float64) / chord
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ],
        dtype=np.float64,
    )


def _foil_norm(data_handler: Any) -> Optional[np.ndarray]:
    chord = float(getattr(data_handler, "chord", 1.0))
    foil = getattr(data_handler, "mesh_airfoil_coords", None)
    if foil is None or len(foil) < 2 or chord <= 0.0:
        return None
    arr = np.asarray(foil, dtype=np.float64) / chord
    return _remove_adjacent_duplicates(arr)


def _build_wall_tree(coords: np.ndarray, dataset: Any) -> Optional[cKDTree]:
    if dataset.boundary_types is None or "wall" not in dataset.type_to_idx:
        return None
    wall_mask = (dataset.boundary_types[:, dataset.type_to_idx["wall"]] > 0.5).cpu().numpy()
    if not np.any(wall_mask):
        return None
    return cKDTree(coords[wall_mask])


def _interior_boundary_types(trainer: Any, n_points: int) -> Optional[torch.Tensor]:
    ds = trainer.data_handler.dataset
    if ds.boundary_types is None or ds.boundary_types.shape[1] == 0:
        return None
    bt = torch.zeros(
        (n_points, ds.boundary_types.shape[1]),
        device=trainer.device,
        dtype=trainer.dtype,
    )
    interior_idx = ds.type_to_idx.get("interior")
    if interior_idx is not None and interior_idx < bt.shape[1]:
        bt[:, interior_idx] = 1.0
    return bt


def valid_interior_mask(
    xy: np.ndarray,
    domain_norm: Optional[np.ndarray],
    foil_norm: Optional[np.ndarray],
) -> np.ndarray:
    valid = np.isfinite(xy).all(axis=1)
    if domain_norm is not None and len(domain_norm) >= 3:
        try:
            from matplotlib.path import Path
            valid &= Path(domain_norm).contains_points(xy)
        except Exception:
            from shapely.geometry import Point, Polygon
            domain_poly = Polygon(domain_norm)
            if not domain_poly.is_valid:
                domain_poly = domain_poly.buffer(0)
            valid &= np.array(
                [domain_poly.contains(Point(float(p[0]), float(p[1]))) for p in xy],
                dtype=bool,
            )

    if foil_norm is not None and len(foil_norm) >= 4 and np.allclose(foil_norm[0], foil_norm[-1]):
        try:
            from matplotlib.path import Path
            valid &= ~Path(foil_norm).contains_points(xy)
        except Exception:
            from shapely.geometry import Point, Polygon
            foil_poly = Polygon(foil_norm)
            if not foil_poly.is_valid:
                foil_poly = foil_poly.buffer(0)
            valid &= ~np.array(
                [foil_poly.contains(Point(float(p[0]), float(p[1]))) for p in xy],
                dtype=bool,
            )
    return valid


def resample_polyline_by_arclength(
    points: np.ndarray,
    n_samples: int,
    *,
    random_offset: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = _remove_adjacent_duplicates(np.asarray(points, dtype=np.float64))
    closed = len(pts) >= 3 and np.allclose(pts[0], pts[-1])
    if closed:
        pts_work = pts[:-1]
        seg_start = pts_work
        seg_end = np.roll(pts_work, -1, axis=0)
    else:
        pts_work = pts
        seg_start = pts_work[:-1]
        seg_end = pts_work[1:]

    seg = seg_end - seg_start
    seg_len = np.linalg.norm(seg, axis=1)
    valid = seg_len > 1e-14
    seg_start = seg_start[valid]
    seg_end = seg_end[valid]
    seg = seg[valid]
    seg_len = seg_len[valid]
    if len(seg_len) == 0:
        raise ValueError("Контур профиля не содержит ненулевых сегментов")

    total = float(seg_len.sum())
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    if random_offset:
        offset = np.random.random()
    else:
        offset = 0.5
    sample_s = (np.arange(n_samples, dtype=np.float64) + offset) / float(n_samples) * total
    if not closed:
        sample_s = np.clip(sample_s, 0.0, np.nextafter(total, 0.0))

    seg_idx = np.searchsorted(cumulative, sample_s, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    local = (sample_s - cumulative[seg_idx]) / seg_len[seg_idx]
    sampled = seg_start[seg_idx] + local[:, None] * seg[seg_idx]
    tangents = seg[seg_idx] / seg_len[seg_idx, None]
    return sampled, tangents


def choose_outward_normals(
    wall_points: np.ndarray,
    tangents: np.ndarray,
    data_handler: Any,
    delta_norm: float,
    eta_levels: np.ndarray,
) -> np.ndarray:
    normal_a = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    normal_b = -normal_a
    coords = _dataset_xy(data_handler.dataset)
    domain_norm = _domain_norm(data_handler, coords)
    foil_norm = _foil_norm(data_handler)
    test_distance = max(float(np.min(eta_levels)) * float(delta_norm), 1e-6)
    valid_a = valid_interior_mask(wall_points + normal_a * test_distance, domain_norm, foil_norm)
    valid_b = valid_interior_mask(wall_points + normal_b * test_distance, domain_norm, foil_norm)

    normals = np.empty_like(normal_a)
    centroid = np.mean(wall_points, axis=0)
    away_a = np.sum(normal_a * (wall_points - centroid), axis=1) >= 0.0
    for i in range(len(wall_points)):
        if valid_a[i] and not valid_b[i]:
            normals[i] = normal_a[i]
        elif valid_b[i] and not valid_a[i]:
            normals[i] = normal_b[i]
        elif valid_a[i] and valid_b[i]:
            normals[i] = normal_a[i] if away_a[i] else normal_b[i]
        else:
            normals[i] = normal_a[i] if away_a[i] else normal_b[i]
    norm = np.linalg.norm(normals, axis=1)
    norm = np.where(norm > 1e-14, norm, 1.0)
    return normals / norm[:, None]


def _remove_adjacent_duplicates(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-14
    return points[keep]


def _positive_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) == 0:
        return np.asarray(DEFAULT_ETA_LEVELS, dtype=np.float64)
    return np.unique(np.sort(arr))


def _format_eta_counts(eta_idx: np.ndarray, eta_levels: np.ndarray) -> str:
    if len(eta_idx) == 0:
        return "empty"
    counts = np.bincount(eta_idx.astype(np.int64), minlength=len(eta_levels))
    parts = [
        f"{float(eta_levels[i]):.3g}:{int(count)}"
        for i, count in enumerate(counts[: len(eta_levels)])
        if count > 0
    ]
    return ", ".join(parts) if parts else "empty"
