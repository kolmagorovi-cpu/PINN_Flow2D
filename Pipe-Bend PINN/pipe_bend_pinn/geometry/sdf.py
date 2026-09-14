"""Differentiable distance fields for polygonal pipe-bend wall geometries.

The module is ported from the Airfoil PINN package
(``airfoil_pinn/geometry/sdf.py``).  The training PDE differentiates network
outputs with respect to normalized coordinates.  Consequently, an SDF used as a
network feature must be computed from those same coordinate tensors inside the
autograd graph.  Loading a precomputed distance and merely enabling
``requires_grad`` would model it as an independent variable and omit the
chain-rule terms.

For the interior fluid domain of a pipe bend the signed distance is positive
towards the channel centre and zero on the walls, so the unsigned distance to
the wall polyline equals the interior branch of the SDF.  This module
intentionally implements that branch.  It is continuous and piecewise smooth,
but not twice differentiable on the medial axis or where the closest segment
changes.  Callers can use the nearest/second-nearest gap diagnostics to exclude
a narrow tie region from second-order PDE residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


def compute_wall_normals(
    points: np.ndarray,
    domain: np.ndarray,
    wall_mask: np.ndarray,
    *,
    probe_offset: float = 1e-4,
) -> np.ndarray:
    """Внутренние (в жидкость) единичные нормали к стенке для точек коллокации.

    Геометрия-агностичный расчёт: для каждой точки с ``wall_mask=True`` находится
    ближайший сегмент замкнутого контура ``domain`` (обе — в одинаковых единицах,
    для PINN — нормированных на pipe_width) и берётся нормаль к сегменту,
    направленная внутрь жидкости. Направление выбирается по ориентации контура
    (знаковая площадь) и проверяется пробой точки на малом смещении внутрь.
    Для точек вне стенки возвращается нулевой вектор.

    Args:
        points: массив (N, 2) координат точек коллокации.
        domain: массив (M, 2) вершин замкнутого контура области.
        wall_mask: булев массив (N,) — точки типа wall.
        probe_offset: шаг пробной точки для проверки «внутрь жидкости».

    Returns:
        массив (N, 2) нормалей (для не-wall точек — (0, 0)).
    """
    out = np.zeros((len(points), 2), dtype=np.float64)
    idx = np.flatnonzero(wall_mask)
    if not len(idx):
        return out
    # Замкнутый контур и его рёбра
    A = domain
    B = np.roll(domain, -1, axis=0)
    ab = B - A
    seg_len = np.linalg.norm(ab, axis=1)
    t = ab / np.maximum(seg_len[:, None], 1e-14)
    # Знак ориентации (shoelace): >0 — CCW, <0 — CW.
    x_, y_ = domain[:, 0], domain[:, 1]
    area2 = np.sum(x_ * np.roll(y_, -1) - np.roll(x_, -1) * y_)
    # Внутрь жидкости: при CCW — левая нормаль (-ty, tx), при CW — правая.
    base = np.column_stack([-t[:, 1], t[:, 0]])
    if area2 < 0:
        base = -base
    denom = np.sum(ab * ab, axis=1)
    import matplotlib.path as mplPath

    path = mplPath.Path(domain)
    for i in idx:
        p = points[i]
        tt = np.clip(np.sum((p - A) * ab, axis=1) / np.maximum(denom, 1e-14), 0.0, 1.0)
        proj = A + tt[:, None] * ab
        d2 = np.sum((proj - p) ** 2, axis=1)
        s = int(np.argmin(d2))
        n = base[s].copy()
        if not path.contains_point(p + n * probe_offset):
            n = -n
        out[i] = n
    return out


@dataclass(frozen=True)
class DistanceFieldDiagnostics:
    """Summary statistics for differentiability and geometric consistency."""

    point_count: int
    distance_min: float
    distance_max: float
    eikonal_mean: float
    eikonal_max_error: float
    tie_fraction: float

    def to_dict(self) -> Dict[str, float]:
        """Return a log/checkpoint-friendly representation."""
        return {
            "point_count": float(self.point_count),
            "distance_min": self.distance_min,
            "distance_max": self.distance_max,
            "eikonal_mean": self.eikonal_mean,
            "eikonal_max_error": self.eikonal_max_error,
            "tie_fraction": self.tie_fraction,
        }


class PolylineDistanceField(nn.Module):
    """Exact piecewise differentiable distance to normalized line segments.

    Args:
        polyline: Ordered ``(N, 2)`` coordinates in pipe-width-normalized units.
        closed: Add a final segment from the last point to the first when the
            input does not already contain it.
        segment_chunk_size: Maximum number of segments materialized in one
            vectorized distance block.  This bounds memory by approximately
            ``batch_size * segment_chunk_size``.
        degeneracy_tolerance: Segments no longer than this value are rejected.

    Segment coordinates are registered as buffers.  They move with
    :meth:`torch.nn.Module.to` but are never optimized.
    """

    def __init__(
        self,
        polyline: np.ndarray | torch.Tensor,
        *,
        closed: bool = True,
        segment_chunk_size: int = 256,
        degeneracy_tolerance: float = 1e-14,
    ) -> None:
        super().__init__()
        if segment_chunk_size <= 0:
            raise ValueError("segment_chunk_size must be a positive integer")
        if degeneracy_tolerance < 0.0:
            raise ValueError("degeneracy_tolerance must be non-negative")

        points = torch.as_tensor(polyline, dtype=torch.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError("polyline must have shape (N, 2) with N >= 2")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("polyline contains NaN or infinite coordinates")

        points = self._remove_adjacent_duplicates(
            points,
            tolerance=degeneracy_tolerance,
        )
        if closed and not torch.allclose(
            points[0],
            points[-1],
            rtol=0.0,
            atol=max(degeneracy_tolerance, 1e-14),
        ):
            points = torch.cat((points, points[0:1]), dim=0)

        starts = points[:-1]
        ends = points[1:]
        vectors = ends - starts
        squared_lengths = torch.sum(vectors * vectors, dim=1)
        valid = squared_lengths > degeneracy_tolerance**2
        if not bool(valid.any()):
            raise ValueError("polyline has no non-degenerate segments")
        starts = starts[valid]
        ends = ends[valid]
        vectors = vectors[valid]
        squared_lengths = squared_lengths[valid]

        self.register_buffer("segment_starts", starts)
        self.register_buffer("segment_ends", ends)
        self.register_buffer("segment_vectors", vectors)
        self.register_buffer("segment_squared_lengths", squared_lengths)
        self.segment_chunk_size = int(segment_chunk_size)
        self.closed = bool(closed)
        self.geometry_fingerprint = self._fingerprint(points)

    @staticmethod
    def _remove_adjacent_duplicates(
        points: torch.Tensor,
        *,
        tolerance: float,
    ) -> torch.Tensor:
        """Remove consecutive duplicate vertices without changing ordering."""
        delta = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
        keep = torch.ones(points.shape[0], dtype=torch.bool)
        keep[1:] = delta > tolerance
        result = points[keep]
        if result.shape[0] < 2:
            raise ValueError("polyline collapses to fewer than two vertices")
        return result

    @staticmethod
    def _fingerprint(points: torch.Tensor) -> str:
        """Hash normalized float64 vertices in their geometrical order."""
        contiguous = np.ascontiguousarray(
            points.detach().cpu().numpy(),
            dtype="<f8",
        )
        return hashlib.sha256(contiguous.tobytes()).hexdigest()

    @property
    def segment_count(self) -> int:
        """Return the number of valid polyline segments."""
        return int(self.segment_starts.shape[0])

    def extra_repr(self) -> str:
        """Describe the immutable geometry in module summaries."""
        return (
            f"segments={self.segment_count}, closed={self.closed}, "
            f"chunk={self.segment_chunk_size}, "
            f"fingerprint={self.geometry_fingerprint[:12]}"
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return minimum segment distance with shape ``(..., 1)``."""
        original_shape = self._validate_points(points)
        flat = points.reshape(-1, 2)
        minimum_squared: Optional[torch.Tensor] = None

        for start in range(0, self.segment_count, self.segment_chunk_size):
            stop = min(start + self.segment_chunk_size, self.segment_count)
            squared = self._squared_distances(flat, start, stop)
            chunk_minimum = torch.min(squared, dim=1).values
            minimum_squared = (
                chunk_minimum
                if minimum_squared is None
                else torch.minimum(minimum_squared, chunk_minimum)
            )

        if minimum_squared is None:
            raise RuntimeError("distance field contains no segments")
        distance = torch.sqrt(torch.clamp_min(minimum_squared, 0.0))
        return distance.reshape(*original_shape, 1)

    def wall_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return wall distance as a differentiable function of ``x``/``y``."""
        if x.shape != y.shape:
            raise ValueError("x and y must have identical shapes")
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x and y must have shape (N, 1)")
        return self(torch.cat((x, y), dim=1))

    def nearest_two(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return distances to the nearest and second-nearest segments."""
        nearest, second, _, _, _, _ = self.nearest_two_geometry(points)
        return nearest, second

    def nearest_two_geometry(
        self,
        points: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the two closest distances, projections and SDF gradients.

        Distance equality alone is insufficient to identify a non-smooth SDF
        switch.  Adjacent polyline segments often tie because both clamp to
        their shared vertex; in that case their closest projection and
        distance gradient are identical, so the minimum is locally smooth.

        The second distance is ``inf`` for a single-segment open polyline.
        Its projection and gradient are returned as ``NaN``.
        """
        original_shape = self._validate_points(points)
        flat = points.reshape(-1, 2)
        dtype = flat.dtype
        device = flat.device
        best = torch.full(
            (flat.shape[0], 2),
            float("inf"),
            dtype=dtype,
            device=device,
        )
        best_indices = torch.full(
            (flat.shape[0], 2),
            -1,
            dtype=torch.long,
            device=device,
        )

        for start in range(0, self.segment_count, self.segment_chunk_size):
            stop = min(start + self.segment_chunk_size, self.segment_count)
            squared = self._squared_distances(flat, start, stop)
            candidates = torch.cat((best, squared), dim=1)
            chunk_indices = torch.arange(
                start,
                stop,
                dtype=torch.long,
                device=device,
            ).expand(flat.shape[0], -1)
            candidate_indices = torch.cat(
                (best_indices, chunk_indices),
                dim=1,
            )
            best, positions = torch.topk(
                candidates,
                k=2,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = torch.gather(
                candidate_indices,
                dim=1,
                index=positions,
            )

        distances = torch.sqrt(torch.clamp_min(best, 0.0))
        nearest = distances[:, 0].reshape(*original_shape, 1)
        second = distances[:, 1].reshape(*original_shape, 1)
        safe_indices = torch.clamp_min(best_indices, 0)
        segment_starts = self.segment_starts[safe_indices].to(dtype=dtype)
        segment_vectors = self.segment_vectors[safe_indices].to(dtype=dtype)
        squared_lengths = self.segment_squared_lengths[safe_indices].to(
            dtype=dtype
        )
        relative = flat[:, None, :] - segment_starts
        parameters = torch.sum(
            relative * segment_vectors,
            dim=-1,
        ) / squared_lengths
        parameters = torch.clamp(parameters, min=0.0, max=1.0)
        projections = (
            segment_starts
            + parameters[:, :, None] * segment_vectors
        )
        finite = torch.isfinite(distances) & (best_indices >= 0)
        nan = torch.full_like(projections, float("nan"))
        projections = torch.where(finite[:, :, None], projections, nan)

        delta = flat[:, None, :] - projections
        safe_distances = torch.where(
            distances > torch.finfo(dtype).eps,
            distances,
            torch.ones_like(distances),
        )
        gradients = delta / safe_distances[:, :, None]
        gradients = torch.where(
            finite[:, :, None],
            gradients,
            nan,
        )

        nearest_projection = projections[:, 0].reshape(*original_shape, 2)
        second_projection = projections[:, 1].reshape(*original_shape, 2)
        nearest_gradient = gradients[:, 0].reshape(*original_shape, 2)
        second_gradient = gradients[:, 1].reshape(*original_shape, 2)
        return (
            nearest,
            second,
            nearest_projection,
            second_projection,
            nearest_gradient,
            second_gradient,
        )

    def surface_normals(self, points: torch.Tensor) -> torch.Tensor:
        """Return unit normals from the nearest SDF segment.

        At an exact surface point the unsigned distance is not differentiable
        across the wall and ``autograd(sqrt(distance_squared))`` may therefore
        produce ``NaN``.  The normal of the nearest segment is the one-sided
        surface limit of ``grad(distance)``.  Its sign is intentionally left
        determined by the polyline orientation; impermeability losses use
        ``(velocity dot normal)^2`` and are invariant to that sign.
        """
        original_shape = self._validate_points(points)
        flat = points.reshape(-1, 2)
        minimum_squared = torch.full(
            (flat.shape[0],),
            float("inf"),
            dtype=flat.dtype,
            device=flat.device,
        )
        nearest_indices = torch.zeros(
            flat.shape[0],
            dtype=torch.long,
            device=flat.device,
        )

        for start in range(0, self.segment_count, self.segment_chunk_size):
            stop = min(start + self.segment_chunk_size, self.segment_count)
            squared = self._squared_distances(flat, start, stop)
            chunk_minimum, chunk_indices = torch.min(squared, dim=1)
            replace = chunk_minimum < minimum_squared
            minimum_squared = torch.where(
                replace,
                chunk_minimum,
                minimum_squared,
            )
            nearest_indices = torch.where(
                replace,
                chunk_indices + start,
                nearest_indices,
            )

        tangents = self.segment_vectors[nearest_indices].to(dtype=flat.dtype)
        tangent_norm = torch.linalg.vector_norm(
            tangents,
            dim=1,
            keepdim=True,
        )
        normals = torch.stack((-tangents[:, 1], tangents[:, 0]), dim=1)
        normals = normals / tangent_norm
        return normals.reshape(*original_shape, 2)

    def tie_mask(
        self,
        nearest: torch.Tensor,
        second_nearest: Optional[torch.Tensor] = None,
        *,
        absolute_tolerance: float = 1e-12,
        relative_tolerance: float = 1e-10,
        projection_tolerance: float = 1e-7,
        gradient_tolerance: float = 1e-5,
        nearest_projection: Optional[torch.Tensor] = None,
        second_projection: Optional[torch.Tensor] = None,
        nearest_gradient: Optional[torch.Tensor] = None,
        second_gradient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Identify genuine closest-segment switches/medial-axis points.

        The preferred call is ``field.tie_mask(points)``.  It first compares
        the two closest distances and then verifies that both their closest
        projections and their SDF gradients differ.  Equal distances caused
        by adjacent segments clamping to one shared vertex are therefore not
        misclassified as ties.

        Passing ``(nearest, second_nearest)`` retains the old conservative
        distance-only check for compatibility with external diagnostic code.
        """
        if (
            absolute_tolerance < 0.0
            or relative_tolerance < 0.0
            or projection_tolerance < 0.0
            or gradient_tolerance < 0.0
        ):
            raise ValueError("tie tolerances must be non-negative")
        if second_nearest is None:
            (
                nearest_distance,
                second_distance,
                nearest_projection,
                second_projection,
                nearest_gradient,
                second_gradient,
            ) = self.nearest_two_geometry(nearest)
        else:
            nearest_distance = nearest
            second_distance = second_nearest
            geometry_values = (
                nearest_projection,
                second_projection,
                nearest_gradient,
                second_gradient,
            )
            if any(value is None for value in geometry_values) and not all(
                value is None for value in geometry_values
            ):
                raise ValueError(
                    "all projection/gradient tensors must be supplied "
                    "together"
                )

        gap = second_distance - nearest_distance
        threshold = absolute_tolerance + relative_tolerance * torch.maximum(
            nearest_distance.abs(),
            second_distance.abs(),
        )
        distance_tie = torch.isfinite(gap) & (gap <= threshold)
        if nearest_projection is None:
            return distance_tie

        projection_gap = torch.linalg.vector_norm(
            second_projection - nearest_projection,
            dim=-1,
            keepdim=True,
        )
        projection_scale = torch.maximum(
            torch.linalg.vector_norm(
                nearest_projection,
                dim=-1,
                keepdim=True,
            ),
            torch.linalg.vector_norm(
                second_projection,
                dim=-1,
                keepdim=True,
            ),
        )
        projections_differ = projection_gap > (
            projection_tolerance
            + relative_tolerance * projection_scale
        )
        gradient_gap = torch.linalg.vector_norm(
            second_gradient - nearest_gradient,
            dim=-1,
            keepdim=True,
        )
        gradients_differ = gradient_gap > gradient_tolerance
        return distance_tie & projections_differ & gradients_differ

    def diagnostics(
        self,
        points: torch.Tensor,
        *,
        absolute_tie_tolerance: float = 1e-12,
        relative_tie_tolerance: float = 1e-10,
    ) -> DistanceFieldDiagnostics:
        """Evaluate distance, Eikonal norm and tie fraction on sample points."""
        sample = points.detach().clone().requires_grad_(True)
        (
            nearest,
            second,
            nearest_projection,
            second_projection,
            nearest_gradient,
            second_gradient,
        ) = self.nearest_two_geometry(sample)
        gradient = torch.autograd.grad(
            nearest,
            sample,
            grad_outputs=torch.ones_like(nearest),
            create_graph=False,
            retain_graph=False,
        )[0]
        eikonal = torch.linalg.vector_norm(gradient, dim=-1)
        ties = self.tie_mask(
            nearest,
            second,
            absolute_tolerance=absolute_tie_tolerance,
            relative_tolerance=relative_tie_tolerance,
            nearest_projection=nearest_projection,
            second_projection=second_projection,
            nearest_gradient=nearest_gradient,
            second_gradient=second_gradient,
        )
        finite_eikonal = eikonal[torch.isfinite(eikonal)]
        if finite_eikonal.numel() == 0:
            raise RuntimeError("SDF diagnostics produced no finite gradients")

        return DistanceFieldDiagnostics(
            point_count=int(sample.reshape(-1, 2).shape[0]),
            distance_min=float(torch.min(nearest).detach().cpu()),
            distance_max=float(torch.max(nearest).detach().cpu()),
            eikonal_mean=float(torch.mean(finite_eikonal).detach().cpu()),
            eikonal_max_error=float(
                torch.max(torch.abs(finite_eikonal - 1.0)).detach().cpu()
            ),
            tie_fraction=float(
                ties.to(dtype=torch.float64).mean().detach().cpu()
            ),
        )

    @torch.no_grad()
    def evaluate_numpy(
        self,
        points: np.ndarray,
        *,
        point_chunk_size: int = 8192,
    ) -> np.ndarray:
        """Evaluate a cache with the same segment geometry as online autograd.

        Args:
            points: ``(N, 2)`` query coordinates in chord-normalized units.
            point_chunk_size: Number of query points evaluated per forward
                call.  Segment chunking is controlled independently by the
                module constructor.
        """
        if point_chunk_size <= 0:
            raise ValueError("point_chunk_size must be a positive integer")
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if not np.isfinite(query).all():
            raise ValueError("points contain NaN or infinite coordinates")

        values = np.empty(query.shape[0], dtype=np.float64)
        geometry = self.segment_starts
        for start in range(0, query.shape[0], point_chunk_size):
            stop = min(start + point_chunk_size, query.shape[0])
            tensor = torch.as_tensor(
                query[start:stop],
                device=geometry.device,
                dtype=geometry.dtype,
            )
            values[start:stop] = (
                self(tensor)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float64, copy=False)
            )
        return values

    def _validate_points(self, points: torch.Tensor) -> torch.Size:
        """Validate query shape/device/dtype and return the leading shape."""
        if not torch.is_tensor(points):
            raise TypeError("points must be a torch.Tensor")
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        if not points.is_floating_point():
            raise TypeError("points must use a floating-point dtype")
        if points.device != self.segment_starts.device:
            raise ValueError(
                "points and SDF geometry must be on the same device; "
                f"received {points.device} and {self.segment_starts.device}"
            )
        return points.shape[:-1]

    def _squared_distances(
        self,
        points: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Compute point-to-segment squared distances for one segment chunk."""
        segment_starts = self.segment_starts[start:stop].to(dtype=points.dtype)
        segment_vectors = self.segment_vectors[start:stop].to(dtype=points.dtype)
        squared_lengths = self.segment_squared_lengths[start:stop].to(
            dtype=points.dtype
        )

        relative = points[:, None, :] - segment_starts[None, :, :]
        projection = torch.sum(
            relative * segment_vectors[None, :, :],
            dim=-1,
        ) / squared_lengths[None, :]
        projection = torch.clamp(projection, min=0.0, max=1.0)
        closest = (
            segment_starts[None, :, :]
            + projection[:, :, None] * segment_vectors[None, :, :]
        )
        delta = points[:, None, :] - closest
        return torch.sum(delta * delta, dim=-1)


def differentiable_sdf_for_interior(
    field: PolylineDistanceField,
    x: torch.Tensor,
    y: torch.Tensor,
    cached_sdf: torch.Tensor,
    interior_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compose SDF with coordinates only where PDE derivatives are required.

    Boundary points retain their exact cached values.  This avoids evaluating
    the square-root derivative at zero on the wall while preserving chain-rule
    terms for all interior points.  Indexed assignment is differentiable in
    PyTorch and creates a ``CopySlices`` node linking the selected values to
    ``x`` and ``y``.
    """
    if x.shape != y.shape or x.shape != cached_sdf.shape:
        raise ValueError("x, y and cached_sdf must have identical shapes")
    if interior_mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        mask = interior_mask.to(device=x.device, dtype=torch.bool)
        if mask.shape != x.shape:
            raise ValueError("interior_mask must have the same shape as x")

    result = cached_sdf.clone()
    flat_mask = mask.reshape(-1)
    if bool(flat_mask.any()):
        points = torch.cat((x, y), dim=1)
        online = field(points[flat_mask])
        result_flat = result.reshape(-1)
        result_flat[flat_mask] = online.reshape(-1)
        result = result_flat.reshape_as(cached_sdf)
    return result
