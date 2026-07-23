"""Differentiable distance fields for polygonal airfoil geometries.

The training PDE differentiates network outputs with respect to normalized
coordinates.  Consequently, an SDF used as a network feature must be computed
from those same coordinate tensors inside the autograd graph.  Loading a
precomputed distance and merely enabling ``requires_grad`` would model it as an
independent variable and omit the chain-rule terms.

For the exterior fluid domain the signed distance is positive, so the unsigned
distance to the airfoil polyline equals the exterior branch of the SDF.  This
module intentionally implements that branch.  It is continuous and piecewise
smooth, but not twice differentiable on the medial axis or where the closest
segment changes.  Callers can use the nearest/second-nearest gap diagnostics to
exclude a narrow tie region from second-order PDE residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


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
        polyline: Ordered ``(N, 2)`` coordinates in chord-normalized units.
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

    def nearest_two(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return distances to the nearest and second-nearest segments.

        The second value is ``inf`` for a single-segment open polyline.
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

        for start in range(0, self.segment_count, self.segment_chunk_size):
            stop = min(start + self.segment_chunk_size, self.segment_count)
            squared = self._squared_distances(flat, start, stop)
            candidates = torch.cat((best, squared), dim=1)
            best = torch.topk(
                candidates,
                k=2,
                dim=1,
                largest=False,
                sorted=True,
            ).values

        distances = torch.sqrt(torch.clamp_min(best, 0.0))
        nearest = distances[:, 0].reshape(*original_shape, 1)
        second = distances[:, 1].reshape(*original_shape, 1)
        return nearest, second

    @staticmethod
    def tie_mask(
        nearest: torch.Tensor,
        second_nearest: torch.Tensor,
        *,
        absolute_tolerance: float = 1e-7,
        relative_tolerance: float = 1e-5,
    ) -> torch.Tensor:
        """Identify points near a closest-segment switch/medial axis."""
        if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
            raise ValueError("tie tolerances must be non-negative")
        gap = second_nearest - nearest
        threshold = absolute_tolerance + relative_tolerance * torch.maximum(
            nearest.abs(),
            second_nearest.abs(),
        )
        return torch.isfinite(gap) & (gap <= threshold)

    def diagnostics(
        self,
        points: torch.Tensor,
        *,
        absolute_tie_tolerance: float = 1e-7,
        relative_tie_tolerance: float = 1e-5,
    ) -> DistanceFieldDiagnostics:
        """Evaluate distance, Eikonal norm and tie fraction on sample points."""
        sample = points.detach().clone().requires_grad_(True)
        nearest, second = self.nearest_two(sample)
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
