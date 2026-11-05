"""Boundary-condition configuration and differentiable output transforms."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from airfoil_pinn.config.boundary import resolve_wall_condition


_BOUNDARY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "wall": {"kind": "no_slip", "enforcement": "soft"},
    "inlet": {"kind": "freestream", "enforcement": "soft"},
    "farfield": {"kind": "freestream", "enforcement": "soft"},
    "outlet": {"kind": "pressure_outlet", "enforcement": "soft"},
    "velocity_dirichlet": {
        "kind": "freestream",
        "enforcement": "soft",
    },
    "slip_wall": {"kind": "slip_wall", "enforcement": "soft"},
    "symmetry": {"kind": "symmetry", "enforcement": "soft"},
    "axis_symmetry": {"kind": "axis_symmetry", "enforcement": "soft"},
}


def resolve_boundary_conditions(
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return a complete per-boundary configuration.

    A legacy global ``type=soft`` is treated as the default for old files.
    Global ``type=hard`` is rejected by static validation and is deliberately
    not translated into a mathematical output transform.
    """
    section = config.get("boundary_conditions", {})
    section = section if isinstance(section, Mapping) else {}
    resolved = {
        name: dict(values)
        for name, values in _BOUNDARY_DEFAULTS.items()
    }
    legacy_slip = section.get("slip")
    if isinstance(legacy_slip, Mapping) and "slip_wall" not in section:
        resolved["slip_wall"].update(dict(legacy_slip))
    for name, value in section.items():
        if name == "type" or not isinstance(value, Mapping):
            continue
        current = resolved.setdefault(
            str(name),
            {"kind": str(name), "enforcement": "soft"},
        )
        current.update(dict(value))
    resolved["wall"] = resolve_wall_condition(
        config.get("pde", {}).get("system", "incompressible_navier_stokes"),
        resolved["wall"],
    )
    # Boundary labels from existing point files may still be named ``slip``.
    # They use the canonical slip-wall rule without creating another physical
    # boundary condition.
    resolved["slip"] = dict(resolved["slip_wall"])
    return resolved


def distance_mask(
    phi_wall: torch.Tensor,
    length_scale: float,
) -> torch.Tensor:
    """Map a non-negative wall distance to a bounded hard-BC multiplier."""
    if not torch.is_tensor(phi_wall):
        raise TypeError("phi_wall must be a torch.Tensor")
    if length_scale <= 0.0:
        raise ValueError("distance_scale must be positive")
    return phi_wall / (phi_wall + length_scale)


class BoundaryOutputTransform(nn.Module):
    """Transform raw network outputs into fields satisfying supported hard BC.

    No-slip scales both velocity components by a wall-distance mask. Slip
    scales only the normal component, preserving tangential velocity.
    Pressure remains unconstrained.
    """

    def __init__(
        self,
        geometry: Optional[nn.Module],
        *,
        wall_kind: str,
        wall_enforcement: str,
        distance_scale: float,
    ) -> None:
        super().__init__()
        if wall_enforcement not in {"soft", "hard"}:
            raise ValueError(
                f"Unknown wall enforcement: {wall_enforcement}"
            )
        if wall_enforcement == "hard" and wall_kind not in {"no_slip", "slip_wall"}:
            raise ValueError(
                "Hard wall output transform requires kind=no_slip or slip_wall"
            )
        if distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive")

        self.geometry = geometry
        self.wall_kind = str(wall_kind)
        self.wall_enforcement = str(wall_enforcement)
        self.distance_scale = float(distance_scale)

    @property
    def enabled(self) -> bool:
        """Whether this transform changes the velocity outputs."""
        return self.wall_enforcement == "hard"

    def wall_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the differentiable wall distance at arbitrary points."""
        if self.geometry is None:
            raise RuntimeError(
                "Hard wall requires an initialized differentiable SDF geometry"
            )
        if x.shape != y.shape:
            raise ValueError("x and y must have identical shapes")
        if hasattr(self.geometry, "wall_distance"):
            return self.geometry.wall_distance(x, y)
        return self.geometry(torch.cat((x, y), dim=1))

    def forward(
        self,
        raw_u: torch.Tensor,
        raw_v: torch.Tensor,
        raw_p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        sdf: Optional[torch.Tensor] = None,
        flow: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply no-slip or impermeability inside the autograd graph."""
        del flow  # Reserved for a future moving-wall lifting.
        if self.wall_enforcement == "soft":
            return raw_u, raw_v, raw_p

        phi_wall = self.wall_distance(x, y) if sdf is None else sdf
        if phi_wall.shape != raw_u.shape:
            raise ValueError(
                "wall distance and network outputs must have identical shapes"
            )
        mask = distance_mask(phi_wall, self.distance_scale)

        if self.wall_kind == "slip_wall":
            if self.geometry is None or not hasattr(self.geometry, "wall_normal_extension"):
                raise RuntimeError("Hard slip wall requires a geometry normal field")
            # The normal extension is differentiated along with the fields.
            # Only the discrete nearest-segment lookup is outside autograd.
            normal = self.geometry.wall_normal_extension(torch.cat((x, y), dim=1))
            nx, ny = normal[:, 0:1], normal[:, 1:2]
            raw_normal = raw_u * nx + raw_v * ny
            correction = (1.0 - mask) * raw_normal
            return raw_u - correction * nx, raw_v - correction * ny, raw_p

        wall_lifting_u = torch.zeros_like(raw_u)
        wall_lifting_v = torch.zeros_like(raw_v)
        u = wall_lifting_u + mask * raw_u
        v = wall_lifting_v + mask * raw_v
        return u, v, raw_p

    def contract(self) -> Dict[str, Any]:
        """Return the graph-defining checkpoint metadata."""
        wall: Dict[str, Any] = {
            "kind": self.wall_kind,
            "mode": self.wall_enforcement,
        }
        if self.enabled:
            wall.update(
                {
                    "transform": (
                        "normalized_distance_normal_projection"
                        if self.wall_kind == "slip_wall"
                        else "normalized_distance"
                    ),
                    "distance_scale": self.distance_scale,
                }
            )
            if self.wall_kind == "slip_wall":
                wall["normal_field"] = "closest_segment_or_endpoint"
        return {"wall": wall}
