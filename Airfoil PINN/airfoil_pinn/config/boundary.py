"""Equation-dependent configuration of the stationary airfoil wall."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


def resolve_wall_condition(
    system: str,
    wall: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind wall physics to the PDE, overriding stale YAML/UI values.

    Preserve the selected enforcement mode: hard no-slip for Navier-Stokes
    and hard normal projection for Euler use different output transforms.
    """
    resolved = dict(wall)
    if system == "incompressible_navier_stokes":
        resolved["kind"] = "no_slip"
        resolved.setdefault("enforcement", "soft")
    elif system == "euler":
        resolved["kind"] = "slip_wall"
        resolved.setdefault("enforcement", "soft")
    return resolved


def normalize_wall_condition(config: MutableMapping[str, Any]) -> None:
    """Store the effective wall condition for display, saving and training.

    Malformed sections are left to schema validation.
    """
    pde = config.get("pde", {})
    if not isinstance(pde, Mapping):
        return
    system = pde.get("system", "incompressible_navier_stokes")
    if system not in ("incompressible_navier_stokes", "euler"):
        return
    boundary = config.setdefault("boundary_conditions", {})
    if not isinstance(boundary, MutableMapping):
        return
    wall = boundary.get("wall", {})
    if isinstance(wall, Mapping):
        boundary["wall"] = resolve_wall_condition(system, wall)
