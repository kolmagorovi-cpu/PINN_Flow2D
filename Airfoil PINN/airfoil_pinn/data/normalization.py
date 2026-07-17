"""Shared helpers for geometry-based nondimensionalization."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import numpy as np


def build_geometry_reference(
    foil: Optional[np.ndarray],
    *,
    geom_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build the physical length reference stored with a sampled geometry."""
    if foil is None:
        return None

    coords = np.asarray(foil, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        return None
    coords = coords[:, :2]
    if not np.all(np.isfinite(coords)):
        return None

    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    chord = x_max - x_min
    if chord <= 0.0:
        return None

    return {
        "geom_name": str(geom_name) if geom_name else None,
        "length_scale": "chord",
        "coordinate_units": "m",
        "chord": chord,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": float(np.min(coords[:, 1])),
        "y_max": float(np.max(coords[:, 1])),
    }


def extract_geometry_reference(metadata: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Read a geometry reference from current or legacy sampler metadata."""
    if not metadata:
        return None

    explicit = metadata.get("geometry_reference")
    if isinstance(explicit, Mapping):
        reference = dict(explicit)
        chord = _positive_float(reference.get("chord"))
        if chord is not None:
            reference["chord"] = chord
            reference.setdefault("source", "geometry_reference")
            return reference

    normal_layer = metadata.get("normal_layer")
    if isinstance(normal_layer, Mapping):
        chord = _positive_float(normal_layer.get("chord"))
        if chord is not None:
            return {
                "geom_name": metadata.get("geom_name"),
                "length_scale": "chord",
                "coordinate_units": "m",
                "chord": chord,
                "source": "normal_layer.chord",
            }

    sdf_statistics = metadata.get("sdf_statistics")
    if isinstance(sdf_statistics, Mapping):
        chord = _positive_float(sdf_statistics.get("chord"))
        if chord is not None:
            return {
                "geom_name": metadata.get("geom_name"),
                "length_scale": "chord",
                "coordinate_units": "m",
                "chord": chord,
                "source": "sdf_statistics.chord",
            }
    return None


def load_companion_metadata(data_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load ``*_metadata.npy`` located next to a NumPy data artifact."""
    if not data_file:
        return None
    base, extension = os.path.splitext(str(data_file))
    metadata_file = (
        f"{base}_metadata.npy"
        if extension.lower() == ".npy"
        else f"{data_file}_metadata.npy"
    )
    if not os.path.exists(metadata_file):
        return None
    try:
        value = np.load(metadata_file, allow_pickle=True)
        metadata = value.item()
    except (OSError, ValueError, TypeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def compute_reynolds_number(
    velocity: float,
    chord: float,
    density: float,
    dynamic_viscosity: float,
) -> float:
    """Compute the chord Reynolds number from one physical reference set."""
    values = (velocity, chord, density, dynamic_viscosity)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Reynolds reference values must be finite")
    if velocity <= 0.0 or chord <= 0.0 or density <= 0.0 or dynamic_viscosity <= 0.0:
        raise ValueError("Velocity, chord, density and dynamic viscosity must be positive")
    return float(velocity) * float(chord) * float(density) / float(dynamic_viscosity)


def _positive_float(value: Any) -> Optional[float]:
    """Convert a metadata value to a finite positive float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 and np.isfinite(result) else None
