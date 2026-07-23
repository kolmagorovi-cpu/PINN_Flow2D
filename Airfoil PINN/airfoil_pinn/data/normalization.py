"""Physical reference scales and geometry nondimensionalization utilities.

The incompressible-flow pipeline uses one explicit CFD convention:

* lengths and signed/unsigned distances are divided by the airfoil chord;
* velocities are divided by the freestream velocity magnitude;
* pressure is represented by the pressure coefficient
  ``Cp = (p - p_inf) / (0.5 * rho_inf * U_inf**2)``;
* the viscous similarity parameter is the chord Reynolds number
  ``Re_c = rho_inf * U_inf * chord / mu``.

The :class:`ReferenceScales` value object is the single source of truth for
these transformations.  Legacy flat metadata remains readable, while newly
written artifacts contain a versioned nested representation and a stable
fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

NORMALIZATION_SCHEMA = "airfoil-pinn.incompressible-cp"
NORMALIZATION_VERSION = 1


@dataclass(frozen=True)
class ReferenceScales:
    """Validated physical reference scales for incompressible external flow.

    Args:
        chord: Airfoil chord used as the length reference, in metres.
        velocity: Freestream velocity magnitude, in metres per second.
        density: Freestream density, in kilograms per cubic metre.
        dynamic_viscosity: Dynamic viscosity, in pascal-seconds.
        static_pressure: Freestream/gauge reference pressure, in pascals.

    The class is immutable so that a scale cannot be changed after tensors
    have already been nondimensionalized.  Use :meth:`with_chord` to construct
    a new consistent scale set when explicitly migrating a legacy artifact.
    """

    chord: float
    velocity: float
    density: float
    dynamic_viscosity: float
    static_pressure: float

    def __post_init__(self) -> None:
        """Validate that every reference value is finite and physically valid."""
        positive = {
            "chord": self.chord,
            "velocity": self.velocity,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
        }
        for name, value in positive.items():
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(
                    f"Reference scale '{name}' must be finite and positive; "
                    f"received {value!r}."
                )
            object.__setattr__(self, name, numeric)

        pressure = float(self.static_pressure)
        if not np.isfinite(pressure):
            raise ValueError(
                "Reference scale 'static_pressure' must be finite; "
                f"received {self.static_pressure!r}."
            )
        object.__setattr__(self, "static_pressure", pressure)

    @property
    def dynamic_pressure(self) -> float:
        """Return ``0.5 * rho_inf * U_inf**2`` in pascals."""
        return 0.5 * self.density * self.velocity**2

    @property
    def reynolds(self) -> float:
        """Return the chord Reynolds number."""
        return compute_reynolds_number(
            self.velocity,
            self.chord,
            self.density,
            self.dynamic_viscosity,
        )

    def length_to_dimensionless(self, value: ArrayLike) -> ArrayLike:
        """Divide a coordinate or distance by the chord."""
        return _restore_scalar(value, np.asarray(value, dtype=np.float64) / self.chord)

    def length_to_dimensional(self, value: ArrayLike) -> ArrayLike:
        """Convert a dimensionless coordinate or distance to metres."""
        return _restore_scalar(value, np.asarray(value, dtype=np.float64) * self.chord)

    def velocity_to_dimensionless(self, value: ArrayLike) -> ArrayLike:
        """Divide a velocity component by the freestream magnitude."""
        return _restore_scalar(
            value,
            np.asarray(value, dtype=np.float64) / self.velocity,
        )

    def velocity_to_dimensional(self, value: ArrayLike) -> ArrayLike:
        """Convert a dimensionless velocity component to metres per second."""
        return _restore_scalar(
            value,
            np.asarray(value, dtype=np.float64) * self.velocity,
        )

    def pressure_to_coefficient(self, value: ArrayLike) -> ArrayLike:
        """Convert static pressure to the aerodynamic pressure coefficient."""
        coefficient = (
            np.asarray(value, dtype=np.float64) - self.static_pressure
        ) / self.dynamic_pressure
        return _restore_scalar(value, coefficient)

    def coefficient_to_pressure(self, value: ArrayLike) -> ArrayLike:
        """Reconstruct static pressure from an aerodynamic coefficient."""
        pressure = (
            self.static_pressure
            + np.asarray(value, dtype=np.float64) * self.dynamic_pressure
        )
        return _restore_scalar(value, pressure)

    def with_chord(self, chord: float) -> "ReferenceScales":
        """Return a new reference set with an explicitly replaced chord."""
        return replace(self, chord=float(chord))

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the scale contract into versioned NumPy-safe metadata."""
        return {
            "schema": NORMALIZATION_SCHEMA,
            "version": NORMALIZATION_VERSION,
            "length_reference": "airfoil_chord",
            "velocity_reference": "freestream_magnitude",
            "pressure_variable": "pressure_coefficient",
            "pressure_reference": "freestream_static",
            "pressure_scale": "0.5*rho_inf*U_inf^2",
            "chord": self.chord,
            "velocity": self.velocity,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
            "static_pressure": self.static_pressure,
            "dynamic_pressure": self.dynamic_pressure,
            "reynolds_chord": self.reynolds,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        """Return a stable SHA-256 identity for checkpoint/data validation."""
        payload = {
            "schema": NORMALIZATION_SCHEMA,
            "version": NORMALIZATION_VERSION,
            "chord": self.chord,
            "velocity": self.velocity,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
            "static_pressure": self.static_pressure,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ReferenceScales":
        """Load current nested or legacy flat initializer metadata.

        Raises:
            ValueError: If required reference values are absent or invalid.
        """
        nested = metadata.get("reference_scales")
        source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else metadata

        return cls(
            chord=_required_number(source, "chord"),
            velocity=_required_number(
                source,
                "velocity",
                legacy=metadata.get("U_inf"),
            ),
            density=_required_number(
                source,
                "density",
                legacy=metadata.get("rho"),
            ),
            dynamic_viscosity=_required_number(
                source,
                "dynamic_viscosity",
                legacy=metadata.get("mu"),
            ),
            static_pressure=_required_number(
                source,
                "static_pressure",
                legacy=metadata.get("p_ref", 0.0),
                positive=False,
            ),
        )


def _required_number(
    source: Mapping[str, Any],
    name: str,
    *,
    legacy: Any = None,
    positive: bool = True,
) -> float:
    """Extract one finite metadata number with an optional legacy fallback."""
    value = source.get(name, legacy)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Normalization metadata is missing a numeric '{name}' value."
        ) from exc
    if not np.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(
            f"Normalization metadata '{name}' must be {qualifier}; "
            f"received {value!r}."
        )
    return numeric


def _restore_scalar(original: ArrayLike, value: np.ndarray) -> ArrayLike:
    """Return a Python float for scalar input and an array otherwise."""
    if np.isscalar(original):
        return float(np.asarray(value))
    return value


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
