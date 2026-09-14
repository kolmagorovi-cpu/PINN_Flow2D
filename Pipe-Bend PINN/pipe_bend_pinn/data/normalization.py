"""Physical reference scales and nondimensionalization for Pipe-Bend PINN.

The module is ported from the Airfoil PINN package (``airfoil_pinn/data/
normalization.py``) and adapted to internal pipe-bend flow.  The single length
reference is the pipe width (``pipe_width``) instead of the airfoil chord:

* lengths and signed/unsigned distances are divided by the pipe width;
* velocities are divided by the freestream/inlet velocity magnitude;
* pressure is represented by the pressure coefficient
  ``Cp = (p - p_ref) / (0.5 * rho * U_inf**2)``;
* the viscous similarity parameter is the pipe Reynolds number
  ``Re = rho * U_inf * pipe_width / mu``.

:class:`ReferenceScales` is an immutable value object and the single source of
truth for these transformations.  Artifacts written with it contain a versioned
nested representation and a stable fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

NORMALIZATION_SCHEMA = "pipe-bend-pinn.incompressible-cp"
NORMALIZATION_VERSION = 1


@dataclass(frozen=True)
class ReferenceScales:
    """Validated physical reference scales for incompressible internal flow.

    Args:
        pipe_width: Channel width used as the length reference, in metres.
        velocity: Inlet/freestream velocity magnitude, in metres per second.
        density: Fluid density, in kilograms per cubic metre.
        dynamic_viscosity: Dynamic viscosity, in pascal-seconds.
        static_pressure: Reference (gauge) pressure, in pascals.

    The class is immutable so that a scale cannot be changed after tensors
    have already been nondimensionalized.  Use :meth:`with_pipe_width` to
    construct a new consistent scale set when explicitly migrating a legacy
    artifact.
    """

    pipe_width: float
    velocity: float
    density: float
    dynamic_viscosity: float
    static_pressure: float

    def __post_init__(self) -> None:
        """Validate that every reference value is finite and physically valid."""
        positive = {
            "pipe_width": self.pipe_width,
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
        """Return ``0.5 * rho * U_inf**2`` in pascals."""
        return 0.5 * self.density * self.velocity**2

    @property
    def reynolds(self) -> float:
        """Return the pipe Reynolds number."""
        return compute_reynolds_number(
            self.velocity,
            self.pipe_width,
            self.density,
            self.dynamic_viscosity,
        )

    def length_to_dimensionless(self, value: ArrayLike) -> ArrayLike:
        """Divide a length (metres) by the pipe width."""
        if isinstance(value, np.ndarray):
            return value / self.pipe_width
        return float(value) / self.pipe_width

    def length_to_dimensional(self, value: ArrayLike) -> ArrayLike:
        """Multiply a dimensionless length by the pipe width."""
        if isinstance(value, np.ndarray):
            return value * self.pipe_width
        return float(value) * self.pipe_width

    def velocity_to_dimensionless(self, value: ArrayLike) -> ArrayLike:
        """Divide a velocity (m/s) by the reference velocity."""
        if isinstance(value, np.ndarray):
            return value / self.velocity
        return float(value) / self.velocity

    def velocity_to_dimensional(self, value: ArrayLike) -> ArrayLike:
        """Multiply a dimensionless velocity by the reference velocity."""
        if isinstance(value, np.ndarray):
            return value * self.velocity
        return float(value) * self.velocity

    def pressure_to_coefficient(self, value: ArrayLike) -> ArrayLike:
        """Convert a pressure (Pa) to ``Cp = (p - p_ref) / q_inf``."""
        return (value - self.static_pressure) / self.dynamic_pressure

    def coefficient_to_pressure(self, value: ArrayLike) -> ArrayLike:
        """Convert ``Cp`` back to pressure (Pa)."""
        return self.static_pressure + value * self.dynamic_pressure

    def with_pipe_width(self, pipe_width: float) -> "ReferenceScales":
        """Return a new scale set with a different length reference."""
        return ReferenceScales(
            pipe_width=float(pipe_width),
            velocity=self.velocity,
            density=self.density,
            dynamic_viscosity=self.dynamic_viscosity,
            static_pressure=self.static_pressure,
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Return the versioned nested metadata representation."""
        return {
            "schema": NORMALIZATION_SCHEMA,
            "version": NORMALIZATION_VERSION,
            "pipe_width": self.pipe_width,
            "velocity": self.velocity,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
            "static_pressure": self.static_pressure,
            "reynolds": self.reynolds,
            "fingerprint": self.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 identity of the reference scales."""
        canonical = json.dumps(
            {
                "pipe_width": float(self.pipe_width),
                "velocity": float(self.velocity),
                "density": float(self.density),
                "dynamic_viscosity": float(self.dynamic_viscosity),
                "static_pressure": float(self.static_pressure),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ReferenceScales":
        """Restore reference scales from current or legacy metadata."""
        if not metadata:
            raise ValueError("ReferenceScales metadata is empty")
        nested = metadata.get("normalization_reference")
        source = nested if isinstance(nested, Mapping) else metadata
        length = _positive_float(source.get("pipe_width"))
        if length is None:
            # Совместимость с метаданными, где длина называлась chord.
            length = _positive_float(source.get("chord"))
        if length is None:
            raise ValueError("Metadata does not define pipe_width/chord")
        velocity = _positive_float(source.get("velocity", source.get("U_inf")))
        density = _positive_float(source.get("density", source.get("rho")))
        viscosity = _positive_float(
            source.get("dynamic_viscosity", source.get("mu"))
        )
        pressure = source.get("static_pressure", source.get("p_ref"))
        if velocity is None or density is None or viscosity is None:
            raise ValueError(
                "Metadata does not define the full reference scale set"
            )
        return cls(
            pipe_width=length,
            velocity=velocity,
            density=density,
            dynamic_viscosity=viscosity,
            static_pressure=float(pressure) if pressure is not None else 0.0,
        )


def compute_reynolds_number(
    velocity: float,
    pipe_width: float,
    density: float,
    dynamic_viscosity: float,
) -> float:
    """Compute the pipe Reynolds number from one physical reference set."""
    values = (velocity, pipe_width, density, dynamic_viscosity)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Reynolds reference values must be finite")
    if velocity <= 0.0 or pipe_width <= 0.0 or density <= 0.0 or dynamic_viscosity <= 0.0:
        raise ValueError(
            "Velocity, pipe_width, density and dynamic viscosity must be positive"
        )
    return (
        float(velocity)
        * float(pipe_width)
        * float(density)
        / float(dynamic_viscosity)
    )


def _positive_float(value: Any) -> Optional[float]:
    """Convert a metadata value to a finite positive float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 and np.isfinite(result) else None
