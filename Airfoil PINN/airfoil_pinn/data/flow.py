"""Dimensionless flow-case contracts for parametric PINN training.

Physical density, viscosity, chord and freestream speed are reference scales,
not independent inputs of a constant-property incompressible solution.  Their
dimensionless influence is represented by the chord Reynolds number.  The
freestream direction is represented by ``cos(alpha)`` and ``sin(alpha)`` to
avoid the angular discontinuity at ``-pi/pi``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class FlowCase:
    """One dimensionless member of a parametric incompressible-flow family.

    Args:
        case_id: Non-negative identifier stored with every collocation point.
        reynolds: Chord Reynolds number.
        angle_degrees: Freestream angle of attack in degrees.
        pressure_drop_coefficient: Optional imposed pressure-drop coefficient
            for internal-flow problems.  External aerodynamics leaves it unset.
        geometry_parameters: Optional dimensionless shape parameters in the
            order declared by the feature configuration.
        name: Human-readable case label.
    """

    case_id: int
    reynolds: float
    angle_degrees: float = 0.0
    pressure_drop_coefficient: Optional[float] = None
    geometry_parameters: Tuple[float, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the dimensionless case and canonicalize scalar types."""
        case_id = int(self.case_id)
        if case_id < 0:
            raise ValueError("case_id must be non-negative")
        object.__setattr__(self, "case_id", case_id)

        reynolds = float(self.reynolds)
        if not np.isfinite(reynolds) or reynolds <= 0.0:
            raise ValueError("FlowCase.reynolds must be finite and positive")
        object.__setattr__(self, "reynolds", reynolds)

        angle = float(self.angle_degrees)
        if not np.isfinite(angle):
            raise ValueError("FlowCase.angle_degrees must be finite")
        object.__setattr__(self, "angle_degrees", angle)

        if self.pressure_drop_coefficient is not None:
            pressure_drop = float(self.pressure_drop_coefficient)
            if not np.isfinite(pressure_drop):
                raise ValueError(
                    "FlowCase.pressure_drop_coefficient must be finite"
                )
            object.__setattr__(
                self,
                "pressure_drop_coefficient",
                pressure_drop,
            )

        geometry = tuple(float(value) for value in self.geometry_parameters)
        if not all(np.isfinite(value) for value in geometry):
            raise ValueError("FlowCase.geometry_parameters must be finite")
        object.__setattr__(self, "geometry_parameters", geometry)
        object.__setattr__(self, "name", str(self.name))

    @property
    def angle_radians(self) -> float:
        """Return angle of attack in radians."""
        return math.radians(self.angle_degrees)

    @property
    def cos_alpha(self) -> float:
        """Return the normalized streamwise freestream component."""
        return math.cos(self.angle_radians)

    @property
    def sin_alpha(self) -> float:
        """Return the normalized transverse freestream component."""
        return math.sin(self.angle_radians)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the dimensionless case for datasets and checkpoints."""
        return {
            "case_id": self.case_id,
            "name": self.name,
            "reynolds_chord": self.reynolds,
            "angle_degrees": self.angle_degrees,
            "cos_alpha": self.cos_alpha,
            "sin_alpha": self.sin_alpha,
            "pressure_drop_coefficient": self.pressure_drop_coefficient,
            "geometry_parameters": list(self.geometry_parameters),
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        """Return a stable identity for case-aware artifacts."""
        payload = {
            "case_id": self.case_id,
            "reynolds": self.reynolds,
            "angle_degrees": self.angle_degrees,
            "pressure_drop_coefficient": self.pressure_drop_coefficient,
            "geometry_parameters": self.geometry_parameters,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "FlowCase":
        """Load a current flow-case metadata dictionary."""
        return cls(
            case_id=int(metadata.get("case_id", 0)),
            name=str(metadata.get("name", "")),
            reynolds=float(
                metadata.get(
                    "reynolds_chord",
                    metadata.get("reynolds"),
                )
            ),
            angle_degrees=float(metadata.get("angle_degrees", 0.0)),
            pressure_drop_coefficient=metadata.get(
                "pressure_drop_coefficient"
            ),
            geometry_parameters=tuple(
                metadata.get("geometry_parameters", ())
            ),
        )

    @classmethod
    def from_freestream_components(
        cls,
        *,
        case_id: int,
        reynolds: float,
        u_normalized: float,
        v_normalized: float,
        name: str = "",
    ) -> "FlowCase":
        """Construct a case from a normalized freestream direction."""
        u_value = float(u_normalized)
        v_value = float(v_normalized)
        magnitude = math.hypot(u_value, v_value)
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError(
                "Normalized freestream components must define a direction"
            )
        if not math.isclose(magnitude, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                "Normalized freestream direction must have unit magnitude; "
                f"received {magnitude:.9g}"
            )
        return cls(
            case_id=case_id,
            reynolds=reynolds,
            angle_degrees=math.degrees(math.atan2(v_value, u_value)),
            name=name,
        )


@dataclass(frozen=True)
class FlowCaseBatch:
    """Per-point tensors used by feature assembly and parameterized physics."""

    case_id: torch.Tensor
    reynolds: torch.Tensor
    cos_alpha: torch.Tensor
    sin_alpha: torch.Tensor
    pressure_drop_coefficient: Optional[torch.Tensor] = None
    geometry_parameters: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """Return the number of point/case associations."""
        return int(self.case_id.shape[0])


def flow_cases_from_metadata(
    metadata: Mapping[str, Any],
    *,
    fallback_reynolds: float,
    fallback_u_normalized: float,
    fallback_v_normalized: float,
) -> Dict[int, FlowCase]:
    """Load a case table or construct one compatible legacy case."""
    raw_cases = metadata.get("flow_cases")
    if isinstance(raw_cases, Sequence) and not isinstance(
        raw_cases,
        (str, bytes),
    ):
        cases = [FlowCase.from_metadata(item) for item in raw_cases]
    else:
        cases = [
            FlowCase.from_freestream_components(
                case_id=0,
                reynolds=fallback_reynolds,
                u_normalized=fallback_u_normalized,
                v_normalized=fallback_v_normalized,
                name=str(metadata.get("conditions_name", "legacy-case")),
            )
        ]

    result = {case.case_id: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("flow_cases metadata contains duplicate case_id values")
    if not result:
        raise ValueError("flow_cases metadata must contain at least one case")
    return result
