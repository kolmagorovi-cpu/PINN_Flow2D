"""Manifest-driven assembly of dimensionless parametric PINN inputs.

The assembler separates physical nondimensionalization from optional machine
learning encodings.  Coordinates and SDF are already normalized by chord.
Reynolds number remains the physical dimensionless value for the PDE, while
the network receives a log-scaled representation over a configured range.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from airfoil_pinn.data.flow import FlowCaseBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSlice:
    """One contiguous semantic field in a network input tensor."""

    name: str
    start: int
    stop: int
    encoding: str

    @property
    def width(self) -> int:
        """Return the number of scalar network inputs in the field."""
        return self.stop - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this slice for logs and checkpoints."""
        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
            "encoding": self.encoding,
        }


@dataclass(frozen=True)
class InputFeatureManifest:
    """Ordered, immutable schema of the network input tensor."""

    features: Tuple[FeatureSlice, ...]

    @property
    def input_dim(self) -> int:
        """Return total network input width."""
        return self.features[-1].stop if self.features else 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete ordered feature contract."""
        return {
            "input_dim": self.input_dim,
            "features": [feature.to_dict() for feature in self.features],
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        """Return a stable identity for model/checkpoint compatibility."""
        payload = [feature.to_dict() for feature in self.features]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class InputFeatureAssembler:
    """Build network inputs from coordinates, SDF and a dimensionless case.

    New configurations use ``input_features``.  Legacy ``input_params`` remain
    readable during migration:

    * ``inlet_velocity`` maps to ``angle_of_attack``;
    * ``viscosity`` maps to ``reynolds``;
    * ``density`` is ignored because it is redundant once ``Re_c`` and the
      reference scales are defined.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        boundary_type_names: Sequence[str],
    ) -> None:
        self.config = config
        self.legacy = config.get("input_params", {}) or {}
        self.features_config = config.get("input_features", {}) or {}
        self.boundary_type_names = tuple(
            str(name) for name in boundary_type_names
        )

        self.coordinates_enabled = self._enabled(
            "coordinates",
            legacy_name="coordinates",
            default=True,
        )
        self.sdf_enabled = self._enabled(
            "sdf",
            legacy_name="sdf",
            default=True,
        )
        if not self.coordinates_enabled:
            raise ValueError(
                "Coordinates x* and y* are mandatory for a differential PINN."
            )
        if not self.sdf_enabled:
            raise ValueError(
                "SDF is a mandatory network input for this project."
            )

        self.boundary_types_enabled = self._enabled(
            "boundary_types",
            legacy_name="boundary_types",
            default=True,
        )
        self.reynolds_enabled = self._enabled(
            "reynolds",
            legacy_name="viscosity",
            default=False,
        )
        self.angle_enabled = self._enabled(
            "angle_of_attack",
            legacy_name="inlet_velocity",
            default=False,
        )
        self.pressure_drop_enabled = self._enabled(
            "pressure_drop_coefficient",
            default=False,
        )

        if bool(self.legacy.get("density", False)):
            logger.warning(
                "Legacy input_params.density is ignored: constant-property "
                "incompressible similarity is represented by Re_c."
            )

        reynolds_config = self._feature_mapping("reynolds")
        reynolds_range = reynolds_config.get("range", (1e3, 1e7))
        if (
            not isinstance(reynolds_range, Sequence)
            or len(reynolds_range) != 2
        ):
            raise ValueError("Reynolds feature range must contain [min, max]")
        self.reynolds_min = float(reynolds_range[0])
        self.reynolds_max = float(reynolds_range[1])
        if (
            not math.isfinite(self.reynolds_min)
            or not math.isfinite(self.reynolds_max)
            or self.reynolds_min <= 0.0
            or self.reynolds_max <= self.reynolds_min
        ):
            raise ValueError(
                "Reynolds feature range must be finite, positive and ordered"
            )
        self.log_reynolds_min = math.log10(self.reynolds_min)
        self.log_reynolds_max = math.log10(self.reynolds_max)

        geometry_config = self._feature_mapping("geometry_parameters")
        geometry_names = geometry_config.get("names", ())
        if not isinstance(geometry_names, Sequence) or isinstance(
            geometry_names,
            (str, bytes),
        ):
            raise ValueError("geometry_parameters.names must be a sequence")
        self.geometry_parameter_names = tuple(
            str(name) for name in geometry_names
        )
        self.geometry_enabled = self._enabled(
            "geometry_parameters",
            default=bool(self.geometry_parameter_names),
        )
        if self.geometry_enabled and not self.geometry_parameter_names:
            raise ValueError(
                "Enabled geometry_parameters require an ordered names list"
            )

        self.requires_flow = (
            self.reynolds_enabled
            or self.angle_enabled
            or self.pressure_drop_enabled
            or self.geometry_enabled
        )
        self.manifest = self._build_manifest()

    def _feature_mapping(self, name: str) -> Mapping[str, Any]:
        """Return a mapping for one new-style feature configuration."""
        value = self.features_config.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def _enabled(
        self,
        name: str,
        *,
        legacy_name: Optional[str] = None,
        default: bool,
    ) -> bool:
        """Resolve a new feature flag with a legacy boolean fallback."""
        if name in self.features_config:
            value = self.features_config[name]
            if isinstance(value, Mapping):
                return bool(value.get("enabled", default))
            return bool(value)
        if legacy_name is not None and legacy_name in self.legacy:
            return bool(self.legacy[legacy_name])
        return bool(default)

    def _build_manifest(self) -> InputFeatureManifest:
        """Construct the deterministic feature order and column ranges."""
        specs = []
        cursor = 0

        def append(name: str, width: int, encoding: str) -> None:
            nonlocal cursor
            specs.append(
                FeatureSlice(
                    name=name,
                    start=cursor,
                    stop=cursor + width,
                    encoding=encoding,
                )
            )
            cursor += width

        append("coordinates", 2, "x/c,y/c")
        append("sdf", 1, "distance/chord")
        if self.reynolds_enabled:
            append("reynolds", 1, "log10_minmax[-1,1]")
        if self.angle_enabled:
            append("angle_of_attack", 2, "cos(alpha),sin(alpha)")
        if self.boundary_types_enabled and self.boundary_type_names:
            append(
                "boundary_types",
                len(self.boundary_type_names),
                "one_hot:" + ",".join(self.boundary_type_names),
            )
        if self.pressure_drop_enabled:
            append("pressure_drop_coefficient", 1, "delta_p/q_inf")
        if self.geometry_enabled:
            append(
                "geometry_parameters",
                len(self.geometry_parameter_names),
                "dimensionless:" + ",".join(self.geometry_parameter_names),
            )
        return InputFeatureManifest(tuple(specs))

    def build(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        flow: Optional[FlowCaseBatch],
    ) -> torch.Tensor:
        """Assemble and validate the tensor described by :attr:`manifest`."""
        self._validate_base_shapes(x, y, sdf)
        inputs = [x, y, sdf]

        if self.reynolds_enabled:
            flow = self._require_flow(flow, x.shape[0])
            if not bool(torch.isfinite(flow.reynolds).all()):
                raise ValueError("FlowCaseBatch.reynolds contains NaN or Inf")
            if bool((flow.reynolds <= 0.0).any()):
                raise ValueError("FlowCaseBatch.reynolds must be positive")
            encoded_reynolds = 2.0 * (
                (
                    torch.log10(flow.reynolds)
                    - self.log_reynolds_min
                )
                / (self.log_reynolds_max - self.log_reynolds_min)
            ) - 1.0
            inputs.append(encoded_reynolds)

        if self.angle_enabled:
            flow = self._require_flow(flow, x.shape[0])
            inputs.extend((flow.cos_alpha, flow.sin_alpha))

        if self.boundary_types_enabled and self.boundary_type_names:
            if boundary_types is None:
                raise ValueError(
                    "Boundary one-hot features are enabled but missing"
                )
            expected = len(self.boundary_type_names)
            if boundary_types.shape != (x.shape[0], expected):
                raise ValueError(
                    "boundary_types shape mismatch: expected "
                    f"({x.shape[0]}, {expected}), received "
                    f"{tuple(boundary_types.shape)}"
                )
            inputs.append(boundary_types)

        if self.pressure_drop_enabled:
            flow = self._require_flow(flow, x.shape[0])
            if flow.pressure_drop_coefficient is None:
                raise ValueError(
                    "pressure_drop_coefficient feature is enabled but absent"
                )
            inputs.append(flow.pressure_drop_coefficient)

        if self.geometry_enabled:
            flow = self._require_flow(flow, x.shape[0])
            geometry = flow.geometry_parameters
            expected = len(self.geometry_parameter_names)
            if geometry is None or geometry.shape != (x.shape[0], expected):
                shape = None if geometry is None else tuple(geometry.shape)
                raise ValueError(
                    "geometry_parameters shape mismatch: expected "
                    f"({x.shape[0]}, {expected}), received {shape}"
                )
            inputs.append(geometry)

        result = torch.cat(inputs, dim=1)
        if result.shape[1] != self.manifest.input_dim:
            raise RuntimeError(
                "Input feature implementation disagrees with its manifest: "
                f"{result.shape[1]} != {self.manifest.input_dim}"
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Assembled network input contains NaN or Inf")
        return result

    @staticmethod
    def _validate_base_shapes(
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
    ) -> None:
        """Validate mandatory coordinate and distance columns."""
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x must have shape (batch, 1)")
        if y.shape != x.shape or sdf.shape != x.shape:
            raise ValueError("x, y and sdf must have identical (batch, 1) shapes")
        if x.device != y.device or x.device != sdf.device:
            raise ValueError("x, y and sdf must be on the same device")
        if x.dtype != y.dtype or x.dtype != sdf.dtype:
            raise ValueError("x, y and sdf must use the same dtype")

    @staticmethod
    def _require_flow(
        flow: Optional[FlowCaseBatch],
        batch_size: int,
    ) -> FlowCaseBatch:
        """Return a compatible case batch or fail with a clear message."""
        if flow is None:
            raise ValueError(
                "Enabled parametric input features require FlowCaseBatch"
            )
        if flow.batch_size != batch_size:
            raise ValueError(
                f"FlowCaseBatch size {flow.batch_size} != point batch {batch_size}"
            )
        return flow
