"""Reproducibility contract stored with every modern PINN checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping

__all__ = [
    "TrainingContract",
    "build_training_contract",
    "validate_checkpoint_contract",
]


_SCHEMA = "airfoil-pinn.training-contract"
_VERSION = 1


@dataclass(frozen=True)
class TrainingContract:
    """Serializable identities of all graph-defining training artifacts."""

    payload: Mapping[str, Any]

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 identity for the full contract."""
        encoded = json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the contract with schema, version and fingerprint."""
        return {
            "schema": _SCHEMA,
            "version": _VERSION,
            "payload": dict(self.payload),
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "TrainingContract":
        """Validate and load checkpoint contract metadata."""
        if metadata.get("schema") != _SCHEMA:
            raise ValueError(
                f"Unsupported training contract schema {metadata.get('schema')!r}"
            )
        if int(metadata.get("version", -1)) != _VERSION:
            raise ValueError(
                f"Unsupported training contract version "
                f"{metadata.get('version')!r}"
            )
        payload = metadata.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Training contract payload must be a mapping")
        contract = cls(dict(payload))
        expected = metadata.get("fingerprint")
        actual = contract.fingerprint()
        if expected != actual:
            raise ValueError(
                "Training contract fingerprint is corrupted: "
                f"{expected!r} != {actual!r}"
            )
        return contract


def build_training_contract(
    config: Mapping[str, Any],
    data_handler: Any,
    loss_computer: Any,
) -> TrainingContract:
    """Build the graph/data identity required for safe checkpoint resume."""
    architecture = config.get("architecture", {})
    pde = config.get("pde", {})
    reference_scales = data_handler.reference_scales
    flow_cases = [
        case.to_metadata()
        for _, case in sorted(data_handler.flow_cases.items())
    ]
    fvm_config = config.get("loss_weights", {}).get("fvm", {})
    if isinstance(fvm_config, Mapping):
        fvm_weight = float(fvm_config.get("weight", 1.0))
        fvm_enabled = (
            bool(fvm_config["enabled"])
            if "enabled" in fvm_config
            else fvm_weight != 0.0
        )
        fvm_reduction = str(
            fvm_config.get(
                "reduction",
                fvm_config.get("loss_reduction", "divergence_l2_volume"),
            )
        ).lower().replace("-", "_")
        fvm_schedule = str(
            fvm_config.get("schedule", "every_optimizer_step")
        ).lower().replace("-", "_")
        fvm_interval = int(fvm_config.get("batch_interval", 1))
    else:
        fvm_enabled = bool(float(fvm_config))
        fvm_weight = float(fvm_config)
        fvm_reduction = "legacy"
        fvm_schedule = "every_optimizer_step"
        fvm_interval = 1
    fvm_operator = getattr(loss_computer, "fvm", None)
    payload = {
        "feature_manifest": loss_computer.feature_manifest.to_dict(),
        "reference_scales": reference_scales.to_metadata(),
        "reference_scales_fingerprint": reference_scales.fingerprint(),
        "sdf_geometry_fingerprint": loss_computer.sdf_geometry_fingerprint,
        "sdf_value_contract": {
            "method": "exact_polyline_segments",
            "evaluation_dtype": "float64",
            "consistency_tolerance": float(
                getattr(
                    loss_computer,
                    "sdf_float64_consistency_tolerance",
                    1e-12,
                )
            ),
            "cached_geometry_fingerprint": getattr(
                data_handler,
                "cached_sdf_geometry_fingerprint",
                None,
            ),
            "cached_method": getattr(
                data_handler,
                "cached_sdf_method",
                None,
            ),
        },
        "fvm_sdf_geometry_fingerprint": getattr(
            data_handler,
            "fvm_sdf_geometry_fingerprint",
            None,
        ),
        "fvm": {
            "enabled": fvm_enabled,
            "reduction": fvm_reduction,
            "weight": fvm_weight,
            "schedule": fvm_schedule,
            "interval": fvm_interval,
            "inclusion_probability": (
                1.0 / fvm_interval
                if fvm_schedule == "stochastic_optimizer_step"
                else 1.0
            ),
            "mesh_identifier": config.get("data", {}).get("mesh_file"),
            "mesh_fingerprint": getattr(
                fvm_operator,
                "mesh_fingerprint",
                None,
            ),
            "normalization": "face_vectors/c;control_volumes/c^2",
            "geometry_is_dimensionless": bool(
                getattr(
                    fvm_operator,
                    "_geometry_is_dimensionless",
                    False,
                )
            ),
            "geometry_length_scale": (
                float(
                    getattr(
                        fvm_operator,
                        "_geometry_length_scale",
                        1.0,
                    )
                )
                if fvm_operator is not None
                else None
            ),
        },
        "flow_cases": flow_cases,
        "boundary_type_names": list(data_handler.boundary_type_names),
        "boundary_enforcement": (
            loss_computer.boundary_enforcement_contract
            if hasattr(loss_computer, "boundary_enforcement_contract")
            else {
                name: {
                    "kind": values.get("kind", name),
                    "mode": values.get("enforcement", "soft"),
                    **(
                        {
                            "transform": "normalized_distance",
                            "distance_scale": float(
                                values.get("distance_scale", 0.02)
                            ),
                        }
                        if (
                            name == "wall"
                            and values.get("enforcement", "soft") == "hard"
                        )
                        else {}
                    ),
                }
                for name, values in config.get(
                    "boundary_conditions",
                    {},
                ).items()
                if isinstance(values, Mapping)
            }
        ),
        "pde": {
            "system": pde.get("system"),
            "flow_type": pde.get("flow_type"),
            "pde_on_boundary": bool(pde.get("pde_on_boundary", False)),
        },
        "architecture": {
            "hidden_layers": list(architecture.get("hidden_layers", ())),
            "activation": architecture.get("activation"),
            "siren_first_omega_0": architecture.get(
                "siren_first_omega_0",
                30.0,
            ),
            "siren_hidden_omega_0": architecture.get(
                "siren_hidden_omega_0",
                30.0,
            ),
            "output_dim": architecture.get("output_dim"),
            "output_bias": list(architecture.get("output_bias", ())),
            "coordinate_scaling": architecture.get(
                "coordinate_scaling",
                {},
            ),
        },
    }
    return TrainingContract(payload)


def _top_level_differences(
    checkpoint: TrainingContract,
    current: TrainingContract,
) -> list[str]:
    keys = set(checkpoint.payload) | set(current.payload)
    return sorted(
        key
        for key in keys
        if checkpoint.payload.get(key) != current.payload.get(key)
    )


def validate_checkpoint_contract(
    metadata: Mapping[str, Any],
    current: TrainingContract,
) -> None:
    """Reject a checkpoint whose graph/data identity differs from the run."""
    checkpoint = TrainingContract.from_metadata(metadata)
    if checkpoint.fingerprint() == current.fingerprint():
        return
    differences = _top_level_differences(checkpoint, current)
    raise RuntimeError(
        "Checkpoint training contract is incompatible with the current "
        "configuration/data. Differing sections: "
        + ", ".join(differences)
        + f". checkpoint={checkpoint.fingerprint()[:12]}, "
        f"current={current.fingerprint()[:12]}"
    )
