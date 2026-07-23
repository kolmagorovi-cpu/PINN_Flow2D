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
    payload = {
        "feature_manifest": loss_computer.feature_manifest.to_dict(),
        "reference_scales": reference_scales.to_metadata(),
        "reference_scales_fingerprint": reference_scales.fingerprint(),
        "sdf_geometry_fingerprint": loss_computer.sdf_geometry_fingerprint,
        "fvm_sdf_geometry_fingerprint": getattr(
            data_handler,
            "fvm_sdf_geometry_fingerprint",
            None,
        ),
        "flow_cases": flow_cases,
        "boundary_type_names": list(data_handler.boundary_type_names),
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
