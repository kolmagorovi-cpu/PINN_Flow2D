"""Machine-readable run manifests and per-epoch PINN telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

__all__ = ["RunTelemetry"]


def _json_value(value: Any) -> Any:
    """Convert common numerical/container values to strict JSON types."""
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunTelemetry:
    """Persist an atomic run manifest and append-only epoch JSONL records."""

    def __init__(self, logs_dir: str) -> None:
        """Create telemetry paths below an experiment log directory."""
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.logs_dir / "run_manifest.json"
        self.epochs_path = self.logs_dir / "epochs.jsonl"

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        """Atomically replace the runtime manifest with strict JSON."""
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        serialized = json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        try:
            temporary_path.write_text(serialized + "\n", encoding="utf-8")
            os.replace(temporary_path, self.manifest_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    def append_epoch(self, payload: Mapping[str, Any]) -> None:
        """Append one fully flushed strict-JSON epoch record."""
        serialized = json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self.epochs_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
