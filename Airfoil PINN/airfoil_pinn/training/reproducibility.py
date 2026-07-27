"""Capture and restore stochastic state for deterministic checkpoint resume."""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping

import numpy as np
import torch

__all__ = ["capture_rng_state", "restore_rng_state"]

logger = logging.getLogger(__name__)


def capture_rng_state() -> dict[str, Any]:
    """Return Python, NumPy, CPU Torch and optional CUDA RNG states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            value.cpu()
            for value in torch.cuda.get_rng_state_all()
        ]
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore available RNG fields from a modern checkpoint.

    Missing state is accepted for backward compatibility with legacy
    checkpoints.  Malformed present fields are allowed to raise, because
    silently accepting them would make a resumed run irreproducible.
    """
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            logger.warning(
                "Checkpoint contains CUDA RNG state, but resume runs without "
                "CUDA; CPU continuation remains reproducible."
            )
            return
        device_count = torch.cuda.device_count()
        for index, value in enumerate(cuda_state[:device_count]):
            torch.cuda.set_rng_state(value.cpu(), device=index)
        if len(cuda_state) != device_count:
            logger.warning(
                "CUDA device count changed across resume (%d -> %d); only "
                "matching RNG streams were restored.",
                len(cuda_state),
                device_count,
            )
