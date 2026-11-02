"""CPU thread-pool configuration for PyTorch training."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

__all__ = ["configure_cpu_threading", "resolve_thread_count"]


def resolve_thread_count(value: Any, *, automatic: int) -> int:
    """Resolve ``auto`` or a positive integer to a thread count."""
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return max(1, int(automatic))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("thread count must be 'auto' or a positive integer")
    return value


def configure_cpu_threading(
    process_config: Mapping[str, Any],
    *,
    cpu_active: bool,
) -> dict[str, Any]:
    """Configure PyTorch intra-op and inter-op CPU thread pools.

    Intra-op threads parallelize tensor kernels used by the forward pass,
    autograd and optimizers. Inter-op threads schedule independent operators;
    keeping that pool small avoids nested oversubscription.

    ``torch.set_num_interop_threads`` may only be called before inter-op work
    starts. If another solver has already initialized that pool, its effective
    value is retained and reported instead of aborting training.
    """
    available_cpus = max(1, os.cpu_count() or 1)
    requested_intra = resolve_thread_count(
        process_config.get("cpu_threads", "auto"),
        automatic=available_cpus,
    )
    requested_interop = resolve_thread_count(
        process_config.get("cpu_interop_threads", 1),
        automatic=min(4, requested_intra),
    )

    if cpu_active:
        torch.set_num_threads(requested_intra)
        if torch.get_num_interop_threads() != requested_interop:
            try:
                torch.set_num_interop_threads(requested_interop)
            except RuntimeError as exc:
                logger.warning(
                    "Не удалось изменить число inter-op потоков PyTorch на %d: %s. "
                    "Используется текущее значение %d.",
                    requested_interop,
                    exc,
                    torch.get_num_interop_threads(),
                )

    settings = {
        "active": cpu_active,
        "available_cpus": available_cpus,
        "requested_intra_op": requested_intra,
        "requested_inter_op": requested_interop,
        "intra_op": torch.get_num_threads(),
        "inter_op": torch.get_num_interop_threads(),
    }
    if cpu_active:
        logger.info(
            "CPU-многопоточность PyTorch: intra-op=%d, inter-op=%d, "
            "доступно логических CPU=%d",
            settings["intra_op"],
            settings["inter_op"],
            available_cpus,
        )
    return settings
