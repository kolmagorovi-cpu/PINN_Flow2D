"""Optional terminal progress for long-running training loops."""

from __future__ import annotations

import logging
import sys
from typing import Iterable

logger = logging.getLogger(__name__)

__all__ = ["build_epoch_progress"]


def build_epoch_progress(
    epochs: Iterable[int],
    *,
    enabled: bool,
    force: bool = False,
    mininterval: float = 0.5,
):
    """Wrap an epoch iterable in tqdm when requested and usable.

    Non-interactive jobs keep plain logging unless ``force`` is set, preventing
    control characters from polluting scheduler and CI logs.  Missing tqdm is
    a supported fallback rather than a training failure.
    """
    if not enabled or (not force and not sys.stderr.isatty()):
        return epochs
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.warning(
            "output.progress_bar=True, but tqdm is unavailable; "
            "continuing with periodic logs."
        )
        return epochs
    return tqdm(
        epochs,
        desc="PINN training",
        unit="epoch",
        dynamic_ncols=True,
        mininterval=float(mininterval),
    )
