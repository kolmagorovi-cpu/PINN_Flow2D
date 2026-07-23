"""Numerical health checks for physics-informed optimization.

PINN residuals contain nested automatic differentiation.  A non-finite value
can therefore originate in the network output, a spatial derivative, an
individual loss component or an optimizer gradient.  These helpers fail before
an optimizer step and report the offending stage instead of silently
propagating a corrupted state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

import torch

__all__ = [
    "GradientHealth",
    "NumericalTrainingError",
    "check_gradient_health",
    "ensure_finite_loss",
]


class NumericalTrainingError(FloatingPointError):
    """Training cannot continue because a tensor contains NaN or infinity."""


@dataclass(frozen=True)
class GradientHealth:
    """Aggregate diagnostics for a collection of parameter gradients."""

    l2_norm: float
    max_abs: float
    tensor_count: int
    nonfinite_parameters: Tuple[str, ...]

    @property
    def is_finite(self) -> bool:
        """Whether every observed gradient value is finite."""
        return not self.nonfinite_parameters


def _component_summary(
    components: Optional[Mapping[str, object]],
) -> str:
    if not components:
        return ""
    rendered = []
    for name, value in components.items():
        if torch.is_tensor(value):
            detached = value.detach()
            if detached.numel() == 1:
                rendered.append(f"{name}={detached.item():.6e}")
            else:
                rendered.append(
                    f"{name}[shape={tuple(detached.shape)}]"
                )
        elif isinstance(value, (int, float)):
            rendered.append(f"{name}={float(value):.6e}")
        else:
            rendered.append(f"{name}={value!r}")
    return "; components: " + ", ".join(rendered)


def ensure_finite_loss(
    loss: torch.Tensor,
    *,
    stage: str,
    components: Optional[Mapping[str, object]] = None,
) -> None:
    """Raise before backward when a scalar loss is non-finite."""
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise NumericalTrainingError(
            f"{stage}: total loss must be a scalar tensor"
        )
    if not bool(torch.isfinite(loss.detach()).all()):
        raise NumericalTrainingError(
            f"{stage}: non-finite total loss {loss.detach().item()!r}"
            + _component_summary(components)
        )


def check_gradient_health(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    stage: str,
) -> GradientHealth:
    """Return gradient norms and raise if any gradient is non-finite."""
    squared_norm = 0.0
    max_abs = 0.0
    tensor_count = 0
    nonfinite = []

    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        tensor_count += 1
        if not bool(torch.isfinite(detached).all()):
            nonfinite.append(name)
            continue
        squared_norm += float(torch.sum(detached.double() ** 2).item())
        if detached.numel() > 0:
            max_abs = max(
                max_abs,
                float(detached.abs().max().item()),
            )

    health = GradientHealth(
        l2_norm=squared_norm**0.5,
        max_abs=max_abs,
        tensor_count=tensor_count,
        nonfinite_parameters=tuple(nonfinite),
    )
    if not health.is_finite:
        names = ", ".join(health.nonfinite_parameters[:8])
        suffix = (
            " ..."
            if len(health.nonfinite_parameters) > 8
            else ""
        )
        raise NumericalTrainingError(
            f"{stage}: non-finite gradients in {names}{suffix}"
        )
    if health.tensor_count == 0:
        raise NumericalTrainingError(
            f"{stage}: backward produced no parameter gradients"
        )
    return health
