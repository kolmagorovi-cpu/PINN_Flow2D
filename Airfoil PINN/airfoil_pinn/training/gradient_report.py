"""Read-only summaries of the gradients used by an optimizer."""
from __future__ import annotations

import math
from typing import Iterable, TypedDict

import torch


class GradientRow(TypedDict):
    name: str
    shape: tuple[int, ...]
    norm: float | None
    mean: float | None
    maximum: float | None
    status: str


GradientReport = tuple[list[GradientRow], list[GradientRow], str, float | None]


def collect_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[GradientRow]:
    """Detach statistics only; never retain graphs or modify gradient buffers."""
    rows = []
    for name, parameter in named_parameters:
        grad = parameter.grad
        status = "frozen" if not parameter.requires_grad else "None (no gradient)"
        norm = mean = maximum = None
        if parameter.requires_grad and grad is not None:
            values = grad.detach().double()
            norm = values.norm().item()
            mean = values.abs().mean().item() if values.numel() else 0.0
            maximum = values.abs().max().item() if values.numel() else 0.0
            status = "zero" if maximum == 0 else "active"
        rows.append(dict(name=name, shape=tuple(parameter.shape),
                         norm=norm, mean=mean, maximum=maximum, status=status))
    return rows


def total_norm(rows: list[GradientRow]) -> float:
    return math.sqrt(sum(row["norm"] ** 2 for row in rows if row["norm"] is not None))


def print_gradient_report(report: GradientReport | None, epoch: int) -> None:
    if report is None:
        print(f"Gradients | epoch {epoch}: no optimizer gradient evaluation")
        return
    before, after, stage, clip = report
    print(f"\nGradients d(L_total)/d(parameter) | epoch {epoch} | {stage}")
    print("Last evaluation; accumulated gradients, AMP unscaled, before clipping.")
    print("weight = weights; bias = offsets; layer = module path in parameter name")
    width = max(30, max((len(row["name"]) for row in before), default=0))
    print(f"{'Parameter / layer':<{width}}  {'Shape':<18} {'L2':>11} {'mean|g|':>11} {'max|g|':>11}  Status")
    for prefix, title in ((False, "Model parameters"), (True, "Trainable loss weights (sa_weights)")):
        group = [row for row in before if row["name"].startswith("sa_weights.") == prefix]
        if not group:
            continue
        print(title)
        for row in group:
            stats = ["--" if row[key] is None else f"{row[key]:.3e}" for key in ("norm", "mean", "maximum")]
            print(f"{row['name']:<{width}}  {str(row['shape']):<18} {stats[0]:>11} {stats[1]:>11} {stats[2]:>11}  {row['status']}")
        post = [row for row in after if row["name"].startswith("sa_weights.") == prefix]
        changed = any(a["norm"] != b["norm"] for a, b in zip(group, post))
        print(f"  L2 pre-clip={total_norm(group):.3e}; post-clip={total_norm(post):.3e}; "
              f"clip limit={clip}; clipping applied={'yes' if changed else 'no'}")
    print(f"Total L2: pre-clip={total_norm(before):.3e}; post-clip={total_norm(after):.3e}")
