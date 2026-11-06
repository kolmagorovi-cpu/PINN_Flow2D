"""Observe weighted loss gradients without touching optimizer gradient buffers."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch


COMPONENTS = ("pde", "bc", "fvm", "integral")
COSINE_PAIRS = (("fvm", "pde"), ("fvm", "bc"), ("pde", "bc"))


class GradientContributions:
    """Accumulate vectors within a step, then average their L2 norms over steps.

    Only model parameters are measured; trainable loss weights are held fixed.
    autograd.grad retains the training graph but never populates parameter.grad.
    """

    def __init__(
        self, parameters: Iterable[torch.nn.Parameter], *,
        compute_cosine: bool = True, include_contributions: bool = True,
    ) -> None:
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        self.compute_cosine = compute_cosine
        self.include_contributions = include_contributions
        self.reset()

    def reset(self) -> None:
        self._buffers: dict[str, list[torch.Tensor | None]] = {}
        self._norm_sums = dict.fromkeys((*COMPONENTS, "total"), 0.0)
        self._cosine_sums = dict.fromkeys(COSINE_PAIRS, 0.0)
        self._cosine_counts = dict.fromkeys(COSINE_PAIRS, 0)
        self.evaluations = 0

    def accumulate(
        self,
        losses: Mapping[str, torch.Tensor],
        *,
        divisor: int = 1,
        amp_scale: float = 1.0,
    ) -> None:
        """Use the same weights, batch divisor and AMP scale as the backward."""
        for name in COMPONENTS:
            loss = losses[name]
            buffers = self._buffers.setdefault(name, [None] * len(self.parameters))
            if not self.parameters or not loss.requires_grad:
                continue
            gradients = torch.autograd.grad(
                (loss / divisor) * amp_scale,
                self.parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                value = gradient.detach()
                if value.dtype in (torch.float16, torch.bfloat16):
                    value = value.float()
                value = value / amp_scale
                if buffers[index] is None:
                    buffers[index] = value
                else:
                    buffers[index].add_(value)

    @staticmethod
    def _norm(buffers: Iterable[torch.Tensor | None]) -> float:
        # Double precision reductions avoid overflow when gradients are finite.
        return math.sqrt(sum(
            float(value.double().square().sum())
            for value in buffers if value is not None
        ))

    def finish_step(self, *, stage: str) -> dict[str, Any]:
        """Release vectors after measuring a complete accumulation window."""
        norms = {name: self._norm(self._buffers.get(name, [])) for name in COMPONENTS}
        for name in COMPONENTS:
            self._norm_sums[name] += norms[name]
        if self.compute_cosine:
            for pair in COSINE_PAIRS:
                if not all(math.isfinite(norms[name]) and norms[name] > 0 for name in pair):
                    continue
                left, right = pair
                # Align by parameter, treating unused parameters as zero gradients.
                # Normalize before the dot product to avoid overflowing norm products.
                cosine = sum(
                    float(((a.double() / norms[left]) * (b.double() / norms[right])).sum())
                    for a, b in zip(self._buffers[left], self._buffers[right])
                    if a is not None and b is not None
                )
                self._cosine_sums[pair] += max(-1.0, min(1.0, cosine))
                self._cosine_counts[pair] += 1
        total = []
        for index in range(len(self.parameters)):
            values = [
                buffers[index] for buffers in self._buffers.values()
                if buffers[index] is not None
            ]
            total.append(sum(values) if values else None)
        self._norm_sums["total"] += self._norm(total)
        self.evaluations += 1
        self._buffers.clear()
        means = {name: value / self.evaluations for name, value in self._norm_sums.items()}
        denominator = sum(means[name] for name in COMPONENTS)

        def finite(value: float) -> float | None:
            return value if math.isfinite(value) else None

        report = {
            "stage": stage,
            "include_contributions": self.include_contributions,
            "compute_cosine": self.compute_cosine,
            "evaluations": self.evaluations,
            "aggregation": "mean_step_l2",
            "parameter_scope": "model",
            "weighted": True,
            "pre_clip": True,
            "amp_unscaled": True,
            "components": {
                name: {
                    "l2": finite(means[name]),
                    "share_percent": finite(100.0 * means[name] / denominator)
                    if denominator else 0.0,
                }
                for name in COMPONENTS
            },
            "total_l2": finite(means["total"]),
        }
        for pair in COSINE_PAIRS:
            key = f"cosine_{pair[0]}_{pair[1]}"
            count = self._cosine_counts[pair]
            report[key] = self._cosine_sums[pair] / count if count else None
            report[f"{key}_evaluations"] = count
        return report


def format_gradient_contributions(report: Mapping[str, Any], epoch: int) -> str:
    """One representation for console and the experiment text log."""
    def number(value: float | None, fmt: str) -> str:
        return format(value, fmt) if value is not None else "non-finite"

    lines = [
        f"Gradient contributions | epoch {epoch} | {report['stage']}",
        "Weighted dL/d(model parameters), AMP unscaled, pre-clip; "
        f"mean L2 over {report['evaluations']} evaluation(s).",
        "Share = component L2 / sum of component L2 norms (not total gradient L2).",
    ]
    for name in COMPONENTS:
        row = report["components"][name]
        lines.append(
            f"  {name.upper():<8} L2={number(row['l2'], '.6e')}  "
            f"share={number(row['share_percent'], '.2f')}%"
        )
    lines.append(f"  TOTAL    L2={number(report['total_l2'], '.6e')}")
    if not report.get("include_contributions", True):
        lines = [f"Gradient cosine | epoch {epoch} | {report['stage']}"]
    if report.get("compute_cosine", True):
        for left, right in COSINE_PAIRS:
            key = f"cosine_{left}_{right}"
            cosine = report.get(key)
            cosine_text = format(cosine, ".6f") if cosine is not None else "n/a"
            lines.append(
                f"  cosine({left.upper()}, {right.upper()})={cosine_text} "
                f"(mean over {report.get(f'{key}_evaluations', 0)} valid evaluation(s))"
            )
    return "\n".join(lines)
