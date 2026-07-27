"""Typed collocation batches with legacy tuple compatibility."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterator, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PINNBatch:
    """Named tensors consumed by training, diagnostics and boundary losses.

    ``case_id`` is intentionally not yielded by :meth:`__iter__`.  Existing
    code that unpacks the historical eleven-element tuple therefore remains
    functional while new parameter-aware code can access ``batch.case_id``.
    """

    index: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    sdf: torch.Tensor
    boundary_types: torch.Tensor
    mask_u_fixed: torch.Tensor
    mask_v_fixed: torch.Tensor
    mask_p_fixed: torch.Tensor
    u_bc: torch.Tensor
    v_bc: torch.Tensor
    p_bc: torch.Tensor
    normal_x: torch.Tensor
    normal_y: torch.Tensor
    case_id: torch.Tensor

    def legacy_tuple(self) -> Tuple[torch.Tensor, ...]:
        """Return the historical eleven-field view."""
        return (
            self.index,
            self.x,
            self.y,
            self.sdf,
            self.boundary_types,
            self.mask_u_fixed,
            self.mask_v_fixed,
            self.mask_p_fixed,
            self.u_bc,
            self.v_bc,
            self.p_bc,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over the backward-compatible tuple view."""
        return iter(self.legacy_tuple())

    def __len__(self) -> int:
        """Return the legacy tuple width."""
        return len(self.legacy_tuple())

    def __getitem__(self, index: int) -> torch.Tensor:
        """Index the legacy tuple view for compatibility."""
        return self.legacy_tuple()[index]

    def pin_memory(self) -> "PINNBatch":
        """Return a batch whose CPU tensors use pinned memory."""
        return PINNBatch(
            **{
                field.name: getattr(self, field.name).pin_memory()
                for field in fields(self)
            }
        )


def collate_pinn_batches(samples: Sequence[PINNBatch]) -> PINNBatch:
    """Stack point samples into a typed mini-batch.

    Raises:
        ValueError: If the dataloader attempts to collate an empty sample list.
    """
    if not samples:
        raise ValueError("Cannot collate an empty PINN sample sequence")
    return PINNBatch(
        **{
            field.name: torch.stack(
                [getattr(sample, field.name) for sample in samples],
                dim=0,
            )
            for field in fields(PINNBatch)
        }
    )
