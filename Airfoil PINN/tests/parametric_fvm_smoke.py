"""Checks for differentiable, case-local finite-volume constraints."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from airfoil_pinn.physics.fvm import FVMResiduals
from airfoil_pinn.training.components.trainer import Trainer


def _one_cell_mesh() -> SimpleNamespace:
    """Return one control volume with opposite inlet/outlet face vectors."""
    return SimpleNamespace(
        cv_volumes=np.array([1.0], dtype=np.float64),
        int_face_node_i=np.empty(0, dtype=np.int64),
        int_face_node_j=np.empty(0, dtype=np.int64),
        int_face_area=np.empty((0, 2), dtype=np.float64),
        bnd_face_node=np.array([0, 0], dtype=np.int64),
        bnd_face_area=np.array(
            [[-1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        ),
        bnd_face_type=np.array(["inlet", "outlet"]),
    )


class ParametricFVMTests(unittest.TestCase):
    """Verify that one immutable mesh supports multiple operating points."""

    def setUp(self) -> None:
        """Build a deterministic float64 one-cell operator."""
        self.fvm = FVMResiduals(_one_cell_mesh()).to(
            torch.device("cpu"),
            dtype=torch.float64,
        )
        self.zero_velocity = torch.zeros((1, 2), dtype=torch.float64)

    def test_case_local_boundary_does_not_mutate_operator(self) -> None:
        """Different alpha values produce independent losses on one mesh."""
        streamwise = self.fvm.mass_loss(
            self.zero_velocity,
            reduction="flux_mean",
            freestream_velocity=torch.tensor(
                [1.0, 0.0],
                dtype=torch.float64,
            ),
        )
        transverse = self.fvm.mass_loss(
            self.zero_velocity,
            reduction="flux_mean",
            freestream_velocity=torch.tensor(
                [0.0, 1.0],
                dtype=torch.float64,
            ),
        )
        repeated = self.fvm.mass_loss(
            self.zero_velocity,
            reduction="flux_mean",
            freestream_velocity=torch.tensor(
                [1.0, 0.0],
                dtype=torch.float64,
            ),
        )

        torch.testing.assert_close(
            streamwise,
            torch.tensor(1.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            transverse,
            torch.tensor(0.0, dtype=torch.float64),
        )
        torch.testing.assert_close(repeated, streamwise)

    def test_fvm_boundary_remains_differentiable_in_alpha(self) -> None:
        """The FVM loss retains autograd through a continuous flow angle."""
        alpha = torch.tensor(
            math.radians(30.0),
            dtype=torch.float64,
            requires_grad=True,
        )
        freestream = torch.stack((torch.cos(alpha), torch.sin(alpha)))
        loss = self.fvm.mass_loss(
            self.zero_velocity,
            reduction="flux_mean",
            freestream_velocity=freestream,
        )
        derivative = torch.autograd.grad(loss, alpha)[0]

        torch.testing.assert_close(
            derivative,
            torch.tensor(
                -math.sin(2.0 * math.radians(30.0)),
                dtype=torch.float64,
            ),
        )

    def test_invalid_boundary_shape_is_rejected(self) -> None:
        """Ambiguous batched BC tensors fail before broadcasting."""
        with self.assertRaisesRegex(ValueError, "shape"):
            self.fvm.mass_loss(
                self.zero_velocity,
                freestream_velocity=torch.ones(
                    (2, 2),
                    dtype=torch.float64,
                ),
            )

    def test_geometry_normalization_survives_rebuild(self) -> None:
        """Face and cell measures use the chord scale exactly once."""
        self.fvm.nondimensionalize_geometry(2.0)
        torch.testing.assert_close(
            self.fvm.V,
            torch.tensor([0.25], dtype=torch.float64),
        )
        torch.testing.assert_close(
            self.fvm.bnd_S,
            torch.tensor(
                [[-0.5, 0.0], [0.5, 0.0]],
                dtype=torch.float64,
            ),
        )
        self.fvm.rebuild_from_mesh(_one_cell_mesh())
        torch.testing.assert_close(
            self.fvm.V,
            torch.tensor([0.25], dtype=torch.float64),
        )
        self.fvm.nondimensionalize_geometry(2.0)
        with self.assertRaisesRegex(ValueError, "different"):
            self.fvm.nondimensionalize_geometry(1.0)

    def test_trainer_evaluates_every_flow_case_separately(self) -> None:
        """The global mesh forward produces one graph per configured case."""

        class CaseModel(torch.nn.Module):
            """Expose the encoded case value as predicted x-velocity."""

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                zeros = torch.zeros_like(inputs)
                return torch.cat((inputs, zeros, zeros), dim=1)

        class CaseLoss:
            """Minimal case-aware contract used by the Trainer helper."""

            bc_type = "soft"

            @staticmethod
            def build_input_tensor(
                x: torch.Tensor,
                y: torch.Tensor,
                sdf: torch.Tensor,
                boundary_types: torch.Tensor,
                case_ids: torch.Tensor,
            ) -> torch.Tensor:
                del y, sdf, boundary_types
                return case_ids.to(dtype=x.dtype).view(-1, 1)

            @staticmethod
            def apply_velocity_dirichlet_constraint(
                u: torch.Tensor,
                v: torch.Tensor,
                boundary_types: torch.Tensor,
                case_ids: torch.Tensor,
            ):
                del boundary_types, case_ids
                return u, v

            @staticmethod
            def get_freestream_targets(
                reference: torch.Tensor,
                case_ids: torch.Tensor,
            ):
                case = int(case_ids[0].item())
                if case == 0:
                    return torch.ones_like(reference), torch.zeros_like(reference)
                return torch.zeros_like(reference), torch.ones_like(reference)

        trainer = Trainer.__new__(Trainer)
        trainer.data_handler = SimpleNamespace(flow_cases={0: None, 1: None})
        trainer.model = CaseModel()
        trainer.loss_computer = CaseLoss()
        x = torch.zeros((2, 1), dtype=torch.float64)
        outputs = trainer._forward_fvm_cases(
            x,
            torch.zeros_like(x),
            torch.zeros_like(x),
            None,
        )

        self.assertEqual([case_id for case_id, _, _ in outputs], [0, 1])
        torch.testing.assert_close(
            outputs[0][1][:, 0],
            torch.zeros(2, dtype=torch.float64),
        )
        torch.testing.assert_close(
            outputs[1][1][:, 0],
            torch.ones(2, dtype=torch.float64),
        )
        torch.testing.assert_close(
            outputs[0][2],
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            outputs[1][2],
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
