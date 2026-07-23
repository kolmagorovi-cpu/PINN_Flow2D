"""Analytical and autograd tests for the segment-based distance field."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from airfoil_pinn.geometry.sdf import (
    PolylineDistanceField,
    differentiable_sdf_for_interior,
)


class PolylineDistanceFieldTests(unittest.TestCase):
    """Verify values, derivatives, chain rule, tie detection and chunking."""

    def test_distance_is_to_segment_not_nearest_vertex(self) -> None:
        """The middle of a long segment must use its orthogonal projection."""
        field = PolylineDistanceField(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            closed=False,
            segment_chunk_size=1,
        )
        query = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

        self.assertTrue(
            torch.allclose(field(query), torch.tensor([[1.0]], dtype=torch.float64))
        )

    def test_numpy_cache_matches_online_engine(self) -> None:
        """Inference/preprocessing cache and autograd mode share exact values."""
        field = PolylineDistanceField(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            closed=False,
            segment_chunk_size=1,
        )
        query = np.array([[0.2, 0.5], [1.0, 1.0], [2.4, 0.3]])
        cached = field.evaluate_numpy(query, point_chunk_size=2)
        online = field(torch.tensor(query, dtype=torch.float64))

        np.testing.assert_allclose(
            cached,
            online.detach().numpy().reshape(-1),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_line_gradient_and_hessian(self) -> None:
        """Distance above a horizontal segment is locally ``d(x,y)=y``."""
        field = PolylineDistanceField(
            np.array([[-2.0, 0.0], [2.0, 0.0]]),
            closed=False,
        )
        query = torch.tensor(
            [[-0.5, 0.7], [0.4, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        distance = field(query)
        gradient = torch.autograd.grad(
            distance,
            query,
            grad_outputs=torch.ones_like(distance),
            create_graph=True,
        )[0]
        hessian_trace = torch.autograd.grad(
            gradient.sum(),
            query,
            create_graph=False,
        )[0]

        torch.testing.assert_close(
            distance,
            query[:, 1:2],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            gradient,
            torch.tensor(
                [[0.0, 1.0], [0.0, 1.0]],
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            hessian_trace,
            torch.zeros_like(query),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_autograd_gradcheck(self) -> None:
        """Double-precision finite differences agree away from SDF kinks."""
        field = PolylineDistanceField(
            np.array([[-2.0, 0.0], [2.0, 0.0]]),
            closed=False,
        )
        query = torch.tensor(
            [[-0.8, 0.4], [0.5, 0.9]],
            dtype=torch.float64,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                field,
                (query,),
                eps=1e-6,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.autograd.gradgradcheck(
                field,
                (query,),
                eps=1e-6,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_manufactured_chain_rule(self) -> None:
        """Coordinate gradients include the network's SDF dependency."""
        field = PolylineDistanceField(
            np.array([[-1.0, 0.0], [0.0, 0.0]]),
            closed=False,
        )
        x = torch.tensor([[0.6]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True)
        cached = torch.tensor([[1.0]], dtype=torch.float64)
        sdf = differentiable_sdf_for_interior(
            field,
            x,
            y,
            cached,
            torch.tensor([[True]]),
        )
        manufactured = x + 2.0 * sdf
        derivative = torch.autograd.grad(manufactured, x)[0]

        self.assertAlmostEqual(float(sdf.detach()), 1.0, places=12)
        self.assertAlmostEqual(float(derivative.detach()), 2.2, places=12)

    def test_boundary_cache_avoids_zero_distance_derivative(self) -> None:
        """Wall points use cached zero while interior points remain composed."""
        field = PolylineDistanceField(
            np.array([[-1.0, 0.0], [1.0, 0.0]]),
            closed=False,
        )
        x = torch.tensor(
            [[0.0], [0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torch.tensor(
            [[0.0], [0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cached = torch.tensor([[0.0], [9.0]], dtype=torch.float64)
        sdf = differentiable_sdf_for_interior(
            field,
            x,
            y,
            cached,
            torch.tensor([[False], [True]]),
        )
        gradient = torch.autograd.grad(
            sdf,
            y,
            grad_outputs=torch.ones_like(sdf),
        )[0]

        torch.testing.assert_close(
            sdf,
            torch.tensor([[0.0], [0.5]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            gradient,
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        )
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_tie_detection_and_eikonal_diagnostics(self) -> None:
        """The square centre is recognized as a medial-axis tie."""
        square = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ]
        )
        field = PolylineDistanceField(square, closed=True, segment_chunk_size=2)
        centre = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        nearest, second = field.nearest_two(centre)
        ties = field.tie_mask(nearest, second)

        self.assertTrue(bool(ties.item()))
        diagnostics = field.diagnostics(
            torch.tensor([[0.1, -0.4], [0.3, 0.2]], dtype=torch.float64)
        )
        self.assertAlmostEqual(diagnostics.eikonal_mean, 1.0, places=12)
        self.assertLess(diagnostics.eikonal_max_error, 1e-12)


if __name__ == "__main__":
    unittest.main()
