"""Analytical and autograd tests for the segment-based distance field."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from airfoil_pinn.geometry.sdf import (
    PolylineDistanceField,
    differentiable_sdf_for_interior,
)
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.components.rar import _canonical_sdf_numpy


def _real_airfoil_contour() -> np.ndarray:
    """Load the repository's NACA contour without filename-encoding coupling."""
    foil_dir = (
        Path(__file__).resolve().parents[1]
        / "workspace"
        / "geometry"
        / "foils"
    )
    matches = sorted(foil_dir.glob("NACA 0012*.txt"))
    if not matches:
        raise FileNotFoundError("NACA 0012 regression contour is unavailable")
    contour = np.loadtxt(matches[0], dtype=np.float64)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    return contour


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

    def test_real_airfoil_training_and_inference_float64_match(self) -> None:
        """The canonical engines agree on a real contour within 1e-12."""
        contour = _real_airfoil_contour()
        field = PolylineDistanceField(
            contour,
            closed=True,
            segment_chunk_size=17,
        )
        x = np.linspace(-0.15, 1.15, 29, dtype=np.float64)
        y = np.linspace(-0.20, 0.20, 23, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        query = np.column_stack((xx.ravel(), yy.ravel()))

        training = (
            field(torch.from_numpy(query))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        inference = field.evaluate_numpy(
            query,
            point_chunk_size=31,
        )
        max_abs_error = float(np.max(np.abs(training - inference)))

        self.assertLessEqual(max_abs_error, 1e-12)

    def test_stale_cached_sdf_and_fingerprint_are_rebuilt(self) -> None:
        """Cached values are derivative data, never trusted geometry state."""
        field = PolylineDistanceField(
            np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64),
            closed=False,
        )
        points = torch.tensor(
            [[0.25, 0.4], [1.0, 0.7], [1.75, 0.2]],
            dtype=torch.float64,
        )
        dataset = SimpleNamespace(
            x=points[:, 0:1].float(),
            y=points[:, 1:2].float(),
            sdf=torch.full((3, 1), 123.0, dtype=torch.float32),
        )
        handler = SimpleNamespace(
            dataset=dataset,
            cached_sdf_geometry_fingerprint="stale-geometry",
            cached_sdf_method="nearest_vertices",
            initial_metadata={},
        )
        loss_computer = LossComputer.__new__(LossComputer)
        loss_computer.sdf_module = field
        loss_computer.sdf_geometry_fingerprint = (
            field.geometry_fingerprint
        )
        loss_computer.sdf_float64_consistency_tolerance = 1e-12
        loss_computer.data_handler = handler

        loss_computer.synchronize_cached_training_sdf()
        cached_points = torch.cat(
            (dataset.x, dataset.y),
            dim=1,
        ).double().numpy()
        canonical = field.evaluate_numpy(cached_points)

        self.assertEqual(dataset.sdf.dtype, torch.float64)
        np.testing.assert_allclose(
            dataset.sdf.numpy().reshape(-1),
            canonical,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            handler.cached_sdf_geometry_fingerprint,
            field.geometry_fingerprint,
        )
        self.assertEqual(
            handler.cached_sdf_method,
            "exact_polyline_segments",
        )
        self.assertGreater(handler.cached_sdf_max_abs_error, 100.0)

    def test_rar_uses_canonical_segment_distance(self) -> None:
        """RAR midpoint distance is to the segment, not its vertices."""
        field = PolylineDistanceField(
            np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64),
            closed=False,
        )
        loss_computer = SimpleNamespace(
            sdf_module=field,
            evaluate_sdf_numpy=field.evaluate_numpy,
        )
        trainer = SimpleNamespace(loss_computer=loss_computer)

        distance = _canonical_sdf_numpy(
            trainer,
            np.array([[1.0, 0.5]], dtype=np.float64),
        )

        np.testing.assert_allclose(
            distance,
            np.array([0.5]),
            rtol=0.0,
            atol=0.0,
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
        ties = field.tie_mask(centre)

        self.assertTrue(bool(ties.item()))
        diagnostics = field.diagnostics(
            torch.tensor([[0.1, -0.4], [0.3, 0.2]], dtype=torch.float64)
        )
        self.assertAlmostEqual(diagnostics.eikonal_mean, 1.0, places=12)
        self.assertLess(diagnostics.eikonal_max_error, 1e-12)

    def test_real_airfoil_ties_preserve_near_wall_pde_coverage(self) -> None:
        """Shared-vertex ties on the real NACA contour are not medial axes."""
        contour_path = (
            Path(__file__).resolve().parents[1]
            / "workspace"
            / "geometry"
            / "foils"
            / "NACA 0012 (толщина 12%).txt"
        )
        contour = np.loadtxt(contour_path, dtype=np.float64)
        if np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]
        field = PolylineDistanceField(contour, closed=True)

        rolled = np.roll(contour, -1, axis=0)
        signed_area = 0.5 * np.sum(
            contour[:, 0] * rolled[:, 1]
            - rolled[:, 0] * contour[:, 1]
        )
        orientation = 1.0 if signed_area > 0.0 else -1.0

        previous_vectors = contour - np.roll(contour, 1, axis=0)
        next_vectors = rolled - contour

        def outward_normals(vectors: np.ndarray) -> np.ndarray:
            lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
            tangents = vectors / lengths
            return orientation * np.column_stack(
                (tangents[:, 1], -tangents[:, 0])
            )

        previous_normals = outward_normals(previous_vectors)
        next_normals = outward_normals(next_vectors)
        vertex_directions = previous_normals + next_normals
        direction_norm = np.linalg.norm(
            vertex_directions,
            axis=1,
            keepdims=True,
        )
        usable_vertices = direction_norm[:, 0] > 1e-8
        vertex_directions = (
            vertex_directions[usable_vertices]
            / direction_norm[usable_vertices]
        )
        vertex_samples = (
            contour[usable_vertices] + 0.003 * vertex_directions
        )

        segment_vectors = rolled - contour
        midpoints = 0.5 * (contour + rolled)
        midpoint_normals = outward_normals(segment_vectors)
        layers = np.array([0.0005, 0.002, 0.005, 0.009])
        layer_samples = np.concatenate(
            [
                midpoints + distance * midpoint_normals
                for distance in layers
            ],
            axis=0,
        )
        query = torch.tensor(
            np.concatenate((vertex_samples, layer_samples), axis=0),
            dtype=torch.float64,
        )

        nearest, second = field.nearest_two(query)
        distance_only_ties = field.tie_mask(nearest, second)
        geometric_ties = field.tie_mask(query)
        pde_mask = ~geometric_ties
        coverage = float(pde_mask.to(torch.float64).mean())
        near_wall_pde = (
            pde_mask
            & (nearest > 0.0)
            & (nearest < 0.01)
        )

        self.assertGreater(
            int(distance_only_ties.sum()),
            int(geometric_ties.sum()),
        )
        self.assertGreaterEqual(coverage, 0.95)
        self.assertTrue(bool(near_wall_pde.any()))

    def test_reduced_tolerances_preserve_real_fvm_mesh_coverage(self) -> None:
        """Reduced float64 tolerances must not erase coherent mesh regions."""
        root = Path(__file__).resolve().parents[1]
        contour_path = next(
            (
                root / "workspace" / "geometry" / "foils"
            ).glob("NACA 0012*.txt")
        )
        contour = np.loadtxt(contour_path, dtype=np.float64)
        if np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]
        field = PolylineDistanceField(
            contour,
            closed=True,
            segment_chunk_size=256,
        )

        mesh_path = root / "workspace" / "sampling" / "mesh" / "mesh2d.npz"
        with np.load(mesh_path, allow_pickle=False) as mesh:
            # A deterministic quarter of the full production mesh is enough
            # to catch the previous percent-level over-masking regression.
            coords = np.asarray(mesh["coords"], dtype=np.float64)[::4]
        query = torch.tensor(coords, dtype=torch.float64)

        reduced = field.tie_mask(
            query,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-10,
        )
        previous = field.tie_mask(
            query,
            absolute_tolerance=1e-7,
            relative_tolerance=1e-5,
        )
        reduced_fraction = float(
            reduced.to(torch.float64).mean()
        )
        previous_fraction = float(
            previous.to(torch.float64).mean()
        )

        self.assertLess(reduced_fraction, 1e-3)
        self.assertGreater(previous_fraction, 1e-2)
        self.assertGreater(previous_fraction, 20.0 * reduced_fraction)


if __name__ == "__main__":
    unittest.main()
