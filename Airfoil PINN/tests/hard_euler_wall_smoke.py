"""Hard Euler walls preserve tangential flow and coordinate derivatives."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from airfoil_pinn.geometry.sdf import PolylineDistanceField
from airfoil_pinn.training.boundary import BoundaryOutputTransform


def _transform(vertices, scale=0.2):
    field = PolylineDistanceField(np.asarray(vertices), closed=False)
    return BoundaryOutputTransform(
        field, wall_kind="slip_wall", wall_enforcement="hard",
        distance_scale=scale,
    )


class HardEulerWallTests(unittest.TestCase):
    def test_unsampled_oblique_wall_preserves_tangent_and_pressure(self):
        vertices = [[-1.0, 0.0], [0.0, 0.4], [1.0, 0.0]]
        transform = _transform(vertices)
        field = transform.geometry
        points = field.segment_starts + 0.371 * field.segment_vectors
        normals = field.surface_normals(points)
        tangent = torch.stack((-normals[:, 1], normals[:, 0]), dim=1)
        raw = (3.0 * tangent + 7.0 * normals).requires_grad_(True)
        p = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
        u, v, pressure = transform(
            raw[:, 0:1], raw[:, 1:2], p, points[:, 0:1], points[:, 1:2]
        )
        velocity = torch.cat((u, v), dim=1)
        torch.testing.assert_close(velocity, 3.0 * tangent, atol=1e-12, rtol=0.0)
        self.assertIs(pressure, p)
        tangent_speed = (velocity * tangent).sum()
        derivative = torch.autograd.grad(tangent_speed, raw)[0]
        torch.testing.assert_close(derivative, tangent)
        # No input-coordinate autograd is needed for prediction/export.
        with torch.inference_mode():
            reversed_transform = _transform(vertices[::-1])
            reversed_uv = reversed_transform(
                raw[:, 0:1], raw[:, 1:2], p, points[:, 0:1], points[:, 1:2]
            )
        torch.testing.assert_close(reversed_uv[0], u)
        torch.testing.assert_close(reversed_uv[1], v)
        self.assertEqual(
            transform.contract()["wall"]["transform"],
            "normalized_distance_normal_projection",
        )

    def test_near_wall_limit_and_manufactured_derivatives(self):
        scale = 0.2
        transform = _transform([[-2.0, 0.0], [2.0, 0.0]], scale)
        x = torch.full((4, 1), 0.4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.7], [0.1], [0.01], [1e-6]],
                         dtype=torch.float64, requires_grad=True)
        raw_u = x.square() + y.square()
        raw_v = x.sin()
        u, v, _ = transform(raw_u, raw_v, torch.zeros_like(x), x, y)
        torch.testing.assert_close(u, raw_u)
        torch.testing.assert_close(v, y / (y + scale) * x.sin())
        v_x, v_y = torch.autograd.grad(v.sum(), (x, y), create_graph=True)
        v_yy = torch.autograd.grad(v_y.sum(), y)[0]
        torch.testing.assert_close(v_x, y / (y + scale) * x.cos())
        torch.testing.assert_close(v_y, scale * x.sin() / (y + scale).square())
        torch.testing.assert_close(v_yy, -2 * scale * x.sin() / (y + scale).pow(3))
        self.assertTrue(bool((v[1:] < v[:-1]).all()))
        self.assertGreater(float(u[-1].detach()), 0.1)

    def test_normal_derivatives_at_endpoints_and_shared_vertex(self):
        for vertices, coordinates in (
            ([[-1.0, 0.0], [1.0, 0.0]], [[1.4, 0.3], [0.2, 0.4]]),
            ([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
             [[-0.001, 1.4], [0.0, 1.4], [0.001, 1.4]]),
        ):
            with self.subTest(vertices=vertices):
                transform = _transform(vertices)
                points = torch.tensor(coordinates, dtype=torch.float64,
                                      requires_grad=True)
                self.assertFalse(bool(transform.geometry.tie_mask(points).any()))

                def evaluate(query):
                    x, y = query[:, 0:1], query[:, 1:2]
                    u, v, _ = transform(
                        x.sin() + y.square(), x.cos() - y,
                        torch.zeros_like(x), x, y,
                    )
                    return torch.cat((u, v), dim=1)

                self.assertTrue(torch.autograd.gradcheck(evaluate, (points,)))
                self.assertTrue(torch.autograd.gradgradcheck(evaluate, (points,)))

    def test_cached_wall_distance_does_not_create_nan_gradients(self):
        transform = _transform([[-1.0, 0.0], [1.0, 0.0]])
        x = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
        y = torch.zeros_like(x, requires_grad=True)
        u, v, _ = transform(
            x.sin() + y, x + y.cos(), torch.zeros_like(x), x, y,
            sdf=torch.zeros_like(x),
        )
        gradients = torch.autograd.grad((u + v).sum(), (x, y))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        torch.testing.assert_close(v, torch.zeros_like(v))


if __name__ == "__main__":
    unittest.main()
