"""Constrained-edge regression cases for thin normal-layer cells."""

import unittest
from collections import Counter

import numpy as np

from airfoil_pinn.sampling.mesh.core import (
    _cross2d,
    _insert_constrained_edge,
    _triangulate_polygon_delaunay,
)


class CDTCavityTests(unittest.TestCase):
    def insert(self, coords, triangles, s, t):
        coords = np.asarray(coords, dtype=float)
        original = coords.copy()
        tris = dict(enumerate(triangles))
        edges, vertices = {}, {}
        for tid, tri in tris.items():
            for u, v in zip(tri, (*tri[1:], tri[0])):
                edges.setdefault(tuple(sorted((u, v))), set()).add(tid)
            for v in tri:
                vertices.setdefault(v, set()).add(tid)
        original_boundary = {edge for edge, ids in edges.items() if len(ids) == 1}
        before_area = sum(abs(_cross2d(*coords[list(tri)])) / 2 for tri in triangles)
        _, split = _insert_constrained_edge(s, t, tris, coords, edges, vertices, len(tris))
        self.assertFalse(split)
        self.assertIn(tuple(sorted((s, t))), edges)
        self.assertTrue(all(len(ids) <= 2 for ids in edges.values()))
        self.assertEqual(original_boundary, {e for e, ids in edges.items() if len(ids) == 1})
        areas = [abs(_cross2d(*coords[list(tri)])) / 2 for tri in tris.values()]
        self.assertTrue(all(area > 1e-18 for area in areas))
        self.assertAlmostEqual(before_area, sum(areas), places=12)
        np.testing.assert_array_equal(coords, original)

    def test_both_endpoint_triangles_are_replaced(self):
        self.insert([(0, 0), (1, 0), (1, 1), (0, 1)],
                    [(0, 1, 2), (0, 2, 3)], 1, 3)

    def test_boundary_chain_doubles_back_past_endpoint(self):
        # Vertices 2, 3, 4 are collinear; projection sorting reverses the
        # lower cavity chain and used to create overlaps/zero-area cells.
        self.insert([(0, 1), (0, 0), (1, 0), (1.01, -0.1), (1.02, -0.2)],
                    [(3, 0, 4), (2, 0, 3), (0, 1, 4)], 1, 2)

    def test_collinear_boundary_vertices_survive_at_small_scale(self):
        for scale in (1.0, 1e-4):
            coords = scale * np.array([(0, 0), (2, 0), (2, 1), (1, 1), (0, 1)])
            tris = _triangulate_polygon_delaunay(list(range(5)), coords)
            self.assertEqual(len(tris), 3)
            self.assertTrue(all(_cross2d(*coords[list(tri)]) > 0 for tri in tris))
            edges = Counter(tuple(sorted((u, v))) for tri in tris
                            for u, v in zip(tri, (*tri[1:], tri[0])))
            self.assertEqual({e for e, count in edges.items() if count == 1},
                             {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)})


if __name__ == '__main__':
    unittest.main()
