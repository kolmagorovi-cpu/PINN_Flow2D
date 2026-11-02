"""Regression tests for mixed triangular/quadrilateral FVM meshes."""

from types import SimpleNamespace

import numpy as np

from airfoil_pinn.sampling.mesh import Mesh2D


def test_two_triangle_pairs_become_conservative_quads() -> None:
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
         [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    points = [
        SimpleNamespace(
            id=index,
            x=float(x),
            y=float(y),
            boundary_type="wall",
            on_airfoil=False,
            on_boundary=True,
        )
        for index, (x, y) in enumerate(coords)
    ]
    domain = coords[[0, 1, 2, 5, 4, 3, 0]]
    requested = np.array([coords[[0, 1, 4, 3]], coords[[1, 2, 5, 4]]])

    mesh = Mesh2D(points, domain_coords=domain, quad_cells_xy=requested)
    mesh.build(verbose=False)

    assert mesh.quads.shape == (2, 4)
    assert len(mesh.cells) == 2
    assert np.isclose(mesh.cv_volumes.sum(), 2.0)
    assert mesh._conservativity_ok is True
