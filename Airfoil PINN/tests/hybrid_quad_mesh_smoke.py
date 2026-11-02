"""Regression tests for mixed triangular/quadrilateral FVM meshes."""

from types import SimpleNamespace

import numpy as np

from airfoil_pinn.sampling.mesh import Mesh2D, build_dual_face_quadrature


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

    quadrature = build_dual_face_quadrature(
        mesh.coords,
        mesh.cells,
        np.asarray([face.node_i for face in mesh.internal_faces]),
        np.asarray([face.node_j for face in mesh.internal_faces]),
        np.asarray([face.area_vector for face in mesh.internal_faces]),
        np.asarray([face.node for face in mesh.boundary_faces]),
        np.asarray([face.neighbor_node for face in mesh.boundary_faces]),
    )
    closure = np.zeros((len(mesh.coords), 2), dtype=np.float64)
    np.add.at(
        closure,
        quadrature["int_quad_node_i"],
        quadrature["int_quad_area"],
    )
    np.add.at(
        closure,
        quadrature["int_quad_node_j"],
        -quadrature["int_quad_area"],
    )
    np.add.at(
        closure,
        np.asarray([face.node for face in mesh.boundary_faces]),
        np.asarray([face.area_vector for face in mesh.boundary_faces]),
    )
    np.testing.assert_allclose(closure, 0.0, atol=1e-14, rtol=0.0)
