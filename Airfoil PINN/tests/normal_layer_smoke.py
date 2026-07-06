"""Smoke tests for normal-layer sampling."""

from pathlib import Path
import sys

import numpy as np
from matplotlib.path import Path as MplPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airfoil_pinn.sampling.normal_layer import (  # noqa: E402
    NormalLayerConfig,
    generate_normal_layer_points,
)


def _rect_domain() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0],
            [2.0, -1.0],
            [2.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )


def _closed_ellipse() -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    return np.column_stack([0.5 + 0.5 * np.cos(theta), 0.08 * np.sin(theta)])


def _open_plate() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 41)
    return np.column_stack([x, np.zeros_like(x)])


def test_closed_profile_points_are_in_flow() -> None:
    domain = _rect_domain()
    foil = _closed_ellipse()
    cfg = NormalLayerConfig(
        mode="manual",
        profile_step=0.05,
        first_layer_height=0.005,
        layer_count=5,
        growth_ratio=1.35,
    )
    result = generate_normal_layer_points(domain, foil, cfg)
    assert len(result.points) > 0
    assert np.all(MplPath(domain).contains_points(result.points))
    assert not np.any(MplPath(foil).contains_points(result.points))


def test_open_profile_uses_both_valid_sides() -> None:
    domain = _rect_domain()
    foil = _open_plate()
    cfg = NormalLayerConfig(
        mode="manual",
        profile_step=0.05,
        first_layer_height=0.01,
        layer_count=4,
        growth_ratio=1.3,
        open_profile_sides="both",
    )
    result = generate_normal_layer_points(domain, foil, cfg)
    assert len(result.points) > 0
    assert np.any(result.points[:, 1] > 0.0)
    assert np.any(result.points[:, 1] < 0.0)
    assert np.all(MplPath(domain).contains_points(result.points))


if __name__ == "__main__":
    test_closed_profile_points_are_in_flow()
    test_open_profile_uses_both_valid_sides()
    print("OK normal_layer_smoke")
