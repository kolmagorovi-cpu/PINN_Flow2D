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
from airfoil_pinn.sampling.core import CollocationPoints, SamplerConfig  # noqa: E402


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
    assert result.points_by_column.shape[:2] == result.valid_mask.shape
    assert result.points_by_column.shape[2] == 2
    assert len(result.column_surface_points) == result.points_by_column.shape[0]


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


def test_hybrid_transition_fills_sharp_trailing_edge_wake() -> None:
    """The background must not inherit an isotropic outer-layer hole at the TE."""
    domain = np.array(
        [
            [-0.5, -0.5],
            [1.5, -0.5],
            [1.5, 0.5],
            [-0.5, 0.5],
            [-0.5, -0.5],
        ],
        dtype=np.float64,
    )
    foil = np.array(
        [
            [1.0, 0.0],
            [0.75, 0.045],
            [0.25, 0.08],
            [0.0, 0.0],
            [0.25, -0.08],
            [0.75, -0.045],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    config = SamplerConfig(
        step_size=0.04,
        step_units="m",
        use_multiprocessing=False,
        show_progress=False,
        sampling_method="normal_layer",
        normal_layer_profile_step=0.01,
        normal_layer_first_height=0.005,
        normal_layer_count=10,
        normal_layer_growth_ratio=1.5,
        normal_layer_transition_layers=4,
    )
    sampler = CollocationPoints(config)
    sampler.foil_is_closed = True
    np.random.seed(123)

    points = sampler.sample_interior_normal_layer(domain, foil, 2000)
    xy = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    wake = (
        (xy[:, 0] > 1.03)
        & (xy[:, 0] < 1.18)
        & (np.abs(xy[:, 1]) < 0.04)
    )

    assert np.count_nonzero(wake) >= 5
    assert sampler._last_normal_layer_stats["far_field_filter"] == "layer_footprint"
    assert sampler._last_normal_layer_stats["transition_applied"] is True
    assert sampler._last_normal_layer_stats["transition_points_removed"] > 0


if __name__ == "__main__":
    test_closed_profile_points_are_in_flow()
    test_open_profile_uses_both_valid_sides()
    test_hybrid_transition_fills_sharp_trailing_edge_wake()
    print("OK normal_layer_smoke")
