"""Smoke test for boundary-layer diagnostics output."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airfoil_pinn.training.boundary_layer_diagnostics import (  # noqa: E402
    run_boundary_layer_diagnostics,
)


class DummyBoundaryLayerModel(torch.nn.Module):
    """Small analytic model with a smooth wall-normal velocity profile."""

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Return u=1-exp(-sdf/delta), v=0 and a simple Cp-like pressure."""

        x = inp[:, 0:1]
        sdf = inp[:, 2:3]
        u = 1.0 - torch.exp(-sdf / 0.01)
        v = torch.zeros_like(u)
        p = -0.1 * x
        return torch.cat([u, v, p], dim=1)


class DummyLossComputer:
    """Minimal adapter exposing the input API used by diagnostics."""

    @staticmethod
    def evaluate_sdf_tensor(
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact distance to the dummy flat-plate wall."""

        del x
        return torch.abs(y)

    def build_input_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate the same logical inputs as the real LossComputer."""

        return torch.cat([x, y, sdf], dim=1)

    def apply_velocity_dirichlet_constraint(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        boundary_types: torch.Tensor,
    ):
        """Return velocities unchanged; the dummy case has no velocity-Dirichlet faces."""

        return u, v


class DummyDataHandler:
    """Geometry and physical metadata needed by the diagnostics module."""

    def __init__(self) -> None:
        """Build a rectangular domain with an open flat-plate wall."""

        self.chord = 1.0
        self.U_inf = 1.0
        self.rho = 1.0
        self.mu = 1.0e-3
        self.Re = 1000.0
        self.boundary_type_names = ["interior", "wall"]
        self.mesh_domain_coords = np.array(
            [
                [-0.2, -0.2],
                [1.2, -0.2],
                [1.2, 0.2],
                [-0.2, 0.2],
                [-0.2, -0.2],
            ],
            dtype=np.float64,
        )
        x = np.linspace(0.0, 1.0, 51)
        self.mesh_airfoil_coords = np.column_stack([x, np.zeros_like(x)])


def test_boundary_layer_diagnostics_outputs_files() -> None:
    """Run diagnostics and verify that metric artifacts are produced."""

    model = DummyBoundaryLayerModel()
    data_handler = DummyDataHandler()
    config = {
        "enabled": True,
        "interval": 1,
        "sections": [0.25, 0.5, 0.75],
        "profile_step": 0.1,
        "first_layer_height": 0.001,
        "layer_count": 12,
        "growth_ratio": 1.35,
        "max_normal_distance": 0.08,
        "save_profiles_npz": True,
        "save_metrics_csv": True,
        "plot_profiles": False,
        "plot_cp": False,
        "plot_cf": False,
        "plot_delta99": False,
        "plot_deficit": False,
    }

    with tempfile.TemporaryDirectory() as tmp:
        result = run_boundary_layer_diagnostics(
            model=model,
            loss_computer=DummyLossComputer(),
            data_handler=data_handler,
            device=torch.device("cpu"),
            dtype=torch.float32,
            savedir=tmp,
            epoch=1,
            config=config,
        )

        assert result is not None
        assert result.summary["profile_count"] > 0
        assert Path(result.files["metrics_csv"]).exists()
        assert Path(result.files["profiles_npz"]).exists()


if __name__ == "__main__":
    test_boundary_layer_diagnostics_outputs_files()
    print("OK boundary_layer_diagnostics_smoke")
