"""Smoke tests for FVM/DataLoader/training refactor."""

import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from data_handler import DataHandler
from fvm_residuals import FVMResiduals
from models import PINNIncompressible
from solver_components.loss_computer import LossComputer
from solver_components.trainer import Trainer


def _load_config() -> dict:
    with open(os.path.join("Configs", "FVM7.yaml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["training"]["batch_size"] = 32
    cfg["data"]["num_workers"] = 0
    cfg["loss_weights"]["fvm"]["enabled"] = False
    cfg["loss_weights"]["fvm"]["weight"] = 0.0
    cfg["diagnostics"]["mass_metrics_interval"] = 10**9
    return cfg


def _input_dim(cfg: dict, handler: DataHandler) -> int:
    dim = 0
    inp = cfg["input_params"]
    if inp.get("coordinates", True):
        dim += 2
    if inp.get("sdf", True):
        dim += 1
    dim += len(handler.boundary_type_names)
    if inp.get("inlet_velocity", False):
        dim += 2
    if inp.get("viscosity", False):
        dim += 1
    if inp.get("density", False):
        dim += 1
    return dim


def test_fvm_checks(handler: DataHandler) -> None:
    fvm = FVMResiduals.from_mesh_data(handler.mesh_data).to("cpu", dtype=torch.float64)
    fvm.set_inlet_velocity(
        torch.tensor([handler.inlet_u_norm, handler.inlet_v_norm], dtype=torch.float64)
    )

    assert fvm.check_geometry_conservativity(tol=1e-8)
    assert fvm.check_conservativity(tol=1e-6)
    zeros = torch.zeros(fvm.N, 2, dtype=torch.float64)
    residual = fvm.compute_residual(zeros)
    assert residual.shape == (fvm.N,)
    assert torch.isfinite(residual).all()


def test_fvm_schedule() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.fvm_batch_interval = 1
    assert trainer._scheduled_fvm_batches(4, True) == {0, 1, 2, 3}
    trainer.fvm_batch_interval = 2
    assert trainer._scheduled_fvm_batches(5, True) == {1, 3}
    trainer.fvm_batch_interval = 10
    assert trainer._scheduled_fvm_batches(5, True) == set()
    assert trainer._scheduled_fvm_batches(5, False) == set()


def test_one_train_batch(cfg: dict, handler: DataHandler) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    loss_computer = LossComputer(
        config=cfg,
        data_handler=handler,
        rho_norm=torch.tensor(1.0, device=device, dtype=dtype),
        mu_norm=torch.tensor(1.0 / handler.Re, device=device, dtype=dtype),
        x_min=handler.x_min,
        x_max=handler.x_max,
        y_min=handler.y_min,
        y_max=handler.y_max,
        device=device,
        dtype=dtype,
    )
    model = PINNIncompressible(
        input_dim=_input_dim(cfg, handler),
        layers=[16, 16],
        activation="tanh",
        output_dim=3,
        output_bias=tuple(cfg["architecture"]["output_bias"]),
        coordinate_scaling=cfg["architecture"].get("coordinate_scaling"),
    ).to(device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_computer=loss_computer,
        data_handler=handler,
        config=cfg["training"],
        device=device,
        dtype=dtype,
    )

    batch = next(iter(handler.get_dataloader()))
    result = trainer.train_batch_adam(batch, accum_step=True)
    assert result["total"] >= 0.0
    assert torch.isfinite(torch.tensor(result["total"]))


def main() -> None:
    cfg = _load_config()
    handler = DataHandler(
        points_file=cfg["data"]["points_file"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        mesh_file=cfg["data"]["mesh_file"],
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=False,
    )

    test_fvm_checks(handler)
    test_fvm_schedule()
    test_one_train_batch(cfg, handler)
    print("OK refactor smoke tests")


if __name__ == "__main__":
    main()
