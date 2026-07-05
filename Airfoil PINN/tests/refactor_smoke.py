"""Smoke tests for FVM/DataLoader/training refactor."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.data.initializer import SCIPY_AVAILABLE, _compute_sdf_chunk
from airfoil_pinn.physics.fvm import FVMResiduals
from airfoil_pinn.training.components.loss_computer import LossComputer
from airfoil_pinn.training.components.trainer import Trainer
from airfoil_pinn.training.models import PINNIncompressible
from airfoil_pinn.utils import safe_grad


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

    if torch.cuda.is_available():
        fvm_cuda = FVMResiduals.from_mesh_data(handler.mesh_data).to("cuda", dtype=torch.float32)
        zeros_cuda = torch.zeros(
            fvm_cuda.N,
            2,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        residual_cuda = fvm_cuda.compute_residual(zeros_cuda)
        assert residual_cuda.shape == (fvm_cuda.N,)
        assert residual_cuda.device == zeros_cuda.device
        assert torch.isfinite(residual_cuda).all()


def test_fvm_schedule() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.fvm_batch_interval = 1
    assert trainer._scheduled_fvm_batches(4, True) == {0, 1, 2, 3}
    trainer.fvm_batch_interval = 2
    assert trainer._scheduled_fvm_batches(5, True) == {1, 3}
    trainer.fvm_batch_interval = 10
    assert trainer._scheduled_fvm_batches(5, True) == {4}
    assert trainer._scheduled_fvm_batches(5, False) == set()


def test_sdf_chunk() -> None:
    xs = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float64)
    wall_points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    expected = np.array([0.0, 1.0, 0.0, np.sqrt(5.0)], dtype=np.float64)

    fallback = _compute_sdf_chunk(
        indices,
        xs=xs,
        ys=ys,
        wall_points=wall_points,
        use_scipy=False,
    )
    assert np.allclose(fallback, expected)

    if SCIPY_AVAILABLE:
        scipy_result = _compute_sdf_chunk(
            indices,
            xs=xs,
            ys=ys,
            wall_points=wall_points,
            use_scipy=True,
        )
        assert np.allclose(scipy_result, expected)


def test_safe_grad_strict_mode() -> None:
    x = torch.tensor([[2.0]], requires_grad=True)
    y = x ** 2
    grad = safe_grad(y, x, silent=True, strict=True)
    assert torch.allclose(grad, torch.tensor([[4.0]]))

    detached_x = torch.tensor([[1.0]], requires_grad=False)
    assert torch.equal(
        safe_grad(torch.ones_like(detached_x), detached_x, silent=True),
        torch.zeros_like(detached_x),
    )
    try:
        safe_grad(torch.ones_like(detached_x), detached_x, silent=True, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict safe_grad must raise when x.requires_grad is False")


def test_rar_score_respects_pde_mask() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.rar_score_mode = "total_loss"
    trainer.loss_computer = SimpleNamespace(
        loss_weights={"cont": 1.0, "mom_x": 1.0, "mom_y": 1.0},
        pde_components=["cont", "mom_x", "mom_y"],
        bc_type="hard",
    )

    cont = torch.tensor([[1.0], [100.0]])
    mom_x = torch.tensor([[2.0], [200.0]])
    mom_y = torch.tensor([[3.0], [300.0]])
    zeros = torch.zeros_like(cont)
    pde_mask = torch.tensor([[True], [False]])

    score = trainer._compute_rar_point_score(
        cont=cont,
        mom1=mom_x,
        mom2=mom_y,
        u=zeros,
        v=zeros,
        p=zeros,
        x=zeros,
        y=zeros,
        boundary_types=None,
        pde_mask=pde_mask,
        pde_mult=1.0,
        w_pde=None,
        w_bc=None,
    )
    assert torch.allclose(score, torch.tensor([6.0, 0.0]))

    trainer.rar_score_mode = "pde"
    score_pde = trainer._compute_rar_point_score(
        cont=cont,
        mom1=mom_x,
        mom2=mom_y,
        u=zeros,
        v=zeros,
        p=zeros,
        x=zeros,
        y=zeros,
        boundary_types=None,
        pde_mask=pde_mask,
        pde_mult=1.0,
        w_pde=None,
        w_bc=None,
    )
    assert torch.allclose(score_pde, torch.tensor([6.0, 0.0]))


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
    test_sdf_chunk()
    test_safe_grad_strict_mode()
    test_rar_score_respects_pde_mask()
    test_one_train_batch(cfg, handler)
    print("OK refactor smoke tests")


if __name__ == "__main__":
    main()
