#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Интегральные метрики обученной модели Pipe-Bend PINN.

Считает по чекпоинту:
  1. Баланс масс: расход через вход Q_in и выход Q_out.
  2. Профиль Пуазейля на входе (относительная L2-ошибка vs целевой профиль).
  3. Прилипание на стенках (max/mean |u|).
  4. Перепад давления между входом и выходом.
  5. FVM-невязку неразрывности (fvm_mse / fvm_rms) на Mesh2D.

Использование:
    python scripts/integral_metrics.py \
        --checkpoint workspace/experiments/<run>/checkpoints/pinn_16000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def load_model(checkpoint_path: str, device: torch.device):
    """Восстанавливает модель по чекпоинту (архитектура из config внутри)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    arch = config["architecture"]
    from pipe_bend_pinn.training.models import PINNIncompressible

    model = PINNIncompressible(
        input_dim=3,
        layers=arch["hidden_layers"],
        activation=arch.get("activation", "gelu"),
        output_dim=arch.get("output_dim", 3),
        output_bias=arch.get("output_bias", [0.0, 0.0, 0.0]),
        coordinate_scaling=arch.get("coordinate_scaling"),
        fourier_features=arch.get("fourier_features"),
        residual=bool(arch.get("residual", False)),
    )
    model = model.to(device)
    state_dict = checkpoint["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def predict_dataset(model, data_handler, device, dtype, batch_size=4000):
    """Предсказывает (u, v, p) по всем точкам датасета."""
    ds = data_handler.dataset
    u, v, p = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds.x), batch_size):
            j = min(i + batch_size, len(ds.x))
            inp = torch.cat([
                ds.x[i:j].to(device, dtype),
                ds.y[i:j].to(device, dtype),
                ds.sdf[i:j].to(device, dtype),
            ], dim=1)
            out = model(inp)
            u.append(out[:, 0].cpu().numpy())
            v.append(out[:, 1].cpu().numpy())
            p.append(out[:, 2].cpu().numpy())
    return (np.concatenate(u), np.concatenate(v), np.concatenate(p))


def boundary_masks(data_handler):
    """Словарь {тип_границы: bool-маска по точкам датасета}."""
    ds = data_handler.dataset
    names = data_handler.boundary_type_names
    masks = {}
    if ds.boundary_types is not None:
        for idx, name in enumerate(names):
            if idx < ds.boundary_types.shape[1]:
                masks[name] = ds.boundary_types[:, idx].numpy() > 0.5
    return masks


def poiseuille_l2_error(u_inlet, u_target):
    """Относительная L2-ошибка профиля на входе."""
    norm = float(np.linalg.norm(u_target))
    if norm <= 0:
        return float("nan")
    return float(np.linalg.norm(u_inlet - u_target) / norm)


def fvm_metrics(model, data_handler, device, dtype):
    """FVM-невязка неразрывности на Mesh2D (fvm_mse, fvm_rms)."""
    from pipe_bend_pinn.app.paths import SAMPLING_COLLOCATION_DIR, GEOMETRY_DOMAINS_DIR
    from pipe_bend_pinn.physics.fvm import FVMResiduals, evaluate_fvm_metrics
    from pipe_bend_pinn.sampling.mesh import build_pipe_bend_mesh
    from pipe_bend_pinn.geometry.sdf import PolylineDistanceField

    colloc = os.path.join(SAMPLING_COLLOCATION_DIR, "collocation_points.npy")
    domain = os.path.join(GEOMETRY_DOMAINS_DIR, "pipe_bend_domain.npy")
    if not (os.path.exists(colloc) and os.path.exists(domain)):
        return None
    arr = np.load(colloc, allow_pickle=True)
    dom = np.load(domain)
    mesh = build_pipe_bend_mesh(arr, domain_coords=dom, verbose=False)
    fvm = FVMResiduals(mesh).to(device, dtype)
    coords = np.asarray(mesh.coords, dtype=np.float64)
    # SDF узлов: цепочки стенок из коллокации.
    wall = arr["boundary_type"] == "wall"
    pts = np.column_stack((arr["x"][wall], arr["y"][wall]))
    pts = np.vstack([pts, pts[0:1]])
    delta = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    threshold = 2.0 * max(float(np.median(delta)), 1e-12)
    breaks = np.where(delta > threshold)[0]
    sdf = None
    start = 0
    for b in np.append(breaks, len(pts) - 1):
        chain = pts[start:b + 1]
        start = b + 1
        if len(chain) >= 2:
            d = PolylineDistanceField(chain, closed=False).evaluate_numpy(coords)
            sdf = d if sdf is None else np.minimum(sdf, d)
    if sdf is None:
        return None

    # Модель обучена на НОРМИРОВАННЫХ входах (x/W, y/W, sdf/W) — нормируем.
    pipe_w = float(getattr(data_handler, 'pipe_width', 1.0) or 1.0)
    coords = coords / pipe_w
    sdf = np.asarray(sdf, dtype=np.float64) / pipe_w

    inlet_vel = None
    if "inlet" in data_handler.boundary_type_names:
        idx = data_handler.boundary_type_names.index("inlet")
        mask = data_handler.dataset.boundary_types[:, idx] > 0.5
        inlet_vel = torch.tensor(
            [data_handler.dataset.u_target[mask].mean().item(),
             data_handler.dataset.v_target[mask].mean().item()],
            device=device, dtype=dtype,
        )

    # FVM-невязка поля модели: вход НЕ предписан (мягкое ГУ), поэтому метрика
    # отражает собственно неразрывность предсказанного поля, а не рассогласование
    # с однородным prescribed-профилем на входе.
    mse, rms = evaluate_fvm_metrics(
        model, fvm, coords[:, 0], coords[:, 1], sdf,
        device, dtype, inlet_velocity=None,
    )
    return mse, rms


def main() -> int:
    parser = argparse.ArgumentParser(description="Интегральные метрики Pipe-Bend PINN")
    parser.add_argument("--checkpoint", required=True, help="Путь к чекпоинту")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    print(f"🔍 Чекпоинт: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    points_file = config.get("data", {}).get("points_file")
    if not points_file or not os.path.exists(points_file):
        points_file = os.path.join(PROJECT_DIR, "workspace", "initial", "fields",
                                   "initial.npy")
    from pipe_bend_pinn.data.handler import DataHandler

    data_handler = DataHandler(points_file, batch_size=4000, shuffle=False)
    print(f"📊 Данные: {points_file} | точек: {data_handler.n_points}")

    u, v, p = predict_dataset(model, data_handler, device, dtype)
    masks = boundary_masks(data_handler)
    ds = data_handler.dataset

    # 1. Баланс масс
    Q_in = float(u[masks["inlet"]].mean()) if masks.get("inlet") is not None else float("nan")
    Q_out = float(v[masks["outlet"]].mean()) if masks.get("outlet") is not None else float("nan")
    balance = abs(Q_in - Q_out) / max(abs(Q_in), 1e-12) * 100.0
    print("\n=== 1. Баланс масс (расход на единицу ширины) ===")
    print(f"   Q_in  = {Q_in:.6f}  (mean u на входе)")
    print(f"   Q_out = {Q_out:.6f}  (mean v на выходе)")
    print(f"   Ошибка баланса: {balance:.4f}%")

    # 2. Профиль Пуазейля на входе
    if masks.get("inlet") is not None:
        u_target_in = ds.u_target[masks["inlet"]].numpy().flatten()
        err_in = poiseuille_l2_error(u[masks["inlet"]], u_target_in)
        print("\n=== 2. Профиль Пуазейля на входе ===")
        print(f"   Относительная L2-ошибка: {err_in * 100:.4f}%")

    # 3. Прилипание на стенках
    if masks.get("wall") is not None:
        wall_speed = np.hypot(u[masks["wall"]], v[masks["wall"]])
        print("\n=== 3. Прилипание на стенках ===")
        print(f"   max|u| = {wall_speed.max():.2e} м/с, "
              f"mean|u| = {wall_speed.mean():.2e} м/с")

    # 4. Перепад давления
    if masks.get("inlet") is not None and masks.get("outlet") is not None:
        rho = float(data_handler.rho or 1.0)
        U_inf = float(data_handler.U_inf or 1.0)
        p_ref = float(data_handler.p_ref or 0.0)
        dcp = float(p[masks["inlet"]].mean() - p[masks["outlet"]].mean())
        dP_pa = dcp * (0.5 * rho * U_inf**2)
        print("\n=== 4. Перепад давления (вход → выход) ===")
        print(f"   ΔCp = {dcp:.4f}  →  ΔP = {dP_pa:.4f} Па")
        print(f"   (Cp = (p-p_ref)/(0.5·ρ·U²), U={U_inf} м/с, ρ={rho} кг/м³, "
              f"p_ref={p_ref} Па)")

    # 5. FVM-невязка
    print("\n=== 5. FVM-невязка неразрывности (Mesh2D) ===")
    fm = fvm_metrics(model, data_handler, device, dtype)
    if fm is not None:
        print(f"   fvm_mse = {fm[0]:.3e}, fvm_rms = {fm[1]:.3e}")
    else:
        print("   FVM-метрики недоступны (нет файлов коллокации/домена)")

    # Область в целом
    speed = np.hypot(u, v)
    print("\n=== Область в целом ===")
    print(f"   mean|u| = {speed.mean():.4f} м/с, max|u| = {speed.max():.4f} м/с")
    return 0


if __name__ == "__main__":
    sys.exit(main())

