#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GIF-анимация эволюции полей скорости Pipe-Bend PINN по чекпоинтам.

Для каждого чекпоинта pinn_<N>.pt строится кадр с полем скорости (scatter),
затем кадры собираются в GIF (matplotlib.animation.PillowWriter).

Использование:
    python scripts/make_training_gif.py \
        --experiment workspace/experiments/<run>
    # флаг --field u|v|p|speed задаёт поле (по умолчанию speed)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def load_model(checkpoint_path: str, device: torch.device):
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
    return model


def field_on_dataset(model, data_handler, device, dtype, field="speed", bs=4000):
    ds = data_handler.dataset
    u, v, p = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds.x), bs):
            j = min(i + bs, len(ds.x))
            inp = torch.cat([
                ds.x[i:j].to(device, dtype),
                ds.y[i:j].to(device, dtype),
                ds.sdf[i:j].to(device, dtype),
            ], dim=1)
            out = model(inp)
            u.append(out[:, 0].cpu().numpy())
            v.append(out[:, 1].cpu().numpy())
            p.append(out[:, 2].cpu().numpy())
    u = np.concatenate(u)
    v = np.concatenate(v)
    p = np.concatenate(p)
    if field == "u":
        return u
    if field == "v":
        return v
    if field == "p":
        return p
    return np.hypot(u, v)


def main() -> int:
    parser = argparse.ArgumentParser(description="GIF эволюции полей")
    parser.add_argument("--experiment", required=True, help="Директория эксперимента")
    parser.add_argument("--field", default="speed", choices=["u", "v", "p", "speed"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--output", default=None, help="Путь к GIF (по умолчанию в plots/)")
    args = parser.parse_args()

    chk_dir = os.path.join(args.experiment, "checkpoints")
    if not os.path.isdir(chk_dir):
        print(f"❌ Нет папки checkpoints: {chk_dir}")
        return 1
    checkpoints = []
    for name in os.listdir(chk_dir):
        m = re.fullmatch(r"pinn_(\d+)\.pt", name)
        if m:
            checkpoints.append((int(m.group(1)), os.path.join(chk_dir, name)))
    checkpoints.sort()
    if not checkpoints:
        print("❌ Чекпоинты не найдены")
        return 1
    print(f"🔍 Чекпоинтов: {len(checkpoints)} "
          f"(эпохи {checkpoints[0][0]}…{checkpoints[-1][0]})")

    device = torch.device(args.device)
    dtype = torch.float32
    first = torch.load(checkpoints[0][1], map_location=device, weights_only=False)
    config = first["config"]
    points_file = config.get("data", {}).get("points_file")
    if not points_file or not os.path.exists(points_file):
        points_file = os.path.join(PROJECT_DIR, "workspace", "initial", "fields",
                                   "initial.npy")
    from pipe_bend_pinn.data.handler import DataHandler

    data_handler = DataHandler(points_file, batch_size=4000, shuffle=False)
    ds = data_handler.dataset
    xp = ds.x.numpy().flatten()
    yp = ds.y.numpy().flatten()

    domain = None
    dp = os.path.join(PROJECT_DIR, "workspace", "geometry", "domains",
                      "pipe_bend_domain.npy")
    if os.path.exists(dp):
        domain = np.load(dp)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(12, 9))
    # Диапазон фиксируем по последнему чекпоинту, чтобы кадры были сопоставимы.
    last_model = load_model(checkpoints[-1][1], device)
    last_vals = field_on_dataset(last_model, data_handler, device, dtype, args.field)
    if args.field == "p":
        vmin = float(np.percentile(last_vals, 1))
        vmax = float(np.percentile(last_vals, 99))
    else:
        vmin, vmax = 0.0, float(np.percentile(last_vals, 99))
    del last_model

    sc = ax.scatter(xp, yp, c=last_vals, s=6, cmap="jet", vmin=vmin, vmax=vmax,
                    edgecolors="none")
    if domain is not None:
        ax.plot(domain[:, 0], domain[:, 1], "k-", linewidth=1.5, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax, label=f"{args.field} (нормированное)")
    ax.set_xlabel("x (норм.)"); ax.set_ylabel("y (норм.)")
    ax.axis("equal"); ax.grid(alpha=0.3)
    title = ax.set_title("")

    def update(frame):
        epoch, path = frame
        model = load_model(path, device)
        vals = field_on_dataset(model, data_handler, device, dtype, args.field)
        sc.set_array(vals)
        title.set_text(f"Эпоха {epoch} — поле {args.field}")
        return (sc, title)

    frames = [(epoch, path) for epoch, path in checkpoints]
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // max(1, args.fps),
                         blit=False)

    out_path = args.output or os.path.join(
        args.experiment, "plots", f"evolution_{args.field}.gif",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = PillowWriter(fps=max(1, args.fps))
    anim.save(out_path, writer=writer, dpi=100)
    print(f"✅ GIF сохранён: {out_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

