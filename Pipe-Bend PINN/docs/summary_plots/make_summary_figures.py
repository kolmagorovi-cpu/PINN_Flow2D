# -*- coding: utf-8 -*-
"""Графики для сводки по проверке гипотез (Pipe-Bend PINN vs CFD).

Формирует в текущем каталоге:
  fig1_radial_cp_cfd_vs_pinn.png
  fig2_inner_wall_dcp_bars.png
  fig3_pressure_range_balance.png
  fig4_collocation_refinement.png  (копия из workspace/pictures)
  fig5_field_p_v6_v10b.png
"""
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipe_bend_pinn.geometry.sdf import PolylineDistanceField
from scripts.integral_metrics import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(PROJECT, "workspace", "experiments")
W = 0.5
RHO = 998.0
UREF = 1.0
DYN = 0.5 * RHO * UREF ** 2

DOM = np.load(os.path.join(PROJECT, "workspace", "geometry", "domains", "pipe_bend_domain.npy"))
PDF = PolylineDistanceField(torch.tensor(DOM / W, dtype=torch.float64), closed=True)
INITIAL = os.path.join(PROJECT, "workspace", "initial", "fields", "initial.npy")

MODELS = [
    ("v5-final", "pipe_bend_laminar_20260828_170454"),
    ("v6-polish", "pipe_bend_refine_v6_polish_20260903_165657"),
    ("v10b (sigma=12)", "pipe_bend_refine_v10b_20260904_125648"),
    ("v12 (wall-P)", "pipe_bend_refine_v12_20260907_151221"),
    ("v13 (wall-mom)", "pipe_bend_exp_v13_wallmom_20260908_194936"),
    ("relu", "pipe_bend_exp_relu_20260908_201049"),
]


def load_cfd():
    """Читает export.csv Fluent и переводит в систему координат PINN."""
    path = os.path.join(PROJECT, "docs", "export.csv")
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    rows = []
    for line in lines:
        if re.fullmatch(r"\s*[-+0-9.eE]+(\s*,\s*[-+0-9.eE]+)+\s*", line):
            rows.append([float(v) for v in line.split(",")])
    data = np.array(rows)
    x = data[:, 0] + 1.0
    y = data[:, 1] - 1.0
    pressure = data[:, 3]
    cp = (pressure - 100000.0) / DYN
    return x, y, cp


def pinn_fields(model_path, xx, yy):
    """Предсказания PINN в физических координатах xx, yy (метры)."""
    model, _ = load_model(model_path, torch.device("cuda"))
    model.eval()
    xy_n = torch.tensor(np.c_[xx / W, yy / W], dtype=torch.float64)
    sdf = PDF(xy_n).double()
    with torch.no_grad():
        out = model(torch.cat([
            xy_n[:, 0:1].to("cuda", dtype=torch.float32),
            xy_n[:, 1:2].to("cuda", dtype=torch.float32),
            sdf.to("cuda", dtype=torch.float32),
        ], dim=1)).cpu().numpy()
    return out


def radial_lines(n=241):
    out = {}
    for theta in (-75, -45, -15):
        ang = np.deg2rad(theta)
        r = np.linspace(0.76, 1.24, n)
        out[theta] = (r, r * np.cos(ang), r * np.sin(ang))
    return out


BALANCE = {  # баланс масс (%) по данным integral_metrics
    "v5-final": 3.70,
    "v6-polish": 0.63,
    "v10b (sigma=12)": 0.09,
    "v12 (wall-P)": 3.42,
    "relu": 8.26,
    "tanh": 4.51,
}


def fig1_radial():
    from scipy.interpolate import LinearNDInterpolator

    cx, cy, ccp = load_cfd()
    interp = LinearNDInterpolator(np.c_[cx, cy], ccp)
    lines = radial_lines()
    series = [("CFD (Fluent)", None)]
    for name, exp in MODELS[:4]:
        series.append((name, os.path.join(EXP, exp, "checkpoints", "pinn_final.pt")))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    for ax, theta in zip(axes, (-75, -45, -15)):
        r, xx, yy = lines[theta]
        ax.plot(r, interp(xx, yy), "k-", lw=2.6, label="CFD (Fluent)")
        for (name, ckpt), ls in zip(series[1:], styles):
            out = pinn_fields(ckpt, xx, yy)
            ax.plot(r, out[:, 2], linestyle=ls, lw=1.7, label=name)
        ax.set_title("theta = %d deg" % theta)
        ax.set_xlabel("r, м")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("Cp = (p - 1e5)/(0.5 rho U^2)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Радиальные профили Cp в изгибе: CFD vs PINN", fontsize=13)
    out = os.path.join(HERE, "fig1_radial_cp_cfd_vs_pinn.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _inner_metrics(ckpt):
    r, xx, yy = radial_lines()[-45]
    cp = pinn_fields(ckpt, xx, yy)[:, 2]
    return cp[5], cp[235] - cp[5]


def fig2_bars():
    from scipy.interpolate import LinearNDInterpolator

    x0, y0, cp0 = load_cfd()
    interp = LinearNDInterpolator(np.c_[x0, y0], cp0)
    r, xx, yy = radial_lines()[-45]
    c = interp(xx, yy)
    names = ["CFD"]
    cp_in = [c[5]]
    dcp = [c[235] - c[5]]
    for name, exp in MODELS[1:]:
        ckpt = os.path.join(EXP, exp, "checkpoints", "pinn_final.pt")
        ci, dd = _inner_metrics(ckpt)
        names.append(name)
        cp_in.append(ci)
        dcp.append(dd)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    idx = np.arange(len(names))
    axes[0].bar(idx, cp_in, color=["#222222"] + ["#4C72B0"] * (len(names) - 1))
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Cp у внутренней стенки (r=0.77)")
    axes[0].set_title("Разрежение у внутренней стенки")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(idx, dcp, color=["#222222"] + ["#DD8452"] * (len(names) - 1))
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("dCp = Cp(1.23) - Cp(0.77)")
    axes[1].set_title("Перепад Cp поперёк канала (theta=-45)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Давление в изгибе: CFD-эталон и варианты PINN", fontsize=13)
    out = os.path.join(HERE, "fig2_inner_wall_dcp_bars.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_range_balance():
    data = np.load(INITIAL)
    xt = torch.tensor(np.ascontiguousarray(data["x_norm"]), dtype=torch.float32).view(-1, 1)
    yt = torch.tensor(np.ascontiguousarray(data["y_norm"]), dtype=torch.float32).view(-1, 1)
    st = torch.tensor(np.ascontiguousarray(data["sdf_norm"]), dtype=torch.float32).view(-1, 1)

    items = [
        ("v5-final", "pipe_bend_laminar_20260828_170454"),
        ("v6-polish", "pipe_bend_refine_v6_polish_20260903_165657"),
        ("v10b (sigma=12)", "pipe_bend_refine_v10b_20260904_125648"),
        ("v12 (wall-P)", "pipe_bend_refine_v12_20260907_151221"),
        ("relu", "pipe_bend_exp_relu_20260908_201049"),
        ("tanh", "pipe_bend_exp_tanh_20260908_202420"),
    ]
    names, ranges, balances = [], [], []
    for name, exp in items:
        ckpt = os.path.join(EXP, exp, "checkpoints", "pinn_final.pt")
        model, _ = load_model(ckpt, torch.device("cuda"))
        model.eval()
        with torch.no_grad():
            p = model(torch.cat([xt.cuda(), yt.cuda(), st.cuda()], dim=1)).cpu().numpy()[:, 2]
        names.append(name)
        ranges.append(float(p.max() - p.min()))
        balances.append(BALANCE.get(name, float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    idx = np.arange(len(names))
    axes[0].bar(idx, ranges, color="#55A868")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("max Cp - min Cp")
    axes[0].set_title("Размах Cp по области коллокации")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(idx, balances, color="#C44E52")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("ошибка баланса масс, %")
    axes[1].set_title("Баланс масс (integral_metrics)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Амплитуда давления vs качество скорости", fontsize=13)
    out = os.path.join(HERE, "fig3_pressure_range_balance.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_collocation():
    src = os.path.join(PROJECT, "workspace", "pictures", "refine_test.png")
    dst = os.path.join(HERE, "fig4_collocation_refinement.png")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
    return dst


def fig5_fields():
    p1 = os.path.join(EXP, "pipe_bend_refine_v6_polish_20260903_165657", "plots", "field_p_final.png")
    p2 = os.path.join(EXP, "pipe_bend_refine_v10b_20260904_125648", "plots", "field_p_final.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    axes[0].imshow(plt.imread(p1))
    axes[0].set_title("v6-polish (sigma=3)")
    axes[0].axis("off")
    axes[1].imshow(plt.imread(p2))
    axes[1].set_title("v10b (sigma=12)")
    axes[1].axis("off")
    fig.suptitle("Поле давления p, Па", fontsize=13)
    out = os.path.join(HERE, "fig5_field_p_v6_v10b.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (fig1_radial, fig2_bars, fig3_range_balance, fig4_collocation, fig5_fields):
        path = fn()
        print("saved:", path, "%.1f KB" % (os.path.getsize(path) / 1024.0))
