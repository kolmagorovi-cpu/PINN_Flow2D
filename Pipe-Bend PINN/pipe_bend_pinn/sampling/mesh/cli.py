#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Интерактивный модуль построения FVM-сетки отвода (Mesh2D)."""

from __future__ import annotations

import os

import numpy as np
import torch

from pipe_bend_pinn.app.paths import (
    GEOMETRY_DOMAINS_DIR,
    SAMPLING_COLLOCATION_DIR,
)
from pipe_bend_pinn.physics.fvm import FVMResiduals
from pipe_bend_pinn.sampling.mesh import build_pipe_bend_mesh


def m2d() -> None:
    """Строит FVM-сетку из текущей коллокации и выводит диагностику."""
    print("\n" + "=" * 60)
    print("FVM-СЕТКА (MESH2D)".center(60))
    print("=" * 60)

    colloc_path = os.path.join(SAMPLING_COLLOCATION_DIR, "collocation_points.npy")
    domain_path = os.path.join(GEOMETRY_DOMAINS_DIR, "pipe_bend_domain.npy")

    if not os.path.exists(colloc_path):
        print(f"❌ Файл коллокации не найден: {colloc_path}")
        print("   Сначала выполните 'Дискретизация (коллокация)'.")
        input("Нажмите Enter...")
        return
    if not os.path.exists(domain_path):
        print(f"❌ Файл области не найден: {domain_path}")
        print("   Сначала выполните 'Геометрия отвода 90°'.")
        input("Нажмите Enter...")
        return

    try:
        points = np.load(colloc_path, allow_pickle=True)
        domain = np.load(domain_path)
        mesh = build_pipe_bend_mesh(points, domain_coords=domain, verbose=True)

        print("\n✅ FVM-сетка построена:")
        print(f"   Узлов:        {mesh.coords.shape[0]}")
        print(f"   Треугольников:{len(mesh.triangles)}")
        print(f"   Консервативность: {mesh.is_conservative()}")

        fvm = FVMResiduals(mesh).to('cpu', torch.float32)
        print(f"   FVM: N={fvm.N}, внутр. граней M={fvm.M}, граничных B={fvm.B}")

        u_zero = torch.zeros(fvm.N, 2, dtype=torch.float32)
        loss_zero = float(fvm.mass_loss(u_zero, reduction='divergence_l2_volume').item())
        print(f"   Массовая невязка нулевого поля: {loss_zero:.2e}")

        summary = mesh.summary()
        if isinstance(summary, dict):
            for key, value in summary.items():
                print(f"   {key}: {value}")

    except Exception as exc:
        print(f"❌ Ошибка построения сетки: {exc}")
        import traceback
        traceback.print_exc()

    input("\nНажмите Enter для возврата...")


if __name__ == "__main__":
    m2d()
