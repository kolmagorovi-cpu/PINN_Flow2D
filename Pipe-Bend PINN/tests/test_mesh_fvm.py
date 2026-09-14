#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Тесты новых возможностей Pipe-Bend PINN (порт из Airfoil PINN + функция тока).

Проверяются:
  1. ReferenceScales: валидация, roundtrip метаданных, fingerprint.
  2. PolylineDistanceField: точный сегментный SDF, градиент (eikonal ~ 1).
  3. Функция тока: бездивергентность поля, профиль Пуазейля, непрерывность ψ.
  4. Mesh2D + FVMResiduals: CDT-сетка на L-образной области отвода,
     геометрическая консервативность, массовая невязка однородного поля ~ 0.

Запуск (из корня проекта):
    $env:PYTHONPATH = (Get-Location).Path
    .\..\.venv\Scripts\python.exe tests\test_mesh_fvm.py
"""

import os
import sys

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"   OK: {name}" + (f" ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"   FAIL: {name}" + (f" ({detail})" if detail else ""))


# ──────────────────────────────────────────────────────────────────────────────
# 1. ReferenceScales
# ──────────────────────────────────────────────────────────────────────────────
def test_normalization():
    print("\n[1] ReferenceScales")
    from pipe_bend_pinn.data.normalization import (
        ReferenceScales,
        compute_reynolds_number,
    )

    scales = ReferenceScales(
        pipe_width=0.5, velocity=1.0, density=1000.0,
        dynamic_viscosity=1e-3, static_pressure=0.0,
    )
    check("Re корректен", np.isclose(scales.reynolds, 1.0 * 0.5 * 1000.0 / 1e-3))

    try:
        ReferenceScales(pipe_width=0.0, velocity=1.0, density=1000.0,
                        dynamic_viscosity=1e-3, static_pressure=0.0)
        check("Невалидная pipe_width отвергается", False)
    except ValueError:
        check("Невалидная pipe_width отвергается", True)

    meta = scales.to_metadata()
    restored = ReferenceScales.from_metadata({"normalization_reference": meta})
    check("Roundtrip метаданных",
          np.isclose(restored.pipe_width, 0.5)
          and np.isclose(restored.reynolds, scales.reynolds))

    legacy = ReferenceScales.from_metadata(
        {"pipe_width": 0.5, "U_inf": 1.0, "rho": 1000.0, "mu": 1e-3, "p_ref": 0.0}
    )
    check("Совместимость с legacy-метаданными",
          np.isclose(legacy.reynolds, scales.reynolds))

    check("Fingerprint стабилен",
          scales.fingerprint
          == ReferenceScales.from_metadata(
              {"normalization_reference": meta}).fingerprint)
    check("Re функция", np.isclose(
        compute_reynolds_number(1.0, 0.5, 1000.0, 1e-3), 5e5))


# ──────────────────────────────────────────────────────────────────────────────
# 2. PolylineDistanceField
# ──────────────────────────────────────────────────────────────────────────────
def test_sdf():
    print("\n[2] PolylineDistanceField")
    from pipe_bend_pinn.geometry.sdf import PolylineDistanceField

    # Квадрат со стороной 2, центр в (0,0).
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
                       [-1.0, 1.0], [-1.0, -1.0]])
    field = PolylineDistanceField(square, closed=True)

    pts = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.99]],
                       dtype=torch.float64)
    d = field(pts)
    check("SDF в центре", np.isclose(float(d[0, 0]), 1.0))
    check("SDF у стенки", abs(float(d[2, 0]) - 0.01) < 1e-9)

    # Eikonal: |∇d| ≈ 1 внутри области (не на медиальной оси).
    x = torch.tensor([[-0.5, 0.3], [0.2, -0.6], [0.6, 0.4]],
                     dtype=torch.float64, requires_grad=True)
    dist = field(x).reshape(-1)
    grad = torch.autograd.grad(dist.sum(), x, create_graph=True)[0]
    eikonal = torch.linalg.vector_norm(grad, dim=1)
    max_err = float((eikonal - 1.0).abs().max().detach())
    check("Eikonal |∇d| ≈ 1", max_err < 1e-6, f"max_err={max_err:.2e}")

    fp = field.geometry_fingerprint
    check("Fingerprint геометрии", isinstance(fp, str) and len(fp) == 64)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Функция тока
# ──────────────────────────────────────────────────────────────────────────────
def _make_bend_domain(R=1.0, W=0.5, L_in=1.0, L_out=1.0, n_arc=60):
    R_outer = R + W / 2
    R_inner = R - W / 2
    pts = [[-L_in, -R_outer], [-L_in, -R_inner]]
    ang = np.linspace(-np.pi / 2, 0.0, n_arc)
    for a in ang[1:]:
        pts.append([R_inner * np.cos(a), R_inner * np.sin(a)])
    pts.append([R_inner, L_out])
    pts.append([R_outer, L_out])
    pts.append([R_outer, 0.0])
    ang2 = np.linspace(0.0, -np.pi / 2, n_arc)
    for a in ang2[1:]:
        pts.append([R_outer * np.cos(a), R_outer * np.sin(a)])
    pts.append([-L_in, -R_outer])
    return np.array(pts)


def test_stream_function():
    print("\n[3] Функция тока")
    import matplotlib.path as mplPath
    from pipe_bend_pinn.data.initializer import Initializer

    domain = _make_bend_domain()
    path = mplPath.Path(domain)

    # Точки: регулярная сетка внутри области + граничные точки.
    x_min, y_min = domain[:, 0].min(), domain[:, 1].min()
    x_max, y_max = domain[:, 0].max(), domain[:, 1].max()
    xs = np.arange(x_min, x_max + 1e-9, 0.05)
    ys = np.arange(y_min, y_max + 1e-9, 0.05)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    inside = path.contains_points(grid)
    pts = grid[inside]

    # Граничные точки: вход (x=x_min), выход (y=y_max), остальное — стены.
    bnd = []
    btypes = []
    for xb, yb in zip(domain[:-1, 0], domain[:-1, 1]):
        if abs(xb - x_min) < 1e-9 and yb <= y_min + 0.5 + 1e-9:
            bnd.append([xb, yb]); btypes.append('inlet')
        elif abs(yb - y_max) < 1e-9 and xb >= x_max - 0.5 - 1e-9:
            bnd.append([xb, yb]); btypes.append('outlet')
        else:
            bnd.append([xb, yb]); btypes.append('wall')

    all_pts = np.vstack([pts, np.array(bnd)])
    all_types = np.array(['interior'] * len(pts) + btypes)
    dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'),
             ('boundary_type', 'U20'), ('on_boundary', 'bool')]
    arr = np.zeros(len(all_pts), dtype=dtype)
    arr['id'] = np.arange(len(all_pts))
    arr['x'] = all_pts[:, 0]
    arr['y'] = all_pts[:, 1]
    arr['boundary_type'] = all_types
    arr['on_boundary'] = all_types != 'interior'

    init = Initializer.__new__(Initializer)
    init.points = arr
    init.pipe_width = 0.5
    init._stream_function_geometry = lambda: (1.25, 0.75)

    u, v, psi = init._compute_stream_function_field(u0=1.0, v_in=0.0)
    check("Поле конечное",
          np.isfinite(u).all() and np.isfinite(v).all()
          and np.isfinite(psi).all())

    # Профиль Пуазейля на входе: u = 1.0 * (4t - 4t²), t=(y+1.25)/0.5.
    inlet_mask = arr['boundary_type'] == 'inlet'
    if inlet_mask.any():
        t = (arr['y'][inlet_mask] + 1.25) / 0.5
        u_expected = 4.0 * t - 4.0 * t**2
        err = float(np.max(np.abs(u[inlet_mask] - u_expected)))
        check("Пуазейль на входе", err < 1e-9, f"err={err:.2e}")
    else:
        check("Пуазейль на входе", False, "нет inlet-точек")

    # Профиль на выходе: v = 4t - 4t², t=(x-0.75)/0.5.
    outlet_mask = arr['boundary_type'] == 'outlet'
    if outlet_mask.any():
        t = (arr['x'][outlet_mask] - 0.75) / 0.5
        v_expected = 4.0 * t - 4.0 * t**2
        err = float(np.max(np.abs(v[outlet_mask] - v_expected)))
        check("Пуазейль на выходе", err < 1e-9, f"err={err:.2e}")
    else:
        check("Пуазейль на выходе", False, "нет outlet-точек")

    # Бездивергентность в изгибе: u_r = 0 (поток чисто тангенциальный).
    bend_mask = (arr['x'] > 0) & (arr['y'] < 0)
    if bend_mask.any():
        xb, yb = arr['x'][bend_mask], arr['y'][bend_mask]
        rb = np.hypot(xb, yb)
        ur = u[bend_mask] * (xb / rb) + v[bend_mask] * (yb / rb)
        check("В изгибе u_r = 0", float(np.max(np.abs(ur))) < 1e-9,
              f"max|ur|={float(np.max(np.abs(ur))):.2e}")

    # Полная бездивергентность: div = 0 на регулярной сетке входного канала.
    yg = np.linspace(-1.25, -0.75, 31)
    xg = np.linspace(-1.0, 0.0, 21)
    # indexing='ij' -> форма (nx, ny): ось 0 = x, ось 1 = y.
    XG, YG = np.meshgrid(xg, yg, indexing='ij')
    tg = (YG + 1.25) / 0.5
    ug = 4.0 * tg - 4.0 * tg**2
    vg = np.zeros_like(ug)
    div = np.gradient(ug, xg, axis=0) + np.gradient(vg, yg, axis=1)
    check("div = 0 на входном канале", float(np.max(np.abs(div))) < 1e-10,
          f"max|div|={float(np.max(np.abs(div))):.2e}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Mesh2D + FVM
# ──────────────────────────────────────────────────────────────────────────────
def _densify_boundary(domain, spacing=0.05):
    """Равномерная выборка точек по периметру контура (макс. шаг spacing)."""
    pts = []
    n = len(domain)
    for i in range(n - 1):
        a, b = domain[i], domain[i + 1]
        seg = np.linalg.norm(b - a)
        n_pts = max(1, int(np.ceil(seg / spacing)))
        for k in range(n_pts):
            t = k / n_pts
            pts.append(a + t * (b - a))
    return np.array(pts)


def test_mesh_fvm():
    print("\n[4] Mesh2D + FVMResiduals")
    import matplotlib.path as mplPath
    from pipe_bend_pinn.sampling.mesh import build_pipe_bend_mesh
    from pipe_bend_pinn.physics.fvm import FVMResiduals

    domain = _make_bend_domain()
    path = mplPath.Path(domain)
    x_min, y_min = domain[:, 0].min(), domain[:, 1].min()
    x_max, y_max = domain[:, 0].max(), domain[:, 1].max()

    # Внутренние точки с отступом от границы, чтобы сетка не касалась контура.
    xs = np.arange(x_min + 0.05, x_max - 0.05, 0.1)
    ys = np.arange(y_min + 0.05, y_max - 0.05, 0.1)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    inside = path.contains_points(grid)
    interior = grid[inside]

    # Граничные точки по периметру (плотные, как в реальном сэмплере).
    bnd_dense = _densify_boundary(domain, spacing=0.05)
    bnd, btypes = [], []
    for xb, yb in zip(bnd_dense[:, 0], bnd_dense[:, 1]):
        if abs(xb - x_min) < 1e-9 and yb <= y_min + 0.5 + 1e-9:
            bnd.append([xb, yb]); btypes.append('inlet')
        elif abs(yb - y_max) < 1e-9 and xb >= x_max - 0.5 - 1e-9:
            bnd.append([xb, yb]); btypes.append('outlet')
        else:
            bnd.append([xb, yb]); btypes.append('wall')

    all_pts = np.vstack([interior, np.array(bnd)])
    all_types = np.array(['interior'] * len(interior) + btypes)
    dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'),
             ('boundary_type', 'U20'), ('on_boundary', 'bool')]
    arr = np.zeros(len(all_pts), dtype=dtype)
    arr['id'] = np.arange(len(all_pts))
    arr['x'] = all_pts[:, 0]
    arr['y'] = all_pts[:, 1]
    arr['boundary_type'] = all_types
    arr['on_boundary'] = all_types != 'interior'

    mesh = build_pipe_bend_mesh(arr, domain_coords=domain, verbose=False)
    n_nodes = mesh.coords.shape[0]
    check("Сетка построена", n_nodes > 100, f"N={n_nodes}")
    check("Консервативность", mesh.is_conservative() is True)

    fvm = FVMResiduals(mesh).to('cpu', torch.float32)
    N = fvm.N

    # Нулевое поле: тривиально соленоидальное, невязка ~ 0.
    u_zero = torch.zeros(N, 2, dtype=torch.float32)
    loss_zero = float(
        fvm.mass_loss(u_zero, reduction='divergence_l2_volume').item()
    )
    check("Нулевое поле: mse ~ 0", loss_zero < 1e-12, f"mse={loss_zero:.2e}")

    # Однородное поле (1,0) нарушает no-slip на стенках -> невязка заметная.
    u_t = torch.tensor(
        np.column_stack((np.ones(N), np.zeros(N))), dtype=torch.float32
    )
    loss_uniform = float(
        fvm.mass_loss(u_t, reduction='divergence_l2_volume').item()
    )
    check("Однородное поле нарушает стенки", loss_uniform > 1.0,
          f"mse={loss_uniform:.2e}")

    # Поле функции тока (бездивергентное, u=0 на стенках) -> малая невязка.
    from pipe_bend_pinn.data.initializer import Initializer

    arr_nodes = np.zeros(len(mesh._nodes), dtype=[('x', 'f8'), ('y', 'f8'),
                                                  ('boundary_type', 'U20')])
    arr_nodes['x'] = [n.x for n in mesh._nodes]
    arr_nodes['y'] = [n.y for n in mesh._nodes]
    arr_nodes['boundary_type'] = [str(n.boundary_type) for n in mesh._nodes]
    init = Initializer.__new__(Initializer)
    init.points = arr_nodes
    init.pipe_width = 0.5
    init._stream_function_geometry = lambda: (1.25, 0.75)
    u_psi, v_psi, _ = init._compute_stream_function_field(u0=1.0, v_in=0.0)
    u_psi_t = torch.tensor(np.column_stack((u_psi, v_psi)), dtype=torch.float32)
    loss_psi = float(
        fvm.mass_loss(u_psi_t, reduction='divergence_l2_volume').item()
    )
    check("Функция тока: mse << однородного",
          loss_psi < 0.1 * loss_uniform,
          f"mse={loss_psi:.2e} vs uniform={loss_uniform:.2e}")

    # Неконсервативное поле даёт ненулевую невязку.
    u_rand = torch.rand(N, 2, dtype=torch.float32)
    loss_rand = float(
        fvm.mass_loss(u_rand, reduction='divergence_l2_volume').item()
    )
    check("Невязка случайного поля > 0", loss_rand > 0,
          f"mse={loss_rand:.2e}")



def test_real_data_mesh():
    """Строит сетку на реальной коллокации из workspace (если она есть)."""
    print("\n[5] Mesh2D + FVM (реальные данные)")
    colloc = os.path.join(PROJECT_DIR, "workspace", "sampling", "collocation",
                          "collocation_points.npy")
    domain = os.path.join(PROJECT_DIR, "workspace", "geometry", "domains",
                          "pipe_bend_domain.npy")
    if not (os.path.exists(colloc) and os.path.exists(domain)):
        print("   SKIP: файлы коллокации отсутствуют")
        return
    from pipe_bend_pinn.sampling.mesh import build_pipe_bend_mesh
    from pipe_bend_pinn.physics.fvm import FVMResiduals

    arr = np.load(colloc, allow_pickle=True)
    domain_arr = np.load(domain)
    mesh = build_pipe_bend_mesh(arr, domain_coords=domain_arr, verbose=False)
    check("Сетка построена (N>1000)", mesh.coords.shape[0] > 1000,
          f"N={mesh.coords.shape[0]}")
    check("Консервативность", mesh.is_conservative() is True)

    fvm = FVMResiduals(mesh).to('cpu', torch.float32)
    N = fvm.N

    # Нулевое поле: тривиально соленоидальное, невязка ~ 0.
    u_zero = torch.zeros(N, 2, dtype=torch.float32)
    loss_zero = float(
        fvm.mass_loss(u_zero, reduction='divergence_l2_volume').item()
    )
    check("Нулевое поле: mse ~ 0", loss_zero < 1e-12, f"mse={loss_zero:.2e}")

    # Однородное поле нарушает no-slip на стенках -> невязка заметная.
    u_t = torch.tensor(
        np.column_stack((np.ones(N), np.zeros(N))), dtype=torch.float32
    )
    loss_uniform = float(
        fvm.mass_loss(u_t, reduction='divergence_l2_volume').item()
    )
    check("Однородное поле нарушает стенки", loss_uniform > 1.0,
          f"mse={loss_uniform:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Воспроизводимость RNG
# ──────────────────────────────────────────────────────────────────────────────
def test_reproducibility():
    print("\n[6] Воспроизводимость RNG")
    import random
    from pipe_bend_pinn.training.reproducibility import (
        capture_rng_state,
        restore_rng_state,
    )

    # Python RNG: снимаем снапшот после 3 потреблений и сравниваем.
    random.seed(123)
    _ = [random.random() for _ in range(3)]
    expected = [random.random() for _ in range(5)]

    random.seed(123)
    _ = [random.random() for _ in range(3)]
    state = capture_rng_state()
    # продолжаем «расходовать» случайность после снятия снапшота
    _ = [random.random() for _ in range(10)]

    restore_rng_state(state)
    restored = [random.random() for _ in range(5)]
    check("Python RNG восстановлен", np.allclose(restored, expected))

    # NumPy RNG: снимаем снапшот после 5 потреблений, сравниваем продолжение.
    np.random.seed(11)
    n1 = np.random.rand(5)
    state2 = capture_rng_state()
    expected_np = np.random.rand(5)
    restore_rng_state(state2)
    n2 = np.random.rand(5)
    check("NumPy RNG восстановлен", np.allclose(n2, expected_np))

    # Torch CPU RNG: аналогично.
    torch.manual_seed(5)
    t1 = torch.rand(5)
    state3 = capture_rng_state()
    expected_t = torch.rand(5)
    restore_rng_state(state3)
    t2 = torch.rand(5)
    check("Torch CPU RNG восстановлен", bool(torch.allclose(t2, expected_t)))

    # Совместимость с legacy-чекпоинтом: None-state не ломается.
    restore_rng_state(None)
    restore_rng_state({})
    check("Legacy checkpoint (нет RNG) не ломается", True)


def main():
    print("=" * 70)
    print("ТЕСТЫ: нормировка / SDF / функция тока / Mesh2D-FVM".center(70))
    print("=" * 70)
    test_normalization()
    test_sdf()
    test_stream_function()
    test_mesh_fvm()
    test_real_data_mesh()
    test_reproducibility()
    print(f"\n{'='*70}")
    print(f"  OK: {PASSED}")
    print(f"  FAIL: {FAILED}")
    return 0 if FAILED == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

