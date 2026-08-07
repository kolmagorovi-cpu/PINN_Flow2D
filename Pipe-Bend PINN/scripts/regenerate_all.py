#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для перегенерации всех данных с исправленными модулями:
1. Геометрия отвода (PB_R100)
2. Точки коллокации (step_size=0.05 → ~5000 точек)
3. Инициализация полей (с исправленным параболическим профилем и нормализацией давления)

Запуск: python regenerate_all.py
"""

import os
import sys
import numpy as np
from collections import Counter

from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT,
    GEOMETRY_DOMAINS_DIR,
    SAMPLING_COLLOCATION_DIR,
    INITIAL_FIELDS_DIR,
)


def main():
    print("=" * 60)
    print("ПЕРЕГЕНЕРАЦИЯ ВСЕХ ДАННЫХ".center(60))
    print("=" * 60)

    # ========== ЭТАП 1: Геометрия ==========
    print("\n[1/3] Геометрия отвода...")
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from pipe_bend_pinn.geometry.core import PipeBendGeometry, GeometryConfig
        geom_config = GeometryConfig(
            save_format='npy',
            plot_show=False,
            verbose=True,
            output_dir=GEOMETRY_DOMAINS_DIR
        )
        geom = PipeBendGeometry(config=geom_config)
        if not geom.load_from_yaml(bend_name='PB_R100'):
            print("❌ Ошибка загрузки геометрии PB_R100")
            return
        geom.create_domain()
        geom.save('pipe_bend')
        print("✅ Геометрия создана")
    except Exception as e:
        print(f"❌ Ошибка геометрии: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== ЭТАП 2: Точки коллокации ==========
    print("\n[2/3] Генерация точек коллокации...")
    try:
        import matplotlib.path as mplPath
        from shapely.geometry import LineString

        domain_path = os.path.join(GEOMETRY_DOMAINS_DIR, "pipe_bend_domain.npy")
        domain = np.load(domain_path)
        if domain.ndim == 1 and domain.dtype.names is not None:
            domain = np.column_stack((domain['x'], domain['y']))
        
        R = 1.0
        W = 0.5
        R_inner = R - W / 2
        x_min, y_min = np.min(domain, axis=0)
        x_max, y_max = np.max(domain, axis=0)
        step_size = 0.05

        path = mplPath.Path(domain)
        x_coords = np.arange(x_min, x_max, step_size)
        y_coords = np.arange(y_min, y_max, step_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.column_stack((X.ravel(), Y.ravel()))
        inside = path.contains_points(points)
        interior_points = points[inside]

        line = LineString(domain)
        total_length = line.length
        num_boundary = 200
        boundary_points = []
        boundary_types = []
        tol = 0.05
        for i in range(num_boundary):
            distance = i * total_length / num_boundary
            pt = line.interpolate(distance)
            xb, yb = pt.x, pt.y
            if abs(xb - x_min) < tol and yb < -R_inner:
                btype = 'inlet'
            elif abs(yb - y_max) < tol and xb > x_min + W:
                btype = 'outlet'
            else:
                btype = 'wall'
            boundary_points.append((xb, yb))
            boundary_types.append(btype)

        all_points_np = np.array(interior_points.tolist() + boundary_points)
        all_types = ['interior'] * len(interior_points) + boundary_types
        
        dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'), ('boundary_type', 'U20'), ('on_boundary', 'bool')]
        structured = np.zeros(len(all_points_np), dtype=dtype)
        structured['id'] = np.arange(len(all_points_np))
        structured['x'] = all_points_np[:, 0]
        structured['y'] = all_points_np[:, 1]
        structured['boundary_type'] = all_types
        structured['on_boundary'] = [t != 'interior' for t in all_types]

        sampler_dir = SAMPLING_COLLOCATION_DIR
        os.makedirs(sampler_dir, exist_ok=True)
        np.save(os.path.join(sampler_dir, 'collocation_points.npy'), structured)
        
        metadata = {
            'num_points': len(structured),
            'num_interior': len(interior_points),
            'num_boundary': len(boundary_points),
            'bounds': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
            'chord': W,
            'pipe_width': W,
        }
        np.save(os.path.join(sampler_dir, 'collocation_points_metadata.npy'), metadata)
        
        type_counts = Counter(all_types)
        for bt, cnt in type_counts.most_common():
            print(f"   {bt}: {cnt}")
        print(f"✅ Точки коллокации созданы (всего: {len(structured)})")
    except Exception as e:
        print(f"❌ Ошибка коллокации: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== ЭТАП 3: Инициализация (с ИСПРАВЛЕННЫМ Initializer.py) ==========
    print("\n[3/3] Инициализация полей...")
    try:
        from pipe_bend_pinn.data.initializer import Initializer
        init = Initializer()
        init.load_sampler_points()
        init.extract_wall_points()

        if init.fluid_lib.list_fluids():
            fluid_name = init.fluid_lib.list_fluids()[0]
        else:
            init.fluid_lib.create_default_library()
            fluid_name = init.fluid_lib.list_fluids()[0]
        
        if init.cond_manager.list_names():
            cond_name = init.cond_manager.list_names()[0]
        else:
            init.cond_manager.create_defaults()
            cond_name = init.cond_manager.list_names()[0]

        init.fluid = init.fluid_lib.get(fluid_name)
        init.conditions = init.cond_manager.get(cond_name)
        init.fluid_name = fluid_name
        init.conditions_name = cond_name

        init.initialize_fields(pressure_mode='linear_x', velocity_initial_mode='parabolic')
        init.save(description='initial')
        print(f"   U_inf = {init.U_inf:.3f} м/с, Re = {init.Re:.2e}")
        print(f"   pipe_width = {init.pipe_width:.4f} м")
        print(f"   Нормализация давления: Cp = (p - p_ref) / (0.5 * ρ * U_inf²)")
        print("✅ Инициализация завершена")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== ФИНИШ ==========
    print("\n" + "=" * 60)
    print("✅ ПЕРЕГЕНЕРАЦИЯ ЗАВЕРШЕНА".center(60))
    print("=" * 60)
    print()
    print("Файлы:")
    print(f"   Geometry/Data/pipe_bend_domain.npy")
    print(f"   Geometry/Data/pipe_bend_metadata.npy")
    print(f"   Sampler/Data/collocation_points.npy")
    print(f"   Initial/Data/initial.npy")
    print()
    print("Конфиг: config.yaml (скопирован из config.yaml.example)")
    print()
    print("Исправления в коде:")
    print("   1. Initializer.py: параболический профиль БОЛЬШЕ НЕ ЗАТИРАЕТСЯ константой")
    print("   2. Initializer.py: p_norm = (p-p_ref) / (0.5*ρ*U_inf²) — согласовано с Airfoil")
    print("   3. solver.py: обратное преобразование давления исправлено (добавлен 0.5)")
    print("   4. solver.py: размер маркеров адаптивный (зависит от числа точек)")
    print("   5. config.yaml: сбалансированные веса loss (PDE:BC ≈ 1:5, было 1:5000)")
    print()


if __name__ == '__main__':
    main()