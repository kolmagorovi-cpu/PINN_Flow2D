#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрая подготовка данных и запуск из config.yaml.

Выполняет полный пайплайн: Геометрия → Коллокация → Инициализация → Обучение.
"""

from __future__ import annotations

import os

from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT,
    GEOMETRY_DOMAINS_DIR,
    GEOMS_LIB_PATH,
    INITIAL_FIELDS_DIR,
    EXPERIMENTS_DIR,
    CONFIGS_DIR,
)
from pipe_bend_pinn.utils.ui_utils import Colors


def quick_setup() -> None:
    """Быстрая подготовка всех данных для обучения."""
    print(f"\n{Colors.YELLOW}🚀 БЫСТРАЯ ПОДГОТОВКА ДАННЫХ{Colors.RESET}")

    # --- Этап 1: Геометрия ---
    print(f"{Colors.CYAN}Этап 1/3: Геометрия отвода{Colors.RESET}")
    try:
        from pipe_bend_pinn.geometry.core import PipeBendGeometry, GeometryConfig

        geom_config = GeometryConfig(
            save_format="npy", plot_show=False, verbose=True,
            output_dir=GEOMETRY_DOMAINS_DIR,
        )
        geom = PipeBendGeometry(config=geom_config)
        if not geom.load_from_yaml(yaml_path=GEOMS_LIB_PATH, bend_name="PB_R100"):
            raise RuntimeError("Ошибка загрузки геометрии PB_R100")
        geom.create_domain()
        geom.save("pipe_bend")
        print(f"{Colors.GREEN}✅ Геометрия создана{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка геометрии: {e}{Colors.RESET}")
        return

    # --- Этап 2: Коллокация ---
    print(f"\n{Colors.CYAN}Этап 2/3: Генерация точек коллокации{Colors.RESET}")
    try:
        from pipe_bend_pinn.sampling.core import generate_collocation

        stat = generate_collocation(step_size=0.05, num_boundary=200)
        for bt, cnt in stat["counts"].items():
            print(f"      {bt}: {cnt}")
        print(f"{Colors.GREEN}✅ Точки коллокации созданы (всего: {stat['num_points']}){Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка коллокации: {e}{Colors.RESET}")
        return

    # --- Этап 3: Инициализация ---
    print(f"\n{Colors.CYAN}Этап 3/3: Инициализация данных{Colors.RESET}")
    try:
        from pipe_bend_pinn.data.initializer import Initializer

        init = Initializer()
        init.load_sampler_points()
        init.extract_wall_points()

        fluid_name = init.fluid_lib.list_fluids()[0] if init.fluid_lib.list_fluids() else None
        cond_name = init.cond_manager.list_names()[0] if init.cond_manager.list_names() else None

        if fluid_name is None or cond_name is None:
            print(f"{Colors.YELLOW}⚠️ Создаю жидкости и условия по умолчанию...{Colors.RESET}")
            init.fluid_lib.create_default_library()
            init.cond_manager.create_defaults()
            fluid_name = init.fluid_lib.list_fluids()[0]
            cond_name = init.cond_manager.list_names()[0]

        init.fluid = init.fluid_lib.get(fluid_name)
        init.conditions = init.cond_manager.get(cond_name)
        init.fluid_name = fluid_name
        init.conditions_name = cond_name

        init.initialize_fields(pressure_mode="linear_x", velocity_initial_mode="parabolic")
        init.save(description="initial")
        print(f"      U_inf = {init.U_inf:.3f} м/с, Re = {init.Re:.2e}")
        print(f"{Colors.GREEN}✅ Инициализация завершена{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка инициализации: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{Colors.GREEN}✅✅✅ Быстрая подготовка завершена!{Colors.RESET}")
    print(f"{Colors.CYAN}Файл данных: workspace/initial/fields/initial.npy{Colors.RESET}")


def run_from_config(config_path: str) -> None:
    """Запускает полный пайплайн из config.yaml без интерактивного меню."""
    print(f"\n{Colors.YELLOW}🚀 ЗАПУСК ИЗ КОНФИГА: {config_path}{Colors.RESET}")
    from pipe_bend_pinn.config.manager import ConfigManager

    cfg = ConfigManager(config_path=config_path)
    print(f"   Проект: {cfg.config.get('project_name', 'unnamed')}")
    print(f"   Файл точек: {cfg.config['data'].get('points_file', 'не указан')}")

    # Шаг 1: Быстрая подготовка (если данных нет)
    points_file = cfg.config["data"].get("points_file")
    if not points_file or not os.path.exists(str(points_file)):
        print(f"{Colors.YELLOW}⚠️ Данные не найдены. Запускаю быструю подготовку...{Colors.RESET}")
        quick_setup()
        cfg = ConfigManager(config_path=config_path)

    # Шаг 2: Нормализация весов BC для pipe bend
    cfg.config["flow_type"] = "planar"
    bc_weights = cfg.config["loss_weights"]["bc_weights"]
    for key in ["symmetry", "axis_symmetry", "symmetry_grad", "axis_symmetry_grad"]:
        bc_weights.pop(key, None)
    if "wall" not in bc_weights:
        bc_weights["wall"] = 5000.0

    # Шаг 3: Запуск обучения
    print(f"{Colors.CYAN}▶ Запуск обучения...{Colors.RESET}")
    from pipe_bend_pinn.training.solver import PINNSolver

    try:
        solver = PINNSolver(cfg.config)
        solver.train()
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return

    print(f"{Colors.GREEN}✅ Обучение завершено{Colors.RESET}")
