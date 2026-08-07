#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматическая реструктуризация Pipe-Bend PINN в пакет ``pipe_bend_pinn/``.

Запуск:  python build_package.py

Скрипт:
1. Переносит доменные модули в подпакеты пакета pipe_bend_pinn/.
2. Переписывает импорты на пакетные.
3. Централизует пути через pipe_bend_pinn.app.paths.
4. Создаёт новые CLI/приложение с реестром модулей.
5. Создаёт совместимые обёртки верхнего уровня (старые имена продолжают работать).
6. Мигрирует данные в workspace/.
7. Обновляет config.yaml, .gitignore, тесты и скрипты запуска.
"""

import ast
import os
import re
import shutil
import sys
from pathlib import Path

# Скрипт находится в scripts/, поэтому корень проекта — двумя уровнями выше
BASE = Path(__file__).resolve().parents[1]
PKG = BASE / "pipe_bend_pinn"
WORKSPACE = BASE / "workspace"


# =============================================================================
# Вспомогательные функции
# =============================================================================

def ensure_dirs() -> None:
    """Создаёт дерево директорий пакета."""
    for sub in (
        "app", "config", "core", "physics", "geometry",
        "data", "sampling", "training/components", "utils",
    ):
        (PKG / sub).mkdir(parents=True, exist_ok=True)


def write_if_missing(target: Path, content: str) -> bool:
    """Пишет файл, если его ещё нет. Возвращает True, если записал."""
    if target.exists():
        print(f"  - {target.relative_to(BASE)} уже существует, пропуск")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"  + {target.relative_to(BASE)}")
    return True


# =============================================================================
# Правила замен
# =============================================================================

IMPORT_RULES = [
    ("from solver_components.", "from pipe_bend_pinn.training.components."),
    ("import solver_components.", "import pipe_bend_pinn.training.components."),
    ("from data_handler import", "from pipe_bend_pinn.data.handler import"),
    ("from checkpoint import", "from pipe_bend_pinn.training.checkpoint import"),
    ("from models import", "from pipe_bend_pinn.training.models import"),
    ("from Fluids import", "from pipe_bend_pinn.physics.fluids import"),
    ("from Conditions import", "from pipe_bend_pinn.physics.conditions import"),
    ("from Geometry import", "from pipe_bend_pinn.geometry.core import"),
    ("from Initializer import", "from pipe_bend_pinn.data.initializer import"),
    ("from config_manager import", "from pipe_bend_pinn.config.manager import"),
    ("from solver import", "from pipe_bend_pinn.training.solver import"),
    ("from pde_residuals import", "from pipe_bend_pinn.physics.pde import"),
    ("from utils import", "from pipe_bend_pinn.utils.helpers import"),
    ("from key_listener import", "from pipe_bend_pinn.training.key_listener import"),
    ("from visualizer import", "from pipe_bend_pinn.training.visualizer import"),
    ("from reference import", "from pipe_bend_pinn.training.reference import"),
    ("from constants import", "from pipe_bend_pinn.core.constants import"),
    ("from type_hints import", "from pipe_bend_pinn.core.type_hints import"),
    ("import type_hints", "import pipe_bend_pinn.core.type_hints"),
]

PATH_RULES = [
    ('os.path.join(PROJECT_ROOT, "Geometry", "Data")', "GEOMETRY_DOMAINS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Geometry', 'Data')", "GEOMETRY_DOMAINS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Sampler", "Data")', "SAMPLING_COLLOCATION_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Sampler', 'Data')", "SAMPLING_COLLOCATION_DIR"),
    ('os.path.join(PROJECT_ROOT, "Initial", "Data")', "INITIAL_FIELDS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Initial', 'Data')", "INITIAL_FIELDS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Initial", "Plots")', "INITIAL_PLOTS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Initial', 'Plots')", "INITIAL_PLOTS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Fluids")', "FLUIDS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Fluids')", "FLUIDS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Conditions")', "CONDITIONS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Conditions')", "CONDITIONS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Checkpoints")', "EXPERIMENTS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Checkpoints')", "EXPERIMENTS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Configs")', "CONFIGS_DIR"),
    ("os.path.join(PROJECT_ROOT, 'Configs')", "CONFIGS_DIR"),
    ('os.path.join(PROJECT_ROOT, "Geometry", "pipe_bend_geoms.yaml")', "GEOMS_LIB_PATH"),
    ("os.path.join(PROJECT_ROOT, 'Geometry', 'pipe_bend_geoms.yaml')", "GEOMS_LIB_PATH"),
    ('os.path.join(PROJECT_ROOT, "geom_config.yaml")', "GEOM_CONFIG_PATH"),
    ("os.path.join(PROJECT_ROOT, 'geom_config.yaml')", "GEOM_CONFIG_PATH"),
    ('config.get(\'checkpoint_dir\', \'Checkpoints\')', "config.get('checkpoint_dir', EXPERIMENTS_DIR)"),
    ('config.get("checkpoint_dir", "Checkpoints")', "config.get('checkpoint_dir', EXPERIMENTS_DIR)"),
    ('directory: str = "Conditions"', "directory: str = CONDITIONS_DIR"),
    ("directory: str = 'Conditions'", "directory: str = CONDITIONS_DIR"),
    ('directory: str = "Fluids"', "directory: str = FLUIDS_DIR"),
    ("directory: str = 'Fluids'", "directory: str = FLUIDS_DIR"),
]

CONFIG_PREPEND = (
    "from pipe_bend_pinn.app.paths import (\n"
    "    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,\n"
    "    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,\n"
    "    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,\n"
    "    CONFIGS_DIR, GEOM_CONFIG_PATH,\n"
    ")\n\n"
)


def apply_rules(text: str) -> str:
    """Применяет правила импортов и путей к тексту модуля."""
    for old, new in IMPORT_RULES:
        text = text.replace(old, new)
    for old, new in PATH_RULES:
        text = text.replace(old, new)
    # Убираем старое определение корня проекта
    text = re.sub(
        r"^\s*PROJECT_ROOT\s*=\s*os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\s*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


# =============================================================================
# Автоматический перенос доменных модулей
# =============================================================================

AUTO_MIGRATE = {
    "Fluids.py": "physics/fluids.py",
    "Conditions.py": "physics/conditions.py",
    "pde_residuals.py": "physics/pde.py",
    "Geometry.py": "geometry/core.py",
    "Initializer.py": "data/initializer.py",
    "data_handler.py": "data/handler.py",
    "models.py": "training/models.py",
    "checkpoint.py": "training/checkpoint.py",
    "key_listener.py": "training/key_listener.py",
    "reference.py": "training/reference.py",
    "visualizer.py": "training/visualizer.py",
    "solver.py": "training/solver.py",
    "PINN_Learning.py": "training/learning_cli.py",
    "solver_components/loss_computer.py": "training/components/loss_computer.py",
    "solver_components/trainer.py": "training/components/trainer.py",
    "solver_components/metrics.py": "training/components/metrics.py",
    "solver_components/optimizers.py": "training/components/optimizers.py",
    "solver_components/interactive.py": "training/components/interactive.py",
}


def migrate_auto_files() -> None:
    """Переносит доменные модули в пакет с заменой импортов."""
    print("\n[1/8] Перенос доменных модулей...")
    for src_name, dst_rel in AUTO_MIGRATE.items():
        src = BASE / src_name
        dst = PKG / dst_rel
        if not src.exists():
            print(f"  ! Исходник {src_name} не найден")
            continue
        if dst.exists():
            print(f"  - {dst_rel} уже существует, пропуск")
            continue
        text = src.read_text(encoding="utf-8")
        text = apply_rules(text)
        text = CONFIG_PREPEND + text
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        print(f"  + {dst_rel}")


# =============================================================================
# Новые модули (без исходников)
# =============================================================================

SAMPLING_CORE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дискретизация Pipe-Bend PINN: генерация точек коллокации.

Классифицирует точки по типам границ (inlet, outlet, wall) и сохраняет
структурированный массив точек в workspace/sampling/collocation/.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import numpy as np
import matplotlib.path as mplPath
from shapely.geometry import LineString

from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT,
    GEOMETRY_DOMAINS_DIR,
    SAMPLING_COLLOCATION_DIR,
)


def load_domain(filename: str = "pipe_bend_domain.npy") -> np.ndarray:
    """Загружает контур расчётной области из workspace/geometry/domains/."""
    domain_path = os.path.join(GEOMETRY_DOMAINS_DIR, filename)
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Файл геометрии не найден: {domain_path}")
    domain = np.load(domain_path)
    if domain.ndim == 1 and domain.dtype.names is not None:
        domain = np.column_stack((domain["x"], domain["y"]))
    return domain


def generate_collocation(
    step_size: float = 0.05,
    num_boundary: int = 200,
    bend_radius: float = 1.0,
    pipe_width: float = 0.5,
    domain_filename: str = "pipe_bend_domain.npy",
    output_name: str = "collocation_points",
) -> dict:
    """
    Генерирует точки коллокации (interior + boundary) для области отвода.

    Returns:
        Словарь со статистикой: num_points, num_interior, num_boundary, counts.
    """
    domain = load_domain(domain_filename)
    R_inner = bend_radius - pipe_width / 2

    x_min, y_min = np.min(domain, axis=0)
    x_max, y_max = np.max(domain, axis=0)

    # Внутренние точки (регулярная сетка внутри контура)
    path = mplPath.Path(domain)
    x_coords = np.arange(x_min, x_max, step_size)
    y_coords = np.arange(y_min, y_max, step_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.ravel(), Y.ravel()))
    inside = path.contains_points(points)
    interior_points = points[inside]

    # Граничные точки равномерно по периметру
    line = LineString(domain)
    total_length = line.length
    boundary_points = []
    boundary_types = []
    tol = step_size

    for i in range(num_boundary):
        distance = i * total_length / num_boundary
        pt = line.interpolate(distance)
        xb, yb = pt.x, pt.y
        if abs(xb - x_min) < tol and yb < -R_inner:
            btype = "inlet"
        elif abs(yb - y_max) < tol and xb > x_min + pipe_width:
            btype = "outlet"
        else:
            btype = "wall"
        boundary_points.append((xb, yb))
        boundary_types.append(btype)

    all_points_np = np.array(interior_points.tolist() + boundary_points)
    all_types = ["interior"] * len(interior_points) + boundary_types

    dtype = [("id", "i4"), ("x", "f8"), ("y", "f8"), ("boundary_type", "U20"), ("on_boundary", "bool")]
    structured = np.zeros(len(all_points_np), dtype=dtype)
    structured["id"] = np.arange(len(all_points_np))
    structured["x"] = all_points_np[:, 0]
    structured["y"] = all_points_np[:, 1]
    structured["boundary_type"] = all_types
    structured["on_boundary"] = [t != "interior" for t in all_types]

    os.makedirs(SAMPLING_COLLOCATION_DIR, exist_ok=True)
    np.save(os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}.npy"), structured)

    metadata = {
        "num_points": len(structured),
        "num_interior": len(interior_points),
        "num_boundary": len(boundary_points),
        "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "chord": pipe_width,
        "pipe_width": pipe_width,
    }
    np.save(os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}_metadata.npy"), metadata)

    counts = dict(Counter(all_types))
    return {
        "num_points": len(structured),
        "num_interior": len(interior_points),
        "num_boundary": len(boundary_points),
        "counts": counts,
        "path": os.path.join(SAMPLING_COLLOCATION_DIR, f"{output_name}.npy"),
    }


def S2DR() -> None:
    """Интерактивный запуск генерации точек коллокации."""
    print("\\n" + "=" * 60)
    print("ДИСКРЕТИЗАЦИЯ (КОЛЛОКАЦИЯ)".center(60))
    print("=" * 60)
    try:
        step = float(input("Шаг сетки [0.05]: ").strip() or 0.05)
        num_bnd = int(input("Число граничных точек [200]: ").strip() or 200)
        stat = generate_collocation(step_size=step, num_boundary=num_bnd)
        print(f"\\n✅ Точки коллокации созданы:")
        print(f"   Всего: {stat['num_points']}")
        print(f"   Interior: {stat['num_interior']}")
        print(f"   Boundary: {stat['num_boundary']}")
        for bt, cnt in stat["counts"].items():
            print(f"   {bt}: {cnt}")
        print(f"   Файл: {stat['path']}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    input("\\nНажмите Enter для возврата...")
'''

SAMPLING_CLI = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-точка входа для дискретизации."""
from pipe_bend_pinn.sampling.core import S2DR

__all__ = ["S2DR"]
'''

GEOMETRY_CLI = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-точка входа для геометрии отвода."""
from pipe_bend_pinn.geometry.core import g2d

__all__ = ["g2d"]
'''

DATA_INITIALIZER_CLI = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-точка входа для инициализации данных."""
from pipe_bend_pinn.data.initializer import initialize_menu

__all__ = ["initialize_menu"]
'''

QUICK_SETUP = '''#!/usr/bin/env python3
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
    print(f"\\n{Colors.YELLOW}🚀 БЫСТРАЯ ПОДГОТОВКА ДАННЫХ{Colors.RESET}")

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
    print(f"\\n{Colors.CYAN}Этап 2/3: Генерация точек коллокации{Colors.RESET}")
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
    print(f"\\n{Colors.CYAN}Этап 3/3: Инициализация данных{Colors.RESET}")
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

    print(f"\\n{Colors.GREEN}✅✅✅ Быстрая подготовка завершена!{Colors.RESET}")
    print(f"{Colors.CYAN}Файл данных: workspace/initial/fields/initial.npy{Colors.RESET}")


def run_from_config(config_path: str) -> None:
    """Запускает полный пайплайн из config.yaml без интерактивного меню."""
    print(f"\\n{Colors.YELLOW}🚀 ЗАПУСК ИЗ КОНФИГА: {config_path}{Colors.RESET}")
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
'''

APP_CLI = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консольный интерфейс Pipe-Bend PINN.

Модуль отвечает только за навигацию пользователя: главное меню, меню модулей,
быстрая подготовка, прямой запуск команд. Расчетные блоки подключаются через
реестр ``module_registry`` и остаются отделёнными от UI-логики.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import os
import sys
from collections.abc import Sequence
from typing import Any

from pipe_bend_pinn.app.module_registry import (
    ModuleDescriptor,
    default_modules,
    find_by_command,
    modules_as_legacy_dicts,
)
from pipe_bend_pinn.utils.ui_utils import (
    Colors,
    clear_screen,
    get_menu_choice,
    pause_and_continue,
)


def print_banner() -> None:
    """Выводит баннер программы с текущим годом."""
    current_year = datetime.datetime.now().year
    banner = f"""
    {Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
    ║                 PIPEBENDPINN v1.1 ({current_year})               ║
    ║         Physics-Informed Neural Networks for Pipe Bends         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  • Моделирование течения в отводах 90°                      ║
    ║  • Генерация геометрии отводов                               ║
    ║  • Параболический профиль скорости (Пуазейль)                ║
    ║  • Решение уравнений Навье-Стокса PINNs                      ║
    ╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)


def print_menu_header(title: str) -> None:
    """Печатает единый заголовок для всех консольных меню."""
    print(f"\\n{Colors.BOLD}{Colors.MAGENTA}═══ {title} ═══{Colors.RESET}")


def print_module_info(modules: list[dict[str, Any]]) -> None:
    """Выводит список доступных расчетных модулей."""
    print_menu_header("ДОСТУПНЫЕ МОДУЛИ")

    for index, info in enumerate(modules, 1):
        if info["status"] == "готов":
            status_icon = f"{Colors.GREEN}✅{Colors.RESET}"
            status_color = Colors.GREEN
        elif info["status"] == "разработка":
            status_icon = f"{Colors.YELLOW}🔄{Colors.RESET}"
            status_color = Colors.YELLOW
        else:
            status_icon = f"{Colors.BLUE}📅{Colors.RESET}"
            status_color = Colors.BLUE

        print(
            f"  {Colors.BOLD}{Colors.CYAN}[{index}]{Colors.RESET} "
            f"{Colors.BOLD}{info['name']}{Colors.RESET}"
        )
        print(
            f"      {status_color}Код: {info['code']} | "
            f"Команда: {info['command']} | "
            f"Статус: {info['status']} {status_icon}{Colors.RESET}"
        )
        if "description" in info:
            print(f"      {Colors.CYAN}↳ {info['description']}{Colors.RESET}")
        if index < len(modules):
            print()

    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")


def print_main_menu() -> None:
    """Выводит главное меню программы."""
    print_menu_header("ГЛАВНОЕ МЕНЮ")

    menu_items = [
        {"num": 1, "desc": "Работа с модулями"},
        {"num": 4, "desc": "🚀 Быстрая подготовка (Геом+Коллок+Иниц)"},
        {"num": 5, "desc": "⚙ Запуск обучения из config.yaml"},
        {"num": 3, "desc": "Справка"},
        {"num": 2, "desc": "Очистить экран"},
        {"num": 0, "desc": "Выход"},
    ]

    for item in menu_items:
        if item["num"] == 0:
            print(
                f"  {Colors.BOLD}{Colors.RED}[{item['num']}]{Colors.RESET} "
                f"{Colors.RED}{item['desc']}{Colors.RESET}"
            )
        elif item["num"] in (4, 5):
            print(
                f"  {Colors.BOLD}{Colors.YELLOW}[{item['num']}]{Colors.RESET} "
                f"{Colors.YELLOW}{item['desc']}{Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.BOLD}{Colors.GREEN}[{item['num']}]{Colors.RESET} "
                f"{item['desc']}"
            )

    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")


def print_help() -> None:
    """Выводит справочную информацию по консольной программе."""
    print_menu_header("СПРАВКА")

    help_text = f"""
    {Colors.BOLD}Как пользоваться программой:{Colors.RESET}

    1. В главном меню выберите {Colors.GREEN}[1]{Colors.RESET} для работы с модулями
    2. В меню модулей выберите номер нужного модуля
    3. Следуйте инструкциям выбранного модуля

    {Colors.BOLD}Прямой запуск без меню:{Colors.RESET}
    • python -m pipe_bend_pinn geometry
    • python -m pipe_bend_pinn sampling
    • python -m pipe_bend_pinn fluids
    • python -m pipe_bend_pinn conditions
    • python -m pipe_bend_pinn init
    • python -m pipe_bend_pinn train --config workspace/configs/config.yaml

    {Colors.BOLD}Горячие клавиши:{Colors.RESET}
    • {Colors.CYAN}0{Colors.RESET} - вернуться в главное меню / выход
    • {Colors.CYAN}Enter{Colors.RESET} - подтвердить ввод
    • {Colors.CYAN}Ctrl+C{Colors.RESET} - прервать выполнение
    """
    print(help_text)
    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")


def load_module(module_info: dict[str, Any]):
    """Загружает функцию расчетного модуля по записи из реестра."""
    try:
        module = importlib.import_module(module_info["module"])
        return getattr(module, module_info["func"])
    except ImportError as exc:
        print(f"{Colors.RED}❌ Ошибка загрузки модуля '{module_info['module']}':{Colors.RESET}")
        print(f"   {exc}")
        return None
    except AttributeError:
        print(
            f"{Colors.RED}❌ Ошибка: функция '{module_info['func']}' "
            f"не найдена в модуле{Colors.RESET}"
        )
        return None
    except Exception as exc:
        print(f"{Colors.RED}❌ Непредвиденная ошибка: {exc}{Colors.RESET}")
        return None


def run_module(module_info: dict[str, Any]) -> None:
    """Выполняет один расчетный модуль из меню или прямой команды."""
    print(f"\\n{Colors.YELLOW}┌─ Запуск модуля: {module_info['name']}{Colors.RESET}")
    print(f"{Colors.YELLOW}│{Colors.RESET} Код: {module_info['code']}")
    print(f"{Colors.YELLOW}│{Colors.RESET} Статус: {module_info['status']}")
    print(f"{Colors.YELLOW}│{Colors.RESET} Описание: {module_info.get('description', 'Нет описания')}")
    print(f"{Colors.YELLOW}└─{Colors.RESET}")

    if module_info["status"] != "готов":
        print(f"{Colors.YELLOW}⚠️  Модуль в разработке.{Colors.RESET}")
        return

    func = load_module(module_info)
    if func is None:
        print(f"{Colors.RED}❌ Не удалось загрузить модуль{Colors.RESET}")
        return

    try:
        print(f"{Colors.GREEN}▶ Запуск...{Colors.RESET}\\n")
        func()
    except Exception as exc:
        print(f"{Colors.RED}❌ Ошибка при выполнении модуля:{Colors.RESET}")
        print(f"   {exc}")


def run_modules_menu(modules: list[dict[str, Any]]) -> None:
    """Показывает меню выбора и запуска расчетных модулей."""
    while True:
        clear_screen()
        print_banner()
        print_module_info(modules)

        try:
            choice = get_menu_choice(
                "Выберите модуль",
                valid_range=(1, len(modules)),
                allow_zero=True,
                zero_action="назад",
            )

            if choice == "0":
                break

            index = int(choice)
            if 1 <= index <= len(modules):
                run_module(modules[index - 1])
                pause_and_continue("Нажмите Enter чтобы продолжить...")
            else:
                print(f"{Colors.RED}❌ Неверный номер.{Colors.RESET}")
                pause_and_continue("Нажмите Enter...")

        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            pause_and_continue("Нажмите Enter...")
        except KeyboardInterrupt:
            print(f"\\n{Colors.YELLOW}⚠️  Возврат в главное меню{Colors.RESET}")
            break


def run_interactive_menu() -> None:
    """Запускает основное интерактивное консольное меню."""
    modules = modules_as_legacy_dicts()

    clear_screen()
    print_banner()

    while True:
        print_main_menu()

        try:
            choice = get_menu_choice(
                "Выберите действие",
                valid_range=(1, 5),
                allow_zero=True,
                zero_action="выход",
            )

            if choice == "0":
                print(f"\\n{Colors.GREEN}👋 Программа завершена. До свидания!{Colors.RESET}")
                break
            if choice == "1":
                run_modules_menu(modules)
                clear_screen()
                print_banner()
            elif choice == "2":
                clear_screen()
                print_banner()
            elif choice == "3":
                clear_screen()
                print_banner()
                print_help()
                pause_and_continue("Нажмите Enter чтобы продолжить...")
                clear_screen()
                print_banner()
            elif choice == "4":
                clear_screen()
                print_banner()
                from pipe_bend_pinn.app.quick_setup import quick_setup
                quick_setup()
                clear_screen()
                print_banner()
            elif choice == "5":
                clear_screen()
                print_banner()
                from pipe_bend_pinn.app.paths import CONFIGS_DIR
                default_cfg = os.path.join(CONFIGS_DIR, "config.yaml")
                if os.path.exists(default_cfg):
                    print(f"\\n{Colors.CYAN}Использую: {default_cfg}{Colors.RESET}")
                    from pipe_bend_pinn.app.quick_setup import run_from_config
                    run_from_config(default_cfg)
                else:
                    print(f"{Colors.RED}❌ config.yaml не найден в workspace/configs/.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Создайте config.yaml или используйте --config file.yaml{Colors.RESET}")
                    input("\\nНажмите Enter...")
                clear_screen()
                print_banner()
            else:
                print(f"{Colors.RED}❌ Неверный номер.{Colors.RESET}")
                pause_and_continue("Нажмите Enter...")
                clear_screen()
                print_banner()

        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            pause_and_continue("Нажмите Enter...")
            clear_screen()
            print_banner()
        except KeyboardInterrupt:
            print(f"\\n\\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
            print(f"{Colors.GREEN}👋 До свидания!{Colors.RESET}")
            break


def _build_parser() -> argparse.ArgumentParser:
    """Создает парсер прямых консольных команд."""
    parser = argparse.ArgumentParser(
        prog="pipe_bend_pinn",
        description="Консольный запуск модулей Pipe-Bend PINN",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=[module.command for module in default_modules()],
        help="Модуль для прямого запуска без главного меню",
    )
    parser.add_argument(
        "--config",
        help="YAML-конфигурация для прямого запуска обучения",
    )
    return parser


def run_command(args: argparse.Namespace) -> None:
    """Выполняет прямую команду пакетного CLI."""
    if args.command == "train" and args.config:
        from pipe_bend_pinn.app.quick_setup import run_from_config
        run_from_config(args.config)
        return

    module = find_by_command(args.command)
    if module is None:
        raise SystemExit(f"Неизвестная команда: {args.command}")

    run_module(module.as_legacy_dict())


def main(argv: Sequence[str] | None = None) -> None:
    """Главная функция консольного приложения."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command:
        run_command(args)
    else:
        run_interactive_menu()


def guarded_main(argv: Sequence[str] | None = None) -> int:
    """Запускает CLI с верхнеуровневой обработкой ошибок."""
    try:
        main(argv)
        return 0
    except KeyboardInterrupt:
        print(f"\\n\\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
        print(f"{Colors.GREEN}👋 До свидания!{Colors.RESET}")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\\n{Colors.RED}❌ Критическая ошибка: {exc}{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(guarded_main())
'''

SPECIAL_FILES = {
    "app/cli.py": APP_CLI,
    "app/quick_setup.py": QUICK_SETUP,
    "sampling/core.py": SAMPLING_CORE,
    "sampling/cli.py": SAMPLING_CLI,
    "geometry/cli.py": GEOMETRY_CLI,
    "data/initializer_cli.py": DATA_INITIALIZER_CLI,
    "physics/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Физика Pipe-Bend PINN: жидкости, граничные условия, PDE."""\n',
    "geometry/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Геометрия отвода Pipe-Bend PINN."""\n',
    "data/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Данные и инициализация Pipe-Bend PINN."""\n',
    "sampling/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Дискретизация Pipe-Bend PINN."""\n',
    "training/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Обучение PINN для Pipe-Bend."""\n',
    "training/components/__init__.py": '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Компоненты обучения PINN."""\n',
}


def write_special_files() -> None:
    """Создаёт новые модули без исходников."""
    print("\\n[2/8] Создание новых модулей пакета...")
    for rel, content in SPECIAL_FILES.items():
        write_if_missing(PKG / rel, content)


# =============================================================================
# Совместимые обёртки верхнего уровня
# =============================================================================

LEGACY_MODULES = [
    "Fluids.py", "Conditions.py", "Geometry.py", "Initializer.py",
    "data_handler.py", "models.py", "checkpoint.py", "key_listener.py",
    "reference.py", "visualizer.py", "pde_residuals.py", "solver.py",
    "config_manager.py", "constants.py", "type_hints.py", "utils.py",
    "PINN_Learning.py",
]

LEGACY_COMPONENTS = [
    "solver_components/__init__.py",
    "solver_components/loss_computer.py",
    "solver_components/trainer.py",
    "solver_components/metrics.py",
    "solver_components/optimizers.py",
    "solver_components/interactive.py",
]

PKG_TARGET = {
    "Fluids.py": "pipe_bend_pinn.physics.fluids",
    "Conditions.py": "pipe_bend_pinn.physics.conditions",
    "Geometry.py": "pipe_bend_pinn.geometry.core",
    "Initializer.py": "pipe_bend_pinn.data.initializer",
    "data_handler.py": "pipe_bend_pinn.data.handler",
    "models.py": "pipe_bend_pinn.training.models",
    "checkpoint.py": "pipe_bend_pinn.training.checkpoint",
    "key_listener.py": "pipe_bend_pinn.training.key_listener",
    "reference.py": "pipe_bend_pinn.training.reference",
    "visualizer.py": "pipe_bend_pinn.training.visualizer",
    "pde_residuals.py": "pipe_bend_pinn.physics.pde",
    "solver.py": "pipe_bend_pinn.training.solver",
    "config_manager.py": "pipe_bend_pinn.config.manager",
    "constants.py": "pipe_bend_pinn.core.constants",
    "type_hints.py": "pipe_bend_pinn.core.type_hints",
    "utils.py": "pipe_bend_pinn.utils.helpers",
    "PINN_Learning.py": "pipe_bend_pinn.training.learning_cli",
}


def extract_names(source: Path) -> list[str]:
    """Извлекает имена верхнего уровня (классы, функции, переменные)."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("__"):
                    names.append(t.id)
    # Отбираем только доверенные имена: не дубли и не имена, начинающиеся с "_"
    result: list[str] = []
    for n in names:
        if n and n[0] != "_" and n not in result:
            result.append(n)
    return result


def make_shim(module_path: str, source: Path) -> str:
    """Генерирует совместимую обёртку старого модуля."""
    names = extract_names(source)
    if not names:
        names = ["*"]
    body = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        f'"""Совместимая обёртка. Логика перенесена в {module_path}."""\n\n'
        f"from {module_path} import {', '.join(names)}\n"
    )
    return body


def write_legacy_shims() -> None:
    """Создаёт обёртки старых модулей верхнего уровня (перезаписывает)."""
    print("\n[3/8] Создание совместимых обёрток верхнего уровня...")
    for src_name, target in PKG_TARGET.items():
        src = BASE / src_name
        if not src.exists():
            print(f"  ! Не найден исходник {src_name}")
            continue
        shim = make_shim(target, src)
        (BASE / src_name).write_text(shim, encoding="utf-8")
        print(f"  ~ {src_name} → обёртка на {target}")

    for rel in LEGACY_COMPONENTS:
        src_name = rel
        src = BASE / src_name
        if not src.exists():
            continue
        # target для компонента
        if rel == "solver_components/__init__.py":
            target = "pipe_bend_pinn.training.components"
        else:
            fname = Path(rel).name
            target = f"pipe_bend_pinn.training.components.{fname[:-3]}"
        shim = make_shim(target, src)
        (BASE / rel).write_text(shim, encoding="utf-8")
        print(f"  ~ {rel} → обёртка на {target}")


# =============================================================================
# Патчинг корневых скриптов
# =============================================================================

PATCH_ROOT_SCRIPTS = [
    "plot_trained_model.py",
    "regenerate_all.py",
    "test_config.py",
    "test_imports.py",
    "test_pipeline.py",
]


def patch_root_scripts() -> None:
    """Обновляет импорты в корневых скриптах."""
    print("\\n[4/8] Обновление корневых скриптов...")
    for name in PATCH_ROOT_SCRIPTS:
        path = BASE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new_text = apply_rules(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"  ~ {name}")


# =============================================================================
# Точка входа верхнего уровня
# =============================================================================

PIPE_BEND_MAIN = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимая консольная точка входа Pipe-Bend PINN.

Основная логика меню перенесена в пакет ``pipe_bend_pinn.app``. Этот файл
оставлен, чтобы привычная команда ``python pipe_bend_main.py`` продолжала
работать во время пошагового рефакторинга проекта.
"""

import sys

from pipe_bend_pinn.app.runtime import bootstrap_environment


def main() -> int:
    """
    Подготавливает окружение и запускает консольное приложение.

    Функция связывает старую точку входа с новым CLI-слоем. Все меню и команды
    находятся в ``pipe_bend_pinn.app.cli``, а техническая подготовка запуска — в
    ``pipe_bend_pinn.app.runtime``.
    """
    bootstrap_environment()
    # Импорт CLI выполняется после проверки venv и зависимостей.
    from pipe_bend_pinn.app.cli import guarded_main

    return guarded_main()


if __name__ == "__main__":
    sys.exit(main())
'''


def write_entry_point() -> None:
    """Перезаписывает pipe_bend_main.py как тонкий адаптер."""
    print("\\n[5/8] Обновление точки входа pipe_bend_main.py...")
    (BASE / "pipe_bend_main.py").write_text(PIPE_BEND_MAIN, encoding="utf-8")
    print("  ~ pipe_bend_main.py")

    # run.bat остаётся прежним, но для надёжности пересоздаём его
    run_bat = (
        "@echo off\r\n"
        'cd /d "%~dp0"\r\n'
        "chcp 65001 > nul\r\n"
        "set PYTHONIOENCODING=utf-8\r\n"
        "call venv\\Scripts\\activate.bat\r\n"
        "python pipe_bend_main.py\r\n"
        "pause\r\n"
    )
    (BASE / "run.bat").write_text(run_bat, encoding="utf-8", newline="\r\n")
    print("  ~ run.bat")


# =============================================================================
# Миграция данных в workspace
# =============================================================================

DATA_MIGRATION = {
    "Geometry/Data": WORKSPACE / "geometry" / "domains",
    "Geometry/pipe_bend_geoms.yaml": WORKSPACE / "geometry" / "pipe_bend_geoms.yaml",
    "Sampler/Data": WORKSPACE / "sampling" / "collocation",
    "Initial/Data": WORKSPACE / "initial" / "fields",
    "Initial/Plots": WORKSPACE / "initial" / "plots",
    "Fluids": WORKSPACE / "libraries" / "fluids",
    "Conditions": WORKSPACE / "libraries" / "conditions",
    "Checkpoints": WORKSPACE / "experiments",
    "Configs": WORKSPACE / "configs",
}


def migrate_data() -> None:
    """Копирует данные из старых корневых папок в workspace."""
    print("\\n[6/8] Миграция данных в workspace...")
    for src, dst in DATA_MIGRATION.items():
        src_path = BASE / src
        if src_path.is_dir():
            if not src_path.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            copied = 0
            for item in src_path.iterdir():
                if item.is_file():
                    target = dst / item.name
                    if not target.exists():
                        shutil.copy2(item, target)
                        copied += 1
            print(f"  → {src} → {dst.relative_to(BASE)} ({copied} файлов)")
        elif src_path.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src_path, dst)
                print(f"  → {src} → {dst.relative_to(BASE)}")


# =============================================================================
# Конфигурация
# =============================================================================

CONFIG_YAML = """# Пример конфигурационного файла для Pipe-Bend PINN
# (обновлённая версия: данные живут в workspace/)

project_name: pipe_bend

flow_type: planar

architecture:
  hidden_layers:
    - 50
    - 50
    - 50
    - 50
  activation: tanh
  output_dim: 3
  output_bias: [0.1, 0.0, 0.0]
  coordinate_scaling:
    enabled: false
    scale_factor: 1.0
    input_min: -6.0
    input_max: 6.0

pde:
  system: incompressible_navier_stokes
  flow_type: planar

boundary_conditions:
  type: soft

loss_weights:
  pde_weights:
    cont: 10.0
    mom_x: 10.0
    mom_y: 10.0
  bc_weights:
    inlet: 50.0
    outlet: 50.0
    wall: 50.0
  adaptive_bc: false
  use_sa_weights: false
  sa_apply_to_bc: true
  integral_condition: true
  integral_weight: 1000000.0

training:
  epochs: 2000
  batch_size: 1200
  gradient_accumulation_steps: 50
  two_stage: true
  stage1_epochs: 0
  stage2_pde_weight_start: 10.0
  stage2_pde_weight_end: 100.0
  stage2_ramp_epochs: 500
  lr_schedule: constant
  lr_cyclic_period: 100
  lr_cyclic_amplitude: 0.5
  lr_detector_factor: 2.0
  lr_detector_patience: 10
  lr_detector_recovery: 50
  reset_interval: 0
  reset_noise: 0.01
  grad_clip: null
  rar_enable: true
  rar_interval: 100
  rar_num_points: 200
  switch_to_lbfgs_epoch: 0
  lbfgs_max_iter: 20
  lbfgs_history_size: 100
  early_stopping:
    enabled: false
    patience: 100
    min_delta: 1e-6
    restore_best: true

optimizer_params:
  lr: 0.001

process:
  load_model_path: null
  device: auto
  num_gpus: 1
  dtype: float32
  use_amp: true
  seed: 42

output:
  console_only: false
  graphical_window:
    loss: true
    pde: true
    bc: false
  plot_frequency: 1

data:
  points_file: workspace/initial/fields/initial.npy

reference:
  path: null

diagnostics:
  interval: 50
  plot_residual_map: true
  plot_profiles: true
  profile_sections: [0.2, 0.5, 0.8]
  plot_spectrum: true
  spectrum_section_x: 0.5
  plot_2d_spectrum: false
  spectrum_2d_resolution: 256
  compute_grad_cosine: false
  compute_ntk_condition: false
  save_rar_history: true

enable_key_listener: true
checkpoint_dir: workspace/experiments
checkpoint:
  interval: 100
  name_pattern: '{project_name}_{timestamp}'
  model_name_pattern: '{project_name}_{timestamp}'
"""


def write_configs() -> None:
    """Записывает config.yaml в workspace/configs/."""
    print("\\n[7/8] Настройка конфигурации...")
    configs = WORKSPACE / "configs"
    configs.mkdir(parents=True, exist_ok=True)

    # config.yaml — новая версия (только если нет существующей в workspace)
    target = configs / "config.yaml"
    if not target.exists():
        target.write_text(CONFIG_YAML, encoding="utf-8")
        print(f"  + workspace/configs/config.yaml")

    # config.yaml.example — как пример
    example = configs / "config.yaml.example"
    if not example.exists():
        example.write_text(CONFIG_YAML, encoding="utf-8")
        print(f"  + workspace/configs/config.yaml.example")


# =============================================================================
# .gitignore
# =============================================================================

GITIGNORE = """# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE and OS files
.vs/
.idea/
.vscode/
.claude/
*.swp
*.swo
*~
.DS_Store
*.suo
*.vsidx
*.log

# Archives
*.zip
*.rar
*.7z
*.tar.gz

# Generated numeric and mesh artifacts
*.msh
*.npy
*.npz
Figure_*.png

# Legacy root-level data folders. They should not reappear after refactoring.
/Geometry/Data/
/Geometry/*.png
/Sampler/Data/
/Initial/Data/
/Initial/Plots/
/Fluids/
/Conditions/
/Checkpoints/
/Configs/
/workspace/backups/
/workspace/experiments/
/workspace/geometry/domains/
/workspace/geometry/metadata/
/workspace/initial/
/workspace/logs/
/workspace/pictures/
/workspace/sampling/
"""


def write_gitignore() -> None:
    """Обновляет .gitignore (перезаписывает)."""
    print("\n[8/8] Обновление .gitignore...")
    (BASE / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    print("  ~ .gitignore")


# =============================================================================
# Главная функция
# =============================================================================

def main() -> int:
    """Выполняет реструктуризацию проекта."""
    print("=" * 70)
    print("РЕСТРУКТУРИЗАЦИЯ PIPE-BEND PINN".center(70))
    print("=" * 70)

    ensure_dirs()
    migrate_auto_files()
    write_special_files()
    write_legacy_shims()
    patch_root_scripts()
    write_entry_point()
    migrate_data()
    write_configs()
    write_gitignore()

    print("\\n" + "=" * 70)
    print("✅ РЕСТРУКТУРИЗАЦИЯ ЗАВЕРШЕНА".center(70))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())