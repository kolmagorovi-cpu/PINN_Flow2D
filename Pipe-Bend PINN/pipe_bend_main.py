#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный модуль управления Pipe-Bend PINN.

Предоставляет главное меню для навигации по модулям проекта.

Модули:
- Геометрия (G2D) — построение расчётной области отвода 90°
- Дискретизация (S2DR) — генерация точек коллокации
- Свойства жидкости (Fluids) — задание плотности и вязкости
- Граничные условия (Conditions) — давление на входе/выходе, профиль скорости
- Инициализация (Initializer) — нормализация данных и вычисление SDF
- Обучение PINN (PINNLearning) — настройка и запуск обучения

Автор: Колмагоров И.Д., Бельчиков Д.А.
Версия: 1.1
"""

import importlib
import datetime
import sys
import os
from typing import Dict, Optional, List, Callable, Any

from constants import IS_WINDOWS

# Корень проекта (абсолютный путь)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class Colors:
    """Цвета для консольного вывода."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    if IS_WINDOWS:
        GREEN = BLUE = YELLOW = RED = CYAN = MAGENTA = BOLD = UNDERLINE = RESET = ''


def clear_screen() -> None:
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print('\033c')


def print_banner() -> None:
    current_year = datetime.datetime.now().year
    banner = f"""
    {Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
    ║                  PIPEBENDPINN v1.1 ({current_year})               ║
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
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ {title} ═══{Colors.RESET}")


def print_module_info(modules: List[Dict[str, Any]]) -> None:
    print_menu_header("ДОСТУПНЫЕ МОДУЛИ")
    for idx, info in enumerate(modules, 1):
        if info["status"] == "готов":
            status_icon = f"{Colors.GREEN}✅{Colors.RESET}"
            status_color = Colors.GREEN
        elif info["status"] == "разработка":
            status_icon = f"{Colors.YELLOW}🔄{Colors.RESET}"
            status_color = Colors.YELLOW
        else:
            status_icon = f"{Colors.BLUE}📅{Colors.RESET}"
            status_color = Colors.BLUE
        print(f"  {Colors.BOLD}{Colors.CYAN}[{idx}]{Colors.RESET} {Colors.BOLD}{info['name']}{Colors.RESET}")
        print(f"      {status_color}Код: {info['code']} | Статус: {info['status']} {status_icon}{Colors.RESET}")
        if "description" in info:
            print(f"      {Colors.CYAN}↳ {info['description']}{Colors.RESET}")
        if idx < len(modules):
            print()
    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def print_main_menu() -> None:
    print_menu_header("ГЛАВНОЕ МЕНЮ")
    menu_items = [
        {"num": 1, "action": "modules", "desc": "Работа с модулями"},
        {"num": 4, "action": "quick_setup", "desc": "🚀 Быстрая подготовка (Геом+Коллок+Иниц)"},
        {"num": 5, "action": "config_run", "desc": "⚙ Запуск обучения из config.yaml"},
        {"num": 3, "action": "help", "desc": "Справка"},
        {"num": 2, "action": "clear", "desc": "Очистить экран"},
        {"num": 0, "action": "exit", "desc": "Выход"}
    ]
    for item in menu_items:
        if item["num"] == 4 or item["num"] == 5:
            num_color = Colors.YELLOW
            desc_color = Colors.YELLOW
        elif item["num"] == 0:
            num_color = Colors.RED
            desc_color = Colors.RED
        else:
            num_color = Colors.GREEN
            desc_color = ""
        print(f"  {Colors.BOLD}{num_color}[{item['num']}]{Colors.RESET} {desc_color}{item['desc']}{Colors.RESET}")
    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def run_modules_menu(modules: List[Dict[str, Any]]) -> None:
    while True:
        clear_screen()
        print_banner()
        print_module_info(modules)
        print(f"\n{Colors.BOLD}{Colors.BLUE}Выберите модуль (1-{len(modules)}) или 0 для возврата:{Colors.RESET} ", end="")
        try:
            choice = input().strip()
            if choice == "0":
                break
            idx = int(choice)
            if 1 <= idx <= len(modules):
                module_info = modules[idx-1]
                print(f"\n{Colors.YELLOW}┌─ Запуск модуля: {module_info['name']}{Colors.RESET}")
                print(f"{Colors.YELLOW}│ Код: {module_info['code']}{Colors.RESET}")
                if module_info["status"] == "готов":
                    try:
                        module = importlib.import_module(module_info["module"])
                        func = getattr(module, module_info["func"])
                        print(f"{Colors.GREEN}▶ Запуск...{Colors.RESET}\n")
                        func()
                    except Exception as e:
                        print(f"{Colors.RED}❌ Ошибка: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️ Модуль в разработке.{Colors.RESET}")
                print(f"\n{Colors.CYAN}Нажмите Enter чтобы продолжить...{Colors.RESET}", end="")
                input()
            else:
                print(f"{Colors.RED}❌ Неверный номер{Colors.RESET}")
                input()
        except (ValueError, KeyboardInterrupt):
            break


def print_help() -> None:
    print_menu_header("СПРАВКА")
    help_text = f"""
    {Colors.BOLD}Как пользоваться программой:{Colors.RESET}
    1. В главном меню выберите [1] для работы с модулями
    2. В меню модулей выберите номер нужного модуля
    3. Следуйте инструкциям выбранного модуля

    {Colors.BOLD}Обязательная последовательность:{Colors.RESET}
    1. {Colors.CYAN}Геометрия{Colors.RESET} — загрузка конфигурации отвода, создание области
    2. {Colors.CYAN}Дискретизация{Colors.RESET} — генерация точек коллокации
    3. {Colors.CYAN}Жидкость{Colors.RESET} — выбор/создание жидкости
    4. {Colors.CYAN}Граничные условия{Colors.RESET} — давление и скорость
    5. {Colors.CYAN}Инициализация{Colors.RESET} — нормализация, SDF
    6. {Colors.CYAN}Обучение PINN{Colors.RESET} — настройка и запуск

    {Colors.BOLD}Быстрый старт:{Colors.RESET}
    - Выберите [4] Быстрая подготовка — выполнит шаги 1, 2, 5
    - Затем запустите модуль Обучение PINN
    """
    print(help_text)
    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def quick_setup() -> None:
    """
    Быстрая подготовка всех данных для обучения.
    Выполняет: Геометрия → Коллокация → Инициализация
    """
    print(f"\n{Colors.YELLOW}🚀 БЫСТРАЯ ПОДГОТОВКА ДАННЫХ{Colors.RESET}")
    print(f"{Colors.CYAN}Этап 1/3: Геометрия отвода{Colors.RESET}")
    try:
        from Geometry import g2d
        # g2d() интерактивная, вызываем в автоматическом режиме
        from Geometry import PipeBendGeometry, GeometryConfig
        geom_config = GeometryConfig(save_format='npy', plot_show=False, verbose=True,
                                     output_dir=os.path.join(PROJECT_ROOT, "Geometry", "Data"))
        geom = PipeBendGeometry(config=geom_config)
        if not geom.load_from_yaml(bend_name='PB_R100'):
            print(f"{Colors.RED}❌ Ошибка загрузки геометрии{Colors.RESET}")
            input("Нажмите Enter...")
            return
        geom.create_domain()
        geom.save('pipe_bend')
        print(f"{Colors.GREEN}✅ Геометрия создана{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка геометрии: {e}{Colors.RESET}")
        input("Нажмите Enter...")
        return

    print(f"\n{Colors.CYAN}Этап 2/3: Генерация точек коллокации{Colors.RESET}")
    try:
        import numpy as np
        import matplotlib.path as mplPath
        from shapely.geometry import LineString

        # Загружаем геометрию
        domain_path = os.path.join(PROJECT_ROOT, "Geometry", "Data", "pipe_bend_domain.npy")
        if not os.path.exists(domain_path):
            print(f"{Colors.RED}❌ Файл геометрии не найден: {domain_path}{Colors.RESET}")
            input("Нажмите Enter...")
            return
        domain = np.load(domain_path)
        if domain.ndim == 1 and domain.dtype.names is not None:
            domain = np.column_stack((domain['x'], domain['y']))

        # Параметры сетки из config.yaml (с дефолтами)
        _cfg_step = 0.05
        _cfg_num_bnd = 200
        try:
            from config_manager import ConfigManager
            _cfg = ConfigManager()
            _cfg_step = _cfg.config.get('sampling', {}).get('step_size', 0.05)
            _cfg_num_bnd = _cfg.config.get('sampling', {}).get('num_boundary', 200)
        except Exception:
            pass
        
        R = 1.0  # bend_radius из PB_R100
        W = 0.5  # pipe_width из PB_R100
        R_inner = R - W / 2
        x_min, y_min = np.min(domain, axis=0)
        x_max, y_max = np.max(domain, axis=0)
        step_size = _cfg_step

        path = mplPath.Path(domain)
        x_coords = np.arange(x_min, x_max, step_size)
        y_coords = np.arange(y_min, y_max, step_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.column_stack((X.ravel(), Y.ravel()))
        inside = path.contains_points(points)
        interior_points = points[inside]

        # Граничные точки
        line = LineString(domain)
        total_length = line.length
        num_boundary = _cfg_num_bnd
        boundary_points = []
        boundary_types = []
        for i in range(num_boundary):
            distance = i * total_length / num_boundary
            pt = line.interpolate(distance)
            xb, yb = pt.x, pt.y
            if abs(xb - x_min) < 0.05 and yb < -R_inner:
                btype = 'inlet'
            elif abs(yb - y_max) < 0.05 and xb > x_min + W:
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

        sampler_dir = os.path.join(PROJECT_ROOT, "Sampler", "Data")
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
        from collections import Counter
        type_counts = Counter(all_types)
        for bt, cnt in type_counts.most_common():
            print(f"      {bt}: {cnt}")
        print(f"{Colors.GREEN}✅ Точки коллокации созданы (всего: {len(structured)}){Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка коллокации: {e}{Colors.RESET}")
        input("Нажмите Enter...")
        return

    print(f"\n{Colors.CYAN}Этап 3/3: Инициализация данных{Colors.RESET}")
    try:
        from Initializer import Initializer
        init = Initializer()
        init.load_sampler_points()
        init.extract_wall_points()

        # Берём первую жидкость и условия
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

        init.initialize_fields(pressure_mode='linear_x', velocity_initial_mode='parabolic')
        init.save(description='initial')
        print(f"      U_inf = {init.U_inf:.3f} м/с, Re = {init.Re:.2e}")
        print(f"{Colors.GREEN}✅ Инициализация завершена{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка инициализации: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        input("Нажмите Enter...")
        return

    print(f"\n{Colors.GREEN}✅✅✅ Быстрая подготовка завершена!{Colors.RESET}")
    print(f"{Colors.CYAN}Теперь можно запускать Обучение PINN.{Colors.RESET}")
    print(f"{Colors.CYAN}Файл данных: Initial/Data/initial.npy{Colors.RESET}")
    input("Нажмите Enter...")


def run_from_config(config_path: str) -> None:
    """Запускает полный пайплайн из config.yaml без интерактивного меню."""
    print(f"\n{Colors.YELLOW}🚀 ЗАПУСК ИЗ КОНФИГА: {config_path}{Colors.RESET}")
    from config_manager import ConfigManager
    
    cfg = ConfigManager(config_path=config_path)
    print(f"   Проект: {cfg.config.get('project_name', 'unnamed')}")
    print(f"   Файл точек: {cfg.config['data'].get('points_file', 'не указан')}")
    
    # Шаг 1: Быстрая подготовка (если данных нет)
    points_file = cfg.config['data'].get('points_file')
    if not points_file or not os.path.exists(str(points_file)):
        print(f"{Colors.YELLOW}⚠️ Данные не найдены. Запускаю быструю подготовку...{Colors.RESET}")
        quick_setup()
        # Перезагружаем конфиг — points_file теперь должен быть
        cfg = ConfigManager(config_path=config_path)
    
    # Шаг 2: Запуск обучения
    print(f"{Colors.CYAN}▶ Запуск обучения...{Colors.RESET}")
    cfg.config['flow_type'] = 'planar'
    bc_weights = cfg.config['loss_weights']['bc_weights']
    for key in ['symmetry', 'axis_symmetry', 'symmetry_grad', 'axis_symmetry_grad']:
        bc_weights.pop(key, None)
    if 'wall' not in bc_weights:
        bc_weights['wall'] = 5000.0
    
    from solver import PINNSolver
    try:
        solver = PINNSolver(cfg.config)
        solver.train()
    except Exception as e:
        print(f"{Colors.RED}❌ Ошибка: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"{Colors.GREEN}✅ Обучение завершено{Colors.RESET}")
    input("Нажмите Enter...")


def main() -> None:
    # Обрабатываем аргументы командной строки
    import argparse
    parser = argparse.ArgumentParser(description="Pipe-Bend PINN")
    parser.add_argument('--config', type=str, default=None,
                        help='Запуск из config.yaml (неинтерактивный режим)')
    args, _ = parser.parse_known_args()
    
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        if not os.path.exists(config_path):
            print(f"❌ Файл конфигурации не найден: {config_path}")
            sys.exit(1)
        run_from_config(config_path)
        sys.exit(0)
    
    modules = [
        {
            "code": "G2D",
            "name": "Геометрия отвода 90°",
            "module": "Geometry",
            "func": "g2d",
            "status": "готов",
            "description": "Построение расчётной области отвода"
        },
        {
            "code": "S2DR",
            "name": "Дискретизация (коллокация)",
            "module": "test_pipeline",
            "func": "generate_collocation",  # заглушка, используем quick_setup
            "status": "готов",
            "description": "Генерация точек коллокации (через Быструю подготовку)"
        },
        {
            "code": "Fluids",
            "name": "Свойства жидкости",
            "module": "Fluids",
            "func": "fluids_menu",
            "status": "готов",
            "description": "Задание плотности и вязкости"
        },
        {
            "code": "Conditions",
            "name": "Граничные условия",
            "module": "Conditions",
            "func": "conditions_menu",
            "status": "готов",
            "description": "Давление на входе/выходе, профиль скорости"
        },
        {
            "code": "Initializer",
            "name": "Инициализация",
            "module": "Initializer",
            "func": "initialize_menu",
            "status": "готов",
            "description": "Нормализация данных и вычисление SDF"
        },
        {
            "code": "PINNLearning",
            "name": "Обучение PINN",
            "module": "PINN_Learning",
            "func": "main_menu",
            "status": "готов",
            "description": "Настройка конфигурации и запуск обучения"
        }
    ]
    clear_screen()
    print_banner()
    while True:
        print_main_menu()
        print(f"\n{Colors.BOLD}{Colors.BLUE}Выберите действие (0-5):{Colors.RESET} ", end="")
        try:
            choice = input().strip()
            if choice == "0":
                print(f"\n{Colors.GREEN}👋 До свидания!{Colors.RESET}")
                break
            elif choice == "1":
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
                input(f"\n{Colors.CYAN}Нажмите Enter...{Colors.RESET}")
                clear_screen()
                print_banner()
            elif choice == "4":
                clear_screen()
                print_banner()
                quick_setup()
                clear_screen()
                print_banner()
            elif choice == "5":
                clear_screen()
                print_banner()
                default_cfg = os.path.join(PROJECT_ROOT, "config.yaml")
                if os.path.exists(default_cfg):
                    print(f"\n{Colors.CYAN}Использую: {default_cfg}{Colors.RESET}")
                    run_from_config(default_cfg)
                else:
                    print(f"{Colors.RED}❌ config.yaml не найден в корне проекта.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Создайте config.yaml или используйте --config file.yaml{Colors.RESET}")
                    input("\nНажмите Enter...")
                clear_screen()
                print_banner()
            else:
                print(f"{Colors.RED}❌ Неверный номер.{Colors.RESET}")
                input()
        except (ValueError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}⚠️ Выход{Colors.RESET}")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}👋 До свидания!{Colors.RESET}")
        sys.exit(0)