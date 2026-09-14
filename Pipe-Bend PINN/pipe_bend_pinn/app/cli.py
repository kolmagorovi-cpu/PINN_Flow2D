#!/usr/bin/env python3
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
    ║  • Начальное поле: Пуазейль / функция тока                    ║
    ║  • FVM-сетка Mesh2D и контроль неразрывности                  ║
    ║  • Решение уравнений Навье-Стокса PINNs                      ║
    ╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)


def print_menu_header(title: str) -> None:
    """Печатает единый заголовок для всех консольных меню."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ {title} ═══{Colors.RESET}")


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
    print(f"\n{Colors.YELLOW}┌─ Запуск модуля: {module_info['name']}{Colors.RESET}")
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
        print(f"{Colors.GREEN}▶ Запуск...{Colors.RESET}\n")
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
            print(f"\n{Colors.YELLOW}⚠️  Возврат в главное меню{Colors.RESET}")
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
                print(f"\n{Colors.GREEN}👋 Программа завершена. До свидания!{Colors.RESET}")
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
                    print(f"\n{Colors.CYAN}Использую: {default_cfg}{Colors.RESET}")
                    from pipe_bend_pinn.app.quick_setup import run_from_config
                    run_from_config(default_cfg)
                else:
                    print(f"{Colors.RED}❌ config.yaml не найден в workspace/configs/.{Colors.RESET}")
                    print(f"{Colors.YELLOW}Создайте config.yaml или используйте --config file.yaml{Colors.RESET}")
                    input("\nНажмите Enter...")
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
            print(f"\n\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
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
        print(f"\n\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
        print(f"{Colors.GREEN}👋 До свидания!{Colors.RESET}")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n{Colors.RED}❌ Критическая ошибка: {exc}{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(guarded_main())
