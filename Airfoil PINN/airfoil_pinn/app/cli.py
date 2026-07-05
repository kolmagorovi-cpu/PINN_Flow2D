#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консольный интерфейс Airfoil PINN.

Модуль отвечает только за навигацию пользователя: главное меню, меню модулей,
настройки и прямой запуск команд. Расчетные блоки подключаются через реестр
``module_registry`` и остаются отделенными от UI-логики.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any

from airfoil_pinn.app.module_registry import (
    ModuleDescriptor,
    default_modules,
    find_by_command,
    modules_as_legacy_dicts,
)
from airfoil_pinn.utils.ui_utils import (
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
    ║                    AEROPINN v1.0 ({current_year})                     ║
    ║         Physics-Informed Neural Networks for Airfoils        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  • Моделирование потенциального течения                      ║
    ║  • Генерация геометрии профилей                              ║
    ║  • Адаптивная расстановка точек коллокации                   ║
    ║  • Решение уравнений Навье-Стокса PINNs                      ║
    ╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)


def print_menu_header(title: str) -> None:
    """Печатает единый заголовок для всех консольных меню."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ {title} ═══{Colors.RESET}")


def print_module_info(modules: list[dict[str, Any]]) -> None:
    """
    Выводит список доступных расчетных модулей.

    Список приходит из ``module_registry``. CLI только отображает его и не
    знает внутреннего устройства конкретных расчетных блоков.
    """
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
        {"num": 2, "desc": "Настройки программы"},
        {"num": 3, "desc": "Справка"},
        {"num": 4, "desc": "Очистить экран"},
        {"num": 0, "desc": "Выход"},
    ]

    for item in menu_items:
        if item["num"] == 0:
            print(
                f"  {Colors.BOLD}{Colors.RED}[{item['num']}]{Colors.RESET} "
                f"{Colors.RED}{item['desc']}{Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.BOLD}{Colors.GREEN}[{item['num']}]{Colors.RESET} "
                f"{item['desc']}"
            )

    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")


def print_settings_menu() -> None:
    """
    Выводит меню настроек консольной оболочки.

    На этом этапе настройки остаются локальными для сессии. Позже этот блок
    можно связать с единым конфигурационным слоем.
    """
    print_menu_header("НАСТРОЙКИ")

    settings_items = [
        {"num": 1, "desc": "Цветной вывод", "value": True},
        {"num": 2, "desc": "Автосохранение", "value": True},
        {"num": 3, "desc": "Подробный вывод", "value": False},
        {"num": 4, "desc": "Уровень логирования", "value": "INFO"},
        {"num": 0, "action": "back", "desc": "Назад в главное меню"},
    ]

    for item in settings_items:
        if item.get("action") == "back":
            print(
                f"  {Colors.BOLD}{Colors.YELLOW}[{item['num']}]{Colors.RESET} "
                f"{Colors.YELLOW}{item['desc']}{Colors.RESET}"
            )
            continue

        if item["value"]:
            value = f"{Colors.GREEN}{item['value']}{Colors.RESET}"
        else:
            value = f"{Colors.RED}{item['value']}{Colors.RESET}"
        print(
            f"  {Colors.BOLD}{Colors.BLUE}[{item['num']}]{Colors.RESET} "
            f"{item['desc']}: {value}"
        )

    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.YELLOW}💡 Для изменения настройки введите ее номер{Colors.RESET}")


def print_help() -> None:
    """Выводит справочную информацию по консольной программе."""
    print_menu_header("СПРАВКА")

    help_text = f"""
    {Colors.BOLD}Как пользоваться программой:{Colors.RESET}

    1. В главном меню выберите {Colors.GREEN}[1]{Colors.RESET} для работы с модулями
    2. В меню модулей выберите номер нужного модуля
    3. Следуйте инструкциям выбранного модуля

    {Colors.BOLD}Прямой запуск без меню:{Colors.RESET}
    • python -m airfoil_pinn geometry
    • python -m airfoil_pinn sampling
    • python -m airfoil_pinn fluids
    • python -m airfoil_pinn conditions
    • python -m airfoil_pinn init
    • python -m airfoil_pinn train --config Configs/FVM7.yaml --experiment-dir Checkpoints/run1

    {Colors.BOLD}Горячие клавиши:{Colors.RESET}
    • {Colors.CYAN}0{Colors.RESET} - вернуться в главное меню / выход
    • {Colors.CYAN}Enter{Colors.RESET} - подтвердить ввод
    • {Colors.CYAN}Ctrl+C{Colors.RESET} - прервать выполнение
    """
    print(help_text)
    print(f"{Colors.CYAN}{'─' * 60}{Colors.RESET}")


def load_module(module_info: dict[str, Any]) -> Callable[[], None] | None:
    """
    Загружает функцию расчетного модуля по записи из реестра.

    Динамический импорт сохраняет прежнюю архитектуру меню: тяжелые модули
    загружаются только при выборе пользователем, а не при старте приложения.
    """
    try:
        module = importlib.import_module(module_info["module"])
        return getattr(module, module_info["func"])
    except ImportError as exc:
        print(f"{Colors.RED}❌ Ошибка загрузки модуля '{module_info['module']}':{Colors.RESET}")
        print(f"   {exc}")
        print(f"   {Colors.YELLOW}Убедитесь, что файл существует в директории{Colors.RESET}")
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
    """
    Выполняет один расчетный модуль из меню или прямой команды.

    Вся обработка ошибок запуска находится здесь, чтобы главное меню и
    argparse-команды использовали одинаковый путь выполнения.
    """
    print(f"\n{Colors.YELLOW}┌─ Запуск модуля: {module_info['name']}{Colors.RESET}")
    print(f"{Colors.YELLOW}│{Colors.RESET} Код: {module_info['code']}")
    print(f"{Colors.YELLOW}│{Colors.RESET} Статус: {module_info['status']}")
    print(
        f"{Colors.YELLOW}│{Colors.RESET} "
        f"Описание: {module_info.get('description', 'Нет описания')}"
    )
    print(f"{Colors.YELLOW}└─{Colors.RESET}")

    if module_info["status"] != "готов":
        print(
            f"{Colors.YELLOW}⚠️  Модуль в разработке. "
            f"Доступные функции ограничены.{Colors.RESET}\n"
        )
        if module_info["status"] == "разработка":
            print(f"{Colors.CYAN}Заглушка модуля {module_info['name']}{Colors.RESET}")
            print("Функциональность будет доступна в следующей версии.")
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
                print(
                    f"{Colors.RED}❌ Неверный номер. "
                    f"Введите число от 0 до {len(modules)}{Colors.RESET}"
                )
                pause_and_continue("Нажмите Enter...")

        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            pause_and_continue("Нажмите Enter...")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Возврат в главное меню{Colors.RESET}")
            break


def run_settings_menu() -> None:
    """
    Управляет настройками консольной сессии.

    Блок пока сохраняет прежнее поведение: настройки живут только во время
    работы программы и не меняют конфиги экспериментов.
    """
    settings: dict[str, Any] = {
        "color": True,
        "autosave": True,
        "verbose": False,
        "log_level": "INFO",
    }

    while True:
        clear_screen()
        print_banner()
        print_settings_menu()

        try:
            choice = get_menu_choice(
                "Выберите действие",
                valid_range=(1, 4),
                allow_zero=True,
                zero_action="назад",
            )

            if choice == "0":
                break

            index = int(choice)
            if index == 1:
                settings["color"] = not settings["color"]
                print(f"{Colors.GREEN}✓ Цветной вывод: {settings['color']}{Colors.RESET}")
            elif index == 2:
                settings["autosave"] = not settings["autosave"]
                print(f"{Colors.GREEN}✓ Автосохранение: {settings['autosave']}{Colors.RESET}")
            elif index == 3:
                settings["verbose"] = not settings["verbose"]
                print(f"{Colors.GREEN}✓ Подробный вывод: {settings['verbose']}{Colors.RESET}")
            elif index == 4:
                levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
                if settings["log_level"] in levels:
                    current_index = levels.index(settings["log_level"])
                else:
                    current_index = 1
                settings["log_level"] = levels[(current_index + 1) % len(levels)]
                print(
                    f"{Colors.GREEN}✓ Уровень логирования: "
                    f"{settings['log_level']}{Colors.RESET}"
                )
            else:
                print(f"{Colors.RED}❌ Неверный номер{Colors.RESET}")

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
                valid_range=(1, 4),
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
                run_settings_menu()
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
            else:
                print(f"{Colors.RED}❌ Неверный номер. Введите число от 0 до 4{Colors.RESET}")
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
    """
    Создает парсер прямых консольных команд.

    Без аргументов запускается интерактивное меню. С аргументом ``command`` CLI
    сразу передает управление нужному модулю, что удобно для пакетной работы.
    """
    parser = argparse.ArgumentParser(
        prog="airfoil_pinn",
        description="Консольный запуск модулей Airfoil PINN",
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
    parser.add_argument(
        "--experiment-dir",
        help="Папка эксперимента для прямого запуска обучения",
    )
    return parser


def _run_training_command(args: argparse.Namespace) -> None:
    """
    Запускает обучение без меню, если передан ``--config``.

    При отсутствии ``--config`` команда ``train`` ведет себя как пункт меню и
    открывает интерактивное меню обучения из ``PINN_Learning``.
    """
    if not args.config:
        module = find_by_command("train")
        if module is not None:
            run_module(module.as_legacy_dict())
        return

    if not args.experiment_dir:
        raise SystemExit("Для train --config необходимо указать --experiment-dir")

    learning_module = importlib.import_module("PINN_Learning")
    learning_module.run_experiment(args.config, args.experiment_dir)


def run_command(args: argparse.Namespace) -> None:
    """Выполняет прямую команду пакетного CLI."""
    if args.command == "train":
        _run_training_command(args)
        return

    module = find_by_command(args.command)
    if module is None:
        raise SystemExit(f"Неизвестная команда: {args.command}")

    run_module(module.as_legacy_dict())


def main(argv: Sequence[str] | None = None) -> None:
    """
    Главная функция консольного приложения.

    ``argv`` нужен для тестов и будущих сценариев автоматизации. При обычном
    запуске используется ``sys.argv``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command:
        run_command(args)
    else:
        run_interactive_menu()


def guarded_main(argv: Sequence[str] | None = None) -> int:
    """
    Запускает CLI с верхнеуровневой обработкой ошибок.

    Старый ``Airfoil_PINN.py`` использует эту функцию, чтобы сохранить прежнее
    дружелюбное сообщение при ``Ctrl+C`` и критических ошибках.
    """
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
