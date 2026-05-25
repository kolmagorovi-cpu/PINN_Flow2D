#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный модуль управления программой моделирования течения около профилей
с помощью физически-информированных нейронных сетей (PINNs).

Предоставляет главное меню для навигации по модулям проекта:
- Геометрия (G2D) — загрузка контура профиля и построение расчётной области
- Дискретизация (S2DR) — генерация точек коллокации
- Свойства жидкости (Fluids) — задание плотности и вязкости
- Граничные условия (Conditions) — давление и скорость на входе/выходе
- Инициализация (Initializer) — нормализация данных и вычисление SDF
- Обучение PINN (PINNLearning) — настройка и запуск обучения

Автор: Колмагоров И.Д.
Дата: 2026
Версия: 3.260226 (исправленная)

Example:
    >>> if __name__ == '__main__':
    ...     main()  # Запуск главного меню программы
"""

# =============================================================================
# ПРОВЕРКА ОКРУЖЕНИЯ (только stdlib, выполняется до всех остальных импортов)
# =============================================================================
import os as _os
import sys as _sys


def _bootstrap_environment() -> None:
    """Проверяет и при необходимости создаёт виртуальное окружение с зависимостями."""
    import importlib.util as _ilu
    import subprocess as _sp
    import shutil as _sh

    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
    _VENV_DIR = _os.path.join(_SCRIPT_DIR, 'venv')

    if _sys.platform == 'win32':
        _VENV_PY = _os.path.join(_VENV_DIR, 'Scripts', 'python.exe')
        _VENV_PIP = _os.path.join(_VENV_DIR, 'Scripts', 'pip.exe')
    else:
        _VENV_PY = _os.path.join(_VENV_DIR, 'bin', 'python')
        _VENV_PIP = _os.path.join(_VENV_DIR, 'bin', 'pip')

    # Пакеты: (import_name, pip_name)
    _PACKAGES = [
        ('yaml',             'pyyaml'),
        ('numpy',            'numpy'),
        ('torch',            'torch'),
        ('matplotlib',       'matplotlib'),
        ('scipy',            'scipy'),
        ('shapely',          'shapely'),
        ('typing_extensions','typing_extensions'),
    ]

    _in_our_venv = (
        _os.path.exists(_VENV_PY)
        and _os.path.abspath(_sys.executable) == _os.path.abspath(_VENV_PY)
    )

    def _ask(prompt: str) -> bool:
        try:
            return input(prompt).strip().lower() in ('y', 'yes', 'да', 'д')
        except (EOFError, KeyboardInterrupt):
            return False

    def _create_venv() -> bool:
        uv = _sh.which('uv')
        print("  Создание виртуального окружения...", flush=True)
        if uv:
            r = _sp.run([uv, 'venv', _VENV_DIR], capture_output=True, text=True)
        else:
            r = _sp.run([_sys.executable, '-m', 'venv', _VENV_DIR],
                        capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  Ошибка: {r.stderr.strip() or r.stdout.strip()}")
            return False
        print("  Виртуальное окружение создано.")
        return True

    def _pip_env() -> dict:
        # Сбрасываем SOCKS/HTTP прокси: pip не умеет работать с socks без PySocks,
        # а прямой доступ к PyPI работает без прокси.
        env = _os.environ.copy()
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                    'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key, None)
        env['NO_PROXY'] = '*'
        env['no_proxy'] = '*'
        return env

    def _ensure_pip() -> bool:
        """Устанавливает pip в venv через ensurepip если его нет."""
        if _os.path.exists(_VENV_PIP):
            return True
        r = _sp.run([_VENV_PY, '-m', 'ensurepip', '--upgrade'],
                    capture_output=True, text=True)
        return r.returncode == 0

    def _install_packages(pip_names: list) -> bool:
        print(f"  Установка: {', '.join(pip_names)} ...", flush=True)
        if not _ensure_pip():
            print("  Не удалось инициализировать pip в окружении.")
            return False
        r = _sp.run([_VENV_PY, '-m', 'pip', 'install'] + pip_names,
                    env=_pip_env())
        if r.returncode != 0:
            print("  Ошибка при установке. Проверьте подключение к интернету.")
            return False
        print("  Зависимости установлены.")
        return True

    # --- Случай 1: виртуальное окружение ещё не создано ---
    if not _os.path.exists(_VENV_DIR):
        print("\n  Виртуальное окружение не обнаружено.")
        if not _ask("  Создать и установить зависимости? [y/N]: "):
            print("  Продолжение без виртуального окружения (возможны ошибки).\n")
            return
        if not _create_venv():
            return
        _install_packages([pip for _, pip in _PACKAGES])
        # Перезапуск внутри нового окружения
        if _os.path.exists(_VENV_PY):
            print("  Перезапуск в виртуальном окружении...\n", flush=True)
            r = _sp.run([_VENV_PY] + _sys.argv)
            _sys.exit(r.returncode)
        return

    # --- Случай 2: venv есть, но мы запущены вне его — перезапускаемся ---
    if not _in_our_venv and _os.path.exists(_VENV_PY):
        r = _sp.run([_VENV_PY] + _sys.argv)
        _sys.exit(r.returncode)

    # --- Случай 3: мы внутри venv — проверяем наличие пакетов ---
    missing_pip = [pip for imp, pip in _PACKAGES if _ilu.find_spec(imp) is None]
    if missing_pip:
        print(f"\n  Отсутствуют пакеты: {', '.join(missing_pip)}")
        if _ask("  Установить недостающие пакеты? [y/N]: "):
            if _install_packages(missing_pip):
                print("  Перезапуск...\n", flush=True)
                r = _sp.run([_VENV_PY] + _sys.argv)
                _sys.exit(r.returncode)
        else:
            print("  Продолжение с отсутствующими пакетами (возможны ошибки).\n")


_bootstrap_environment()
del _bootstrap_environment

# =============================================================================
# ОСНОВНЫЕ ИМПОРТЫ
# =============================================================================
import importlib
import datetime
import sys
import os
from typing import Dict, Optional, List, Callable, Any

from constants import IS_WINDOWS


# =============================================================================
# ЦВЕТОВОЕ ОФОРМЛЕНИЕ
# =============================================================================

class Colors:
    """
    Цвета для консольного вывода.
    
    Работает в Unix-системах через ANSI escape-последовательности.
    Для Windows цвета отключаются (автоматически).
    
    Example:
        >>> print(f"{Colors.GREEN}Успех!{Colors.RESET}")
        >>> print(f"{Colors.BOLD}{Colors.RED}Ошибка!{Colors.RESET}")
    """
    GREEN: str = '\033[92m'
    BLUE: str = '\033[94m'
    YELLOW: str = '\033[93m'
    RED: str = '\033[91m'
    CYAN: str = '\033[96m'
    MAGENTA: str = '\033[95m'
    BOLD: str = '\033[1m'
    UNDERLINE: str = '\033[4m'
    RESET: str = '\033[0m'

    # Для Windows можно отключить цвета
    if IS_WINDOWS:
        GREEN = BLUE = YELLOW = RED = CYAN = MAGENTA = BOLD = UNDERLINE = RESET = ''


def clear_screen() -> None:
    """Очищает экран консоли"""
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print('\033c')


def print_banner() -> None:
    """Выводит баннер программы с информацией о версии"""
    current_year: int = datetime.datetime.now().year
    banner: str = f"""
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
    """Выводит заголовок меню"""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ {title} ═══{Colors.RESET}")


def print_module_info(modules: List[Dict[str, Any]]) -> None:
    """
    Выводит информацию о доступных модулях в виде нумерованного списка
    
    Args:
        modules: Список словарей с информацией о модулях
    """
    print_menu_header("ДОСТУПНЫЕ МОДУЛИ")
    
    for idx, info in enumerate(modules, 1):
        # Выбор иконки статуса
        if info["status"] == "готов":
            status_icon = f"{Colors.GREEN}✅{Colors.RESET}"
            status_color = Colors.GREEN
        elif info["status"] == "разработка":
            status_icon = f"{Colors.YELLOW}🔄{Colors.RESET}"
            status_color = Colors.YELLOW
        else:
            status_icon = f"{Colors.BLUE}📅{Colors.RESET}"
            status_color = Colors.BLUE
        
        # Форматированный вывод с номером
        print(f"  {Colors.BOLD}{Colors.CYAN}[{idx}]{Colors.RESET} {Colors.BOLD}{info['name']}{Colors.RESET}")
        print(f"      {status_color}Код: {info['code']} | Статус: {info['status']} {status_icon}{Colors.RESET}")
        
        # Краткое описание (если есть)
        if "description" in info:
            print(f"      {Colors.CYAN}↳ {info['description']}{Colors.RESET}")
        
        # Разделитель между модулями
        if idx < len(modules):
            print()
    
    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def print_main_menu() -> None:
    """Выводит главное меню программы"""
    print_menu_header("ГЛАВНОЕ МЕНЮ")

    menu_items: List[Dict[str, Any]] = [
        {"num": 1, "action": "modules", "desc": "Работа с модулями"},
        {"num": 2, "action": "settings", "desc": "Настройки программы"},
        {"num": 3, "action": "help", "desc": "Справка"},
        {"num": 4, "action": "clear", "desc": "Очистить экран"},
        {"num": 0, "action": "exit", "desc": "Выход"}
    ]

    for item in menu_items:
        if item["num"] == 0:
            print(f"  {Colors.BOLD}{Colors.RED}[{item['num']}]{Colors.RESET} {Colors.RED}{item['desc']}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}{Colors.GREEN}[{item['num']}]{Colors.RESET} {item['desc']}")

    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def print_settings_menu() -> None:
    """Выводит меню настроек"""
    print_menu_header("НАСТРОЙКИ")

    settings_items: List[Dict[str, Any]] = [
        {"num": 1, "setting": "color", "desc": "Цветной вывод", "value": True},
        {"num": 2, "setting": "autosave", "desc": "Автосохранение", "value": True},
        {"num": 3, "setting": "verbose", "desc": "Подробный вывод", "value": False},
        {"num": 4, "setting": "log_level", "desc": "Уровень логирования", "value": "INFO"},
        {"num": 0, "action": "back", "desc": "Назад в главное меню"}
    ]

    for item in settings_items:
        if "action" in item and item["action"] == "back":
            print(f"  {Colors.BOLD}{Colors.YELLOW}[{item['num']}]{Colors.RESET} {Colors.YELLOW}{item['desc']}{Colors.RESET}")
        else:
            # Показываем текущее значение
            value_str: str = f"{Colors.GREEN}{item['value']}{Colors.RESET}" if item['value'] else f"{Colors.RED}{item['value']}{Colors.RESET}"
            print(f"  {Colors.BOLD}{Colors.BLUE}[{item['num']}]{Colors.RESET} {item['desc']}: {value_str}")

    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")
    print(f"{Colors.YELLOW}💡 Для изменения настройки введите её номер{Colors.RESET}")


def print_help() -> None:
    """Выводит справочную информацию"""
    print_menu_header("СПРАВКА")

    help_text: str = f"""
    {Colors.BOLD}Как пользоваться программой:{Colors.RESET}

    1. В главном меню выберите {Colors.GREEN}[1]{Colors.RESET} для работы с модулями
    2. В меню модулей выберите номер нужного модуля
    3. Следуйте инструкциям выбранного модуля

    {Colors.BOLD}Статусы модулей:{Colors.RESET}
    {Colors.GREEN}✅ готов{Colors.RESET} - модуль полностью работает
    {Colors.YELLOW}🔄 разработка{Colors.RESET} - модуль в разработке (может быть нестабильным)
    {Colors.BLUE}📅 план{Colors.RESET} - модуль запланирован

    {Colors.BOLD}Горячие клавиши:{Colors.RESET}
    • {Colors.CYAN}0{Colors.RESET} - вернуться в главное меню / выход
    • {Colors.CYAN}Enter{Colors.RESET} - подтвердить ввод
    • {Colors.CYAN}Ctrl+C{Colors.RESET} - прервать выполнение

    {Colors.BOLD}Если возникли проблемы:{Colors.RESET}
    • Проверьте, что все файлы модулей находятся в той же директории
    • Убедитесь, что установлены все зависимости
    • Создайте issue на GitHub проекта
    """
    print(help_text)
    print(f"{Colors.CYAN}{'─'*60}{Colors.RESET}")


def load_module(module_info: Dict) -> Optional[callable]:
    """
    Загружает и возвращает функцию модуля
    
    Args:
        module_info: Словарь с информацией о модуле
        
    Returns:
        Функция модуля или None в случае ошибки
    """
    try:
        module = importlib.import_module(module_info["module"])
        func = getattr(module, module_info["func"])
        return func
    except ImportError as e:
        print(f"{Colors.RED}❌ Ошибка загрузки модуля '{module_info['module']}':{Colors.RESET}")
        print(f"   {e}")
        print(f"   {Colors.YELLOW}Убедитесь, что файл существует в директории{Colors.RESET}")
        return None
    except AttributeError as e:
        print(f"{Colors.RED}❌ Ошибка: функция '{module_info['func']}' не найдена в модуле{Colors.RESET}")
        return None
    except Exception as e:
        print(f"{Colors.RED}❌ Непредвиденная ошибка: {e}{Colors.RESET}")
        return None


def run_modules_menu(modules: List[Dict[str, Any]]) -> None:
    """
    Меню выбора и запуска модулей

    Args:
        modules: Список модулей
    """
    while True:
        clear_screen()
        print_banner()
        print_module_info(modules)

        print(f"\n{Colors.BOLD}{Colors.BLUE}Выберите модуль (1-{len(modules)}) или 0 для возврата:{Colors.RESET} ", end="")

        try:
            choice: str = input().strip()

            if choice == "0":
                break

            idx: int = int(choice)
            if 1 <= idx <= len(modules):
                module_info: Dict[str, Any] = modules[idx-1]

                print(f"\n{Colors.YELLOW}┌─ Запуск модуля: {module_info['name']}{Colors.RESET}")
                print(f"{Colors.YELLOW}│{Colors.RESET} Код: {module_info['code']}")
                print(f"{Colors.YELLOW}│{Colors.RESET} Статус: {module_info['status']}")
                print(f"{Colors.YELLOW}│{Colors.RESET} Описание: {module_info.get('description', 'Нет описания')}")
                print(f"{Colors.YELLOW}└─{Colors.RESET}")

                if module_info["status"] == "готов":
                    func: Optional[Callable] = load_module(module_info)
                    if func:
                        try:
                            print(f"{Colors.GREEN}▶ Запуск...{Colors.RESET}\n")
                            func()
                        except Exception as e:
                            print(f"{Colors.RED}❌ Ошибка при выполнении модуля:{Colors.RESET}")
                            print(f"   {e}")
                    else:
                        print(f"{Colors.RED}❌ Не удалось загрузить модуль{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️  Модуль в разработке. Доступные функции ограничены.{Colors.RESET}\n")

                    # Для модулей в разработке можно показать заглушку
                    if module_info["status"] == "разработка":
                        print(f"{Colors.CYAN}Заглушка модуля {module_info['name']}{Colors.RESET}")
                        print("Функциональность будет доступна в следующей версии.")

                print(f"\n{Colors.CYAN}Нажмите Enter чтобы продолжить...{Colors.RESET}", end="")
                input()

            else:
                print(f"{Colors.RED}❌ Неверный номер. Введите число от 0 до {len(modules)}{Colors.RESET}")
                print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
                input()

        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
            input()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Возврат в главное меню{Colors.RESET}")
            break


def run_settings_menu() -> None:
    """Меню настроек программы"""
    settings: Dict[str, Any] = {
        "color": True,
        "autosave": True,
        "verbose": False,
        "log_level": "INFO"
    }

    while True:
        clear_screen()
        print_banner()
        print_settings_menu()

        print(f"\n{Colors.BOLD}{Colors.BLUE}Выберите действие (0-4):{Colors.RESET} ", end="")

        try:
            choice: str = input().strip()

            if choice == "0":
                break

            idx: int = int(choice)
            if idx == 1:
                settings["color"] = not settings["color"]
                print(f"{Colors.GREEN}✓ Цветной вывод: {settings['color']}{Colors.RESET}")
            elif idx == 2:
                settings["autosave"] = not settings["autosave"]
                print(f"{Colors.GREEN}✓ Автосохранение: {settings['autosave']}{Colors.RESET}")
            elif idx == 3:
                settings["verbose"] = not settings["verbose"]
                print(f"{Colors.GREEN}✓ Подробный вывод: {settings['verbose']}{Colors.RESET}")
            elif idx == 4:
                # Циклическое изменение уровня логирования
                levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
                current_idx = levels.index(settings["log_level"]) if settings["log_level"] in levels else 1
                next_idx = (current_idx + 1) % len(levels)
                settings["log_level"] = levels[next_idx]
                print(f"{Colors.GREEN}✓ Уровень логирования: {settings['log_level']}{Colors.RESET}")
            else:
                print(f"{Colors.RED}❌ Неверный номер{Colors.RESET}")
            
            print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
            input()
            
        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
            input()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Возврат в главное меню{Colors.RESET}")
            break


def main() -> None:
    """Главная функция программы"""

    # Информация о модулях в виде списка для нумерации
    modules: List[Dict[str, Any]] = [
        {
            "code": "G2D",
            "name": "Геометрия (2D)",
            "module": "Geometry",
            "func": "g2d",
            "status": "готов",
            "description": "Загрузка контура профиля и построение расчётной области"
        },
        {
            "code": "S2DR",
            "name": "Дискретизация",
            "module": "Collocation",
            "func": "S2DR",
            "status": "готов",
            "description": "Генерация точек коллокации с классификацией границ"
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
            "description": "Давление на входе/выходе, скорость на входе"
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
    
    # Очищаем экран при запуске
    clear_screen()
    
    # Показываем баннер
    print_banner()
    
    # Основной цикл программы
    while True:
        print_main_menu()
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}Выберите действие (0-4):{Colors.RESET} ", end="")
        
        try:
            choice: str = input().strip()

            if choice == "0":
                print(f"\n{Colors.GREEN}👋 Программа завершена. До свидания!{Colors.RESET}")
                break
                
            elif choice == "1":
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
                print(f"\n{Colors.CYAN}Нажмите Enter чтобы продолжить...{Colors.RESET}", end="")
                input()
                clear_screen()
                print_banner()
                
            elif choice == "4":
                clear_screen()
                print_banner()
                
            else:
                print(f"{Colors.RED}❌ Неверный номер. Введите число от 0 до 4{Colors.RESET}")
                print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
                input()
                clear_screen()
                print_banner()
                
        except ValueError:
            print(f"{Colors.RED}❌ Ошибка: введите число!{Colors.RESET}")
            print(f"{Colors.CYAN}Нажмите Enter...{Colors.RESET}", end="")
            input()
            clear_screen()
            print_banner()
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
            print(f"{Colors.GREEN}👋 До свидания!{Colors.RESET}")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Программа прервана пользователем{Colors.RESET}")
        print(f"{Colors.GREEN}👋 До свидания!{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Критическая ошибка: {e}{Colors.RESET}")
        sys.exit(1)