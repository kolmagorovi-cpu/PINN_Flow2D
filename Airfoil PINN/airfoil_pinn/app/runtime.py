#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка процесса перед запуском консольного приложения.

Здесь собрана логика, которая раньше находилась в начале ``Airfoil_PINN.py``:
настройка UTF-8 вывода в Windows, добавление корня проекта в ``sys.path`` и
переход в общее виртуальное окружение из ``D:\\Python``. Модуль не содержит
меню и расчетной логики, поэтому его можно безопасно переиспользовать из
нескольких точек входа.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


# ``runtime.py`` лежит в airfoil_pinn/app, поэтому parents[2] указывает на
# корень репозитория. Этот путь нужен для запуска через Airfoil_PINN.py и для
# корректного разрешения относительных путей из workspace-конфигов.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON_ENV_DIR = Path("D:/Python/BaseEnv")


def _select_python_environment() -> Path:
    """Prefer an explicit override, then D:/Python/BaseEnv, then local venv."""
    override = os.environ.get("AIRFOIL_PINN_PYTHON_ENV")
    if override:
        return Path(override).expanduser()
    executable = DEFAULT_PYTHON_ENV_DIR / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if executable.is_file():
        return DEFAULT_PYTHON_ENV_DIR
    return PROJECT_ROOT / "venv"


PYTHON_ENV_DIR = _select_python_environment()
# Compatibility name for existing callers.
VENV_DIR = PYTHON_ENV_DIR

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("yaml", "pyyaml"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("shapely", "shapely"),
    ("typing_extensions", "typing_extensions"),
)


def configure_text_output() -> None:
    """
    Настраивает UTF-8 вывод для консольного интерфейса.

    Консольные модули активно используют русские сообщения и символы статуса.
    На Windows без явной настройки кодовой страницы такие сообщения могут
    отображаться некорректно или приводить к ошибкам кодирования.
    """
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if sys.stdout.isatty():
            os.system("chcp 65001 > nul")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def ensure_project_root_on_path() -> None:
    """
    Добавляет корень проекта в ``sys.path``.

    Консольная точка входа может запускаться из разных рабочих директорий.
    Явный корень в ``sys.path`` гарантирует импорт пакета ``airfoil_pinn`` и
    корректную работу относительных путей внутри ``workspace``.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _venv_python() -> Path:
    """Возвращает Python из общего, заранее подготовленного окружения."""
    if sys.platform == "win32":
        return PYTHON_ENV_DIR / "Scripts" / "python.exe"
    return PYTHON_ENV_DIR / "bin" / "python"


def _restart_inside_venv() -> None:
    """
    Перезапускает текущую команду под Python из общего окружения.

    ``sys.argv`` сохраняет исходный способ запуска: старый
    ``python Airfoil_PINN.py`` и новый ``python -m airfoil_pinn`` продолжают
    работать одинаково.
    """
    package_main = PROJECT_ROOT / "airfoil_pinn" / "__main__.py"
    argv0 = Path(sys.argv[0]).resolve()

    if argv0 == package_main.resolve():
        # При запуске ``python -m airfoil_pinn`` Python передает в sys.argv[0]
        # путь к __main__.py. Если выполнить этот файл напрямую, пакетный
        # импорт ``airfoil_pinn`` может потерять корень проекта в sys.path.
        command = [str(_venv_python()), "-m", "airfoil_pinn", *sys.argv[1:]]
    else:
        command = [str(_venv_python()), *sys.argv]

    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    sys.exit(result.returncode)


def ensure_virtual_environment() -> None:
    """
    Использует общее окружение или подготавливает локальный venv.

    В локальном venv автоматически устанавливаются отсутствующие зависимости.
    Общее и явно указанные внешние окружения автоматически не изменяются.
    """
    venv_python = _venv_python()
    is_local = PYTHON_ENV_DIR.resolve() == (PROJECT_ROOT / "venv").resolve()
    if not venv_python.is_file() and is_local:
        print(f"  Создание локального окружения: {PYTHON_ENV_DIR}", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(PYTHON_ENV_DIR)], check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"  Не удалось создать venv: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    in_our_venv = (
        venv_python.exists()
        and Path(sys.executable).resolve() == venv_python.resolve()
    )

    if not venv_python.is_file():
        print(
            "\n  ОШИБКА: подготовленное окружение Python не найдено:\n"
            f"  {PYTHON_ENV_DIR}\n"
            "  Автоматическое создание окружения и установка пакетов отключены.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not in_our_venv and venv_python.exists():
        _restart_inside_venv()

    missing_pip = [
        pip_name
        for import_name, pip_name in REQUIRED_PACKAGES
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing_pip:
        return

    if is_local:
        print(f"  Установка зависимостей в venv: {', '.join(missing_pip)}", flush=True)
        try:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", *missing_pip], check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"  Не удалось установить зависимости: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _restart_inside_venv()
        return

    print(
        "\n  ОШИБКА: в подготовленном окружении отсутствуют пакеты: "
        f"{', '.join(missing_pip)}\n"
        f"  Окружение: {PYTHON_ENV_DIR}\n"
        "  Проект не будет устанавливать или обновлять их автоматически.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def bootstrap_environment() -> None:
    """
    Выполняет все подготовительные шаги перед запуском CLI.

    Порядок важен: сначала настраивается вывод, затем путь импорта, затем
    общее окружение. После возможного перезапуска управление вернется
    уже в корректно настроенный процесс.
    """
    configure_text_output()
    ensure_project_root_on_path()
    from airfoil_pinn.app.paths import ensure_workspace_dirs

    ensure_workspace_dirs()
    ensure_virtual_environment()
