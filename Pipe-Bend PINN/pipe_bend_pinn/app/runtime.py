#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка процесса перед запуском консольного приложения.

Здесь собрана логика, которая раньше находилась в начале ``pipe_bend_main.py``:
настройка UTF-8 вывода в Windows, добавление корня проекта в ``sys.path`` и
автоматический переход в локальное виртуальное окружение.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ``runtime.py`` лежит в pipe_bend_pinn/app, поэтому parents[2] указывает на
# корень репозитория.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Предпочитаем общее виртуальное окружение родительского каталога (PINN_Flow2D/.venv),
# иначе используем локальное Pipe-Bend PINN/venv.
_SHARED_VENV_DIR = PROJECT_ROOT.parent / ".venv"
_LOCAL_VENV_DIR = PROJECT_ROOT / "venv"
VENV_DIR = _SHARED_VENV_DIR if _SHARED_VENV_DIR.exists() else _LOCAL_VENV_DIR

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("yaml", "pyyaml"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("shapely", "shapely"),
    ("typing_extensions", "typing_extensions"),
    ("keyboard", "keyboard"),
)


def configure_text_output() -> None:
    """Настраивает UTF-8 вывод для консольного интерфейса."""
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
    """Добавляет корень проекта в ``sys.path``."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _venv_python() -> Path:
    """Возвращает путь к Python внутри локального виртуального окружения."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    """Возвращает путь к pip внутри локального виртуального окружения."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def _ask(prompt: str) -> bool:
    """Запрашивает подтверждение у пользователя в интерактивной консоли."""
    try:
        return input(prompt).strip().lower() in ("y", "yes", "да", "д")
    except (EOFError, KeyboardInterrupt):
        return False


def _create_venv() -> bool:
    """Создает локальное виртуальное окружение."""
    uv_path = shutil.which("uv")
    print("  Создание виртуального окружения...", flush=True)

    if uv_path:
        result = subprocess.run(
            [uv_path, "venv", str(VENV_DIR)],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        print(f"  Ошибка: {result.stderr.strip() or result.stdout.strip()}")
        return False

    print("  Виртуальное окружение создано.")
    return True


def _pip_environment() -> dict[str, str]:
    """Возвращает окружение для pip без прокси-переменных."""
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def _ensure_pip() -> bool:
    """Инициализирует pip внутри venv, если он еще не создан."""
    venv_pip = _venv_pip()
    if venv_pip.exists():
        return True

    result = subprocess.run(
        [str(_venv_python()), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _install_packages(pip_names: list[str]) -> bool:
    """Устанавливает недостающие зависимости в локальное окружение."""
    print(f"  Установка: {', '.join(pip_names)} ...", flush=True)
    if not _ensure_pip():
        print("  Не удалось инициализировать pip в окружении.")
        return False

    result = subprocess.run(
        [str(_venv_python()), "-m", "pip", "install", *pip_names],
        env=_pip_environment(),
        check=False,
    )
    if result.returncode != 0:
        print("  Ошибка при установке. Проверьте подключение к интернету.")
        return False

    print("  Зависимости установлены.")
    return True


def _restart_inside_venv() -> None:
    """Перезапускает текущую команду под Python из локального venv."""
    package_main = PROJECT_ROOT / "pipe_bend_pinn" / "__main__.py"
    argv0 = Path(sys.argv[0]).resolve()

    if argv0 == package_main.resolve():
        command = [str(_venv_python()), "-m", "pipe_bend_pinn", *sys.argv[1:]]
    else:
        command = [str(_venv_python()), *sys.argv]

    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    sys.exit(result.returncode)


def ensure_virtual_environment() -> None:
    """Проверяет локальное виртуальное окружение и зависимости."""
    venv_python = _venv_python()
    in_our_venv = (
        venv_python.exists()
        and Path(sys.executable).resolve() == venv_python.resolve()
    )

    if not VENV_DIR.exists():
        print("\n  Виртуальное окружение не обнаружено.")
        if not _ask("  Создать и установить зависимости? [y/N]: "):
            print("  Продолжение без виртуального окружения (возможны ошибки).\n")
            return
        if not _create_venv():
            return
        _install_packages([pip_name for _, pip_name in REQUIRED_PACKAGES])
        if venv_python.exists():
            print("  Перезапуск в виртуальном окружении...\n", flush=True)
            _restart_inside_venv()
        return

    if not in_our_venv and venv_python.exists():
        _restart_inside_venv()

    missing_pip = [
        pip_name
        for import_name, pip_name in REQUIRED_PACKAGES
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing_pip:
        return

    print(f"\n  Отсутствуют пакеты: {', '.join(missing_pip)}")
    if _ask("  Установить недостающие пакеты? [y/N]: "):
        if _install_packages(missing_pip):
            print("  Перезапуск...\n", flush=True)
            _restart_inside_venv()
    else:
        print("  Продолжение с отсутствующими пакетами (возможны ошибки).\n")


def bootstrap_environment() -> None:
    """Выполняет все подготовительные шаги перед запуском CLI."""
    configure_text_output()
    ensure_project_root_on_path()
    from pipe_bend_pinn.app.paths import ensure_workspace_dirs

    ensure_workspace_dirs()
    ensure_virtual_environment()