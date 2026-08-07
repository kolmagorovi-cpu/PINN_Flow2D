#!/usr/bin/env python3
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
