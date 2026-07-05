#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запуск консольного приложения командой ``python -m airfoil_pinn``."""

import sys

from airfoil_pinn.app.runtime import bootstrap_environment


def run() -> int:
    """
    Подготавливает окружение и передает управление консольному интерфейсу.

    Эта функция отделяет техническую подготовку запуска от меню. Благодаря
    этому старый скрипт ``Airfoil_PINN.py`` и новый пакетный запуск используют
    один и тот же код.
    """
    bootstrap_environment()
    # CLI импортируется после bootstrap: модуль меню использует старый пакет
    # utils, который может подтянуть тяжелые зависимости вроде torch.
    from airfoil_pinn.app.cli import main

    main()
    return 0


if __name__ == "__main__":
    sys.exit(run())
