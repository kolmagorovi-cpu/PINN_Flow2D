#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запуск консольного приложения командой ``python -m pipe_bend_pinn``."""

import sys

from pipe_bend_pinn.app.runtime import bootstrap_environment


def run() -> int:
    """
    Подготавливает окружение и передает управление консольному интерфейсу.
    """
    bootstrap_environment()
    from pipe_bend_pinn.app.cli import guarded_main

    return guarded_main()


if __name__ == "__main__":
    sys.exit(run())