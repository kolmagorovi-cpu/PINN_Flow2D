#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консольная точка входа обучения PINN.

Модуль оставляет в одном месте только разбор аргументов командной строки и
выбор режима запуска. Интерактивные экраны перенесены в ``airfoil_pinn.training.menus``,
а запуск обучения - в ``airfoil_pinn.training.runner``.
"""

import argparse
import logging
import sys

from airfoil_pinn.training.menus.configuration import main_menu
from airfoil_pinn.training.runner import run_experiment

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """
    Обрабатывает аргументы командной строки.

    Без ``--config`` запускает интерактивное консольное меню. С ``--config``
    работает в пакетном режиме и передаёт управление runner-слою.
    """
    parser = argparse.ArgumentParser(description="PINN Learning - главный модуль")
    parser.add_argument(
        "--config",
        type=str,
        help="Путь к файлу конфигурации для запуска эксперимента (без меню)",
    )
    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dir",
        type=str,
        help="Директория для сохранения результатов эксперимента",
    )
    args = parser.parse_args()

    if args.config:
        # Пакетный режим: запускаем эксперимент
        if not args.experiment_dir:
            logger.error(
                "При использовании --config необходимо указать --experiment-dir"
            )
            sys.exit(1)
        run_experiment(args.config, args.experiment_dir)
    else:
        # Интерактивный режим
        main_menu()


if __name__ == "__main__":
    main()
