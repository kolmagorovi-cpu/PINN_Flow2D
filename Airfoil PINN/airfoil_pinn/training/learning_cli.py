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

# Импорты ниже оставлены как совместимые реэкспорты для PINN_Learning.py и внешних скриптов.
from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.menus.architecture import (
    _configure_activation,
    _configure_coordinate_scaling,
    _configure_hidden_layers,
    _configure_input_params,
    _configure_output_params,
    architecture_menu,
)
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.training.menus.configuration import (
    create_new_config,
    edit_config,
    load_config,
    main_menu,
    save_config,
)
from airfoil_pinn.training.menus.loss import (
    loss_function_menu,
    weights_menu,
)
from airfoil_pinn.training.menus.process import process_settings
from airfoil_pinn.training.menus.results import (
    checkpoint_menu,
    diagnostics_menu,
    output_menu,
)
from airfoil_pinn.training.menus.strategy import training_strategy_menu
from airfoil_pinn.training.runner import (
    DEFAULT_POINTS_FILE,
    run_experiment,
    start_training,
)
from airfoil_pinn.utils import clear_screen

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Константы
CONFIG_DIR = ConfigManager.CONFIG_DIR


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
        "--experiment_dir",
        type=str,
        help="Директория для сохранения результатов эксперимента",
    )
    args = parser.parse_args()

    if args.config:
        # Пакетный режим: запускаем эксперимент
        if not args.experiment_dir:
            logger.error(
                "При использовании --config необходимо указать --experiment_dir"
            )
            sys.exit(1)
        run_experiment(args.config, args.experiment_dir)
    else:
        # Интерактивный режим
        main_menu()


if __name__ == "__main__":
    main()
