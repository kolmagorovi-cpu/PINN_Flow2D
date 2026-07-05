#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск обучения PINN из консоли и пакетных сценариев.

Этот модуль отделяет создание ``PINNSolver`` и проверку входных данных от
интерактивных меню ``learning_cli``. Меню собирает конфигурацию, а runner
получает уже готовый ``ConfigManager`` или YAML-файл и отвечает только за
старт обучения.
"""

import logging
import os
import sys
from typing import Optional

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.solver import PINNSolver
from airfoil_pinn.utils import clear_screen

logger = logging.getLogger(__name__)

DEFAULT_POINTS_FILE = os.path.join("Initial", "Data", "initial.npy")

__all__ = ["DEFAULT_POINTS_FILE", "run_experiment", "start_training"]


def run_experiment(config_path: str, experiment_dir: str) -> None:
    """
    Запускает обучение по YAML-конфигурации без интерактивного меню.

    Args:
        config_path: путь к YAML-файлу конфигурации.
        experiment_dir: директория для сохранения результатов эксперимента.
    """
    cfg = ConfigManager()
    try:
        # Пакетный режим всегда читает конфигурацию из указанного файла.
        cfg.load_from_file(config_path)
    except Exception as exc:
        logger.error("Не удалось загрузить конфигурацию %s: %s", config_path, exc)
        sys.exit(1)

    points_file = cfg.config["data"].get("points_file")
    if not points_file or not os.path.exists(points_file):
        logger.error("Файл с точками не найден: %s", points_file)
        sys.exit(1)

    solver = PINNSolver(cfg.config, experiment_dir=experiment_dir)
    solver.train()


def start_training(
    cfg: ConfigManager,
    experiment_dir: Optional[str] = None,
) -> None:
    """
    Запускает обучение с текущей конфигурацией.

    Args:
        cfg: менеджер с загруженной или собранной конфигурацией.
        experiment_dir: если указан, результаты пишутся в эту директорию.
    """
    if not cfg.config["data"].get("points_file"):
        cfg.config["data"]["points_file"] = DEFAULT_POINTS_FILE
        print(f"Используется путь к точкам по умолчанию: {DEFAULT_POINTS_FILE}")

    if not os.path.exists(cfg.config["data"]["points_file"]):
        print(f"Файл с точками не найден: {cfg.config['data']['points_file']}")
        alt = input(
            "Введите путь к файлу вручную (или Enter для отмены): "
        ).strip()
        if alt and os.path.exists(alt):
            cfg.config["data"]["points_file"] = alt
        else:
            print("Файл не найден. Обучение невозможно.")
            input("Нажмите Enter...")
            return

    # Старые конфигурации могли быть сохранены до появления этого раздела.
    if "boundary_conditions" not in cfg.config:
        cfg.config["boundary_conditions"] = {"type": "soft"}
        logger.warning(
            "В конфигурации отсутствует раздел 'boundary_conditions', "
            "установлен 'soft' по умолчанию."
        )

    if experiment_dir is None:
        # В интерактивном режиме пользователь видит итоговую конфигурацию.
        clear_screen()
        print("=" * 60)
        print("ЗАПУСК ОБУЧЕНИЯ".center(60))
        print("=" * 60)
        cfg.show()
        print("\nНажмите Enter для начала обучения или Ctrl+C для отмены...")
        input()

    solver = PINNSolver(cfg.config, experiment_dir=experiment_dir)
    solver.train()
