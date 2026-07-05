#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль пакетного запуска экспериментов для проекта PINN Learning.
Позволяет запускать серию экспериментов с различными конфигурациями,
сохранять результаты в структурированном виде и собирать сводные метрики.

Автор: Колмагоров И.Д.
Версия: 3.260226
"""

import os
import sys
import yaml
import argparse
import subprocess
import time
import logging
import csv
from typing import List, Dict, Any, Optional
from itertools import product

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Константы
DEFAULT_BASE_CONFIG = "Configs/base.yaml"
DEFAULT_EXPERIMENTS_ROOT = "experiments"
SUMMARY_CSV = "summary.csv"


def load_base_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает базовую конфигурацию из YAML-файла.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Базовая конфигурация не найдена: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Базовая конфигурация загружена из {config_path}")
    return config


def load_experiment_list(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Загружает список экспериментов из YAML-файла.
    Ожидается структура:
        base_config: "path/to/base.yaml"
        experiments:
          - name: "exp_001"
            overrides:
              key1: value1
              key2: value2
          ...
    Возвращает список словарей с ключами 'name' и 'overrides'.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    base_path = data.get('base_config', DEFAULT_BASE_CONFIG)
    experiments = data.get('experiments', [])
    logger.info(f"Загружено {len(experiments)} экспериментов из {yaml_path}")
    return base_path, experiments


def generate_experiments_from_params(base_config: Dict[str, Any],
                                     param_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Генерирует список экспериментов на основе сетки параметров.
    param_lists: словарь вида {'path.to.param': [values...]}
    Возвращает список словарей с полем 'overrides' для каждой комбинации.
    """
    keys = list(param_lists.keys())
    value_combinations = product(*param_lists.values())
    experiments = []
    for idx, combo in enumerate(value_combinations, 1):
        overrides = {}
        for key, value in zip(keys, combo):
            # поддержка точечной нотации: разбиваем по '.' и создаём вложенный словарь
            parts = key.split('.')
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        experiments.append({
            'name': f"exp_{idx:03d}",
            'overrides': overrides
        })
    logger.info(f"Сгенерировано {len(experiments)} экспериментов из сетки параметров")
    return experiments


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Рекурсивно обновляет словарь base значениями из overrides.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def create_experiment_dir(exp_name: str, root_dir: str = DEFAULT_EXPERIMENTS_ROOT) -> str:
    """
    Создаёт директорию для эксперимента и возвращает её путь.
    """
    exp_dir = os.path.join(root_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Создана директория эксперимента: {exp_dir}")
    return exp_dir


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Сохраняет конфигурацию в YAML-файл.
    """
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Конфигурация сохранена: {path}")


def run_experiment(config_path: str, exp_dir: str) -> bool:
    """
    Запускает обучение для одного эксперимента через подпроцесс.
    Возвращает True, если процесс завершился успешно (код 0).
    """
    cmd = [
        sys.executable, "-m", "airfoil_pinn.training.learning_cli",
        "--config", config_path,
        "--experiment_dir", exp_dir
    ]
    logger.info(f"Запуск эксперимента в {exp_dir}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Эксперимент в {exp_dir} завершён успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при выполнении эксперимента в {exp_dir}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def collect_metrics(exp_dir: str, summary_path: str, exp_name: str) -> None:
    """
    Собирает метрики из файла metrics.csv в директории эксперимента
    и добавляет их в общий сводный CSV-файл.
    """
    metrics_file = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        logger.warning(f"Файл метрик не найден: {metrics_file}")
        return

    # Читаем метрики (предполагается одна строка с заголовком)
    with open(metrics_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            logger.warning(f"Файл метрик пуст: {metrics_file}")
            return
        metrics_row = rows[0]  # берём первую строку

    # Добавляем колонку с именем эксперимента
    metrics_row['experiment'] = exp_name

    # Запись в сводный CSV
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)
    logger.info(f"Метрики эксперимента {exp_name} добавлены в {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Пакетный запуск экспериментов PINN")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="YAML-файл со списком экспериментов")
    group.add_argument("--grid", type=str, help="YAML-файл с сеткой параметров (будет сгенерирована комбинация)")

    parser.add_argument("--base", type=str, default=DEFAULT_BASE_CONFIG,
                        help="Базовая конфигурация (используется с --grid)")
    parser.add_argument("--root", type=str, default=DEFAULT_EXPERIMENTS_ROOT,
                        help="Корневая директория для экспериментов")
    parser.add_argument("--summary", type=str, default=SUMMARY_CSV,
                        help="Имя сводного CSV-файла (будет создан в корне экспериментов)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Количество одновременно запускаемых экспериментов (по умолчанию 1)")

    args = parser.parse_args()

    # Создаём корневую директорию для экспериментов
    os.makedirs(args.root, exist_ok=True)
    summary_path = os.path.join(args.root, args.summary)

    # Загружаем базовую конфигурацию
    base_config = load_base_config(args.base)

    experiments = []

    if args.config:
        # Загружаем список экспериментов из файла
        base_path, exp_list = load_experiment_list(args.config)
        if base_path != args.base:
            # Если в файле указана другая базовая конфигурация, перезагружаем
            base_config = load_base_config(base_path)
        for exp in exp_list:
            name = exp.get('name', 'unnamed')
            overrides = exp.get('overrides', {})
            experiments.append((name, overrides))
    elif args.grid:
        # Загружаем файл с сеткой параметров
        with open(args.grid, 'r', encoding='utf-8') as f:
            grid_params = yaml.safe_load(f)
        # Ожидается словарь вида { 'param.path': [values] }
        exp_list = generate_experiments_from_params(base_config, grid_params)
        experiments = [(exp['name'], exp['overrides']) for exp in exp_list]
    else:
        parser.error("Не указан источник экспериментов (--config или --grid)")

    if not experiments:
        logger.error("Нет экспериментов для запуска")
        sys.exit(1)

    logger.info(f"Всего экспериментов для запуска: {len(experiments)}")

    # Здесь можно реализовать параллельный запуск через пул процессов,
    # но для простоты реализуем последовательный.
    # Для параллельного запуска потребуется более сложная логика управления процессами.
    # Оставим задел, но пока последовательно.

    successful = 0
    failed = 0

    for name, overrides in experiments:
        logger.info(f"Обработка эксперимента: {name}")
        exp_dir = create_experiment_dir(name, args.root)

        # Модифицируем конфигурацию
        exp_config = deep_update(base_config.copy(), overrides)
        exp_config['project_name'] = name  # переопределяем имя проекта

        # Сохраняем конфигурацию эксперимента
        config_path = os.path.join(exp_dir, "config.yaml")
        save_config(exp_config, config_path)

        # Запускаем эксперимент
        success = run_experiment(config_path, exp_dir)
        if success:
            successful += 1
            # Собираем метрики
            collect_metrics(exp_dir, summary_path, name)
        else:
            failed += 1

        # Небольшая пауза между запусками (опционально)
        time.sleep(2)

    logger.info(f"Завершено. Успешно: {successful}, с ошибками: {failed}")


if __name__ == "__main__":
    main()
