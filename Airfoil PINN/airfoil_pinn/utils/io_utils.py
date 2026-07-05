#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для ввода-вывода и работы с файлами.

Этот модуль содержит общие функции для сохранения и загрузки данных,
устраняя дублирование кода в различных модулях проекта.

Example:
    >>> from airfoil_pinn.utils import save_json, load_json, sanitize_filename
    >>> data = {"key": "value"}
    >>> path = save_json(data, "output", "config")
    >>> loaded = load_json("output", "config.json")
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from airfoil_pinn.core.constants import (
    DEFAULT_ENCODING,
    JSON_INDENT,
    FILE_EXT_JSON,
)


def sanitize_filename(name: str) -> str:
    """
    Очищает имя для использования в качестве имени файла.

    Заменяет пробелы и специальные символы на подчёркивания.

    Args:
        name: Исходное имя

    Returns:
        Безопасное имя файла

    Example:
        >>> sanitize_filename("My Fluid Name")
        'my_fluid_name'
        >>> sanitize_filename("Profile NACA 0012")
        'profile_naca_0012'
    """
    # Замена пробелов и специальных символов
    sanitized = name.strip().lower()
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.replace("/", "_")
    sanitized = sanitized.replace("\\", "_")
    sanitized = sanitized.replace(".", "_")
    sanitized = sanitized.replace(",", "_")
    sanitized = sanitized.replace(";", "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace("(", "_")
    sanitized = sanitized.replace(")", "_")
    sanitized = sanitized.replace("[", "_")
    sanitized = sanitized.replace("]", "_")
    sanitized = sanitized.replace("{", "_")
    sanitized = sanitized.replace("}", "_")

    # Удаление повторяющихся подчёркиваний
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    return sanitized


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Создаёт директорию если она не существует.

    Args:
        directory: Путь к директории

    Returns:
        Path объект созданной/существующей директории

    Example:
        >>> data_dir = ensure_directory("data/results")
        >>> ensure_directory("/path/to/output")
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(
    data: Dict[str, Any],
    directory: Union[str, Path],
    filename: str,
    sanitize: bool = True,
    indent: int = JSON_INDENT,
    encoding: str = DEFAULT_ENCODING
) -> str:
    """
    Сохраняет данные в JSON файл.

    Args:
        data: Данные для сохранения (словарь)
        directory: Директория для сохранения
        filename: Имя файла (без расширения или с .json)
        sanitize: Очищать ли имя файла от специальных символов
        indent: Отступ для форматирования
        encoding: Кодировка файла

    Returns:
        Полный путь к сохранённому файлу

    Raises:
        IOError: Если не удалось записать файл

    Example:
        >>> data = {"name": "fluid", "viscosity": 0.001}
        >>> path = save_json(data, "data/fluids", "water")
        >>> path = save_json(data, "configs", "config.yaml", sanitize=False)
    """
    dir_path = ensure_directory(directory)

    # Обработка имени файла
    if sanitize:
        # Удаляем расширение если есть
        if filename.lower().endswith(FILE_EXT_JSON):
            filename = filename[:-len(FILE_EXT_JSON)]
        filename = sanitize_filename(filename) + FILE_EXT_JSON

    file_path = dir_path / filename

    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return str(file_path)
    except IOError as e:
        raise IOError(f"Не удалось сохранить файл {file_path}: {e}")


def load_json(
    directory: Union[str, Path],
    filename: str,
    encoding: str = DEFAULT_ENCODING
) -> Dict[str, Any]:
    """
    Загружает данные из JSON файла.

    Args:
        directory: Директория с файлом
        filename: Имя файла
        encoding: Кодировка файла

    Returns:
        Загруженные данные

    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл содержит некорректный JSON

    Example:
        >>> data = load_json("data/fluids", "water.json")
        >>> config = load_json("configs", "config.yaml")
    """
    dir_path = Path(directory)
    file_path = dir_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Некорректный JSON в файле {file_path}",
            e.doc,
            e.pos
        )


def save_text(
    content: str,
    filepath: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    ensure_dir: bool = True
) -> str:
    """
    Сохраняет текстовое содержимое в файл.

    Args:
        content: Текст для сохранения
        filepath: Путь к файлу
        encoding: Кодировка файла
        ensure_dir: Создать директорию если не существует

    Returns:
        Полный путь к сохранённому файлу

    Example:
        >>> save_text("Hello, World!", "output.txt")
        >>> save_text(config_str, "config/config.yaml", ensure_dir=True)
    """
    path = Path(filepath)

    if ensure_dir:
        ensure_directory(path.parent)

    with open(path, 'w', encoding=encoding) as f:
        f.write(content)

    return str(path)


def load_text(
    filepath: Union[str, Path],
    encoding: str = DEFAULT_ENCODING
) -> str:
    """
    Загружает текстовое содержимое из файла.

    Args:
        filepath: Путь к файлу
        encoding: Кодировка файла

    Returns:
        Содержимое файла

    Raises:
        FileNotFoundError: Если файл не найден
        IOError: Если не удалось прочитать файл

    Example:
        >>> content = load_text("README.md")
        >>> config = load_text("config/config.yaml")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def file_exists(
    directory: Union[str, Path],
    filename: str,
    extension: Optional[str] = None
) -> bool:
    """
    Проверяет существование файла.

    Args:
        directory: Директория для проверки
        filename: Имя файла
        extension: Расширение (опционально, добавляется если нет в filename)

    Returns:
        True если файл существует

    Example:
        >>> file_exists("data/fluids", "water")
        >>> file_exists("configs", "config", extension=".yaml")
    """
    dir_path = Path(directory)

    if extension and not filename.endswith(extension):
        filename = filename + extension

    file_path = dir_path / filename
    return file_path.exists()


def get_file_stem(filepath: Union[str, Path]) -> str:
    """
    Возвращает имя файла без расширения.

    Args:
        filepath: Путь к файлу

    Returns:
        Имя файла без расширения

    Example:
        >>> get_file_stem("data/fluids/water.json")
        'water'
        >>> get_file_stem("/path/to/config.yaml")
        'config'
    """
    return Path(filepath).stem


def list_files(
    directory: Union[str, Path],
    extension: Optional[str] = None,
    recursive: bool = False
) -> list:
    """
    Возвращает список файлов в директории.

    Args:
        directory: Директория для поиска
        extension: Фильтр по расширению (например, '.json')
        recursive: Искать рекурсивно во всех поддиректориях

    Returns:
        Список путей к файлам

    Example:
        >>> list_files("data", extension=".json")
        >>> list_files("configs", recursive=True)
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        return []

    if recursive:
        pattern = f"**/*{extension}" if extension else "**/*"
    else:
        pattern = f"*{extension}" if extension else "*"

    return [str(p) for p in dir_path.glob(pattern) if p.is_file()]
