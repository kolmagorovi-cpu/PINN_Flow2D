#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузчик CFD-экспорта (Fluent/CFX) в массивы numpy.

Поддерживается формат с секциями::

    [Name]
    Plane 1

    [Data]
    X [ m ], Y [ m ], Z [ m ], Pressure [ Pa ], Velocity [ m s^-1 ],
    Velocity u [ m s^-1 ], Velocity v [ m s^-1 ]
    <числовые строки>

а также обычный CSV с одной строкой заголовка (алиасы в ``constants.py``).

Заголовки сопоставляются по имени, а не по позиции: ``Velocity u`` не путается
с ``Velocity``, потому что матчинг идёт по самому длинному алиасу.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Optional, Sequence

import numpy as np

from .constants import (
    COLUMN_ALIASES,
    DEFAULT_DELIMITER,
    REQUIRED_COLUMNS,
    UnsetColumnError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RawCFD",
    "clean_header_token",
    "parse_cfd_export",
    "load_cfd_export",
    "iter_cfd_files",
]

_UNIT_RE = re.compile(r"\[[^\]]*\]")
_SECTION_RE = re.compile(r"^\[[^\]]+\]$")


def clean_header_token(token: str) -> str:
    """Убирает единицы в квадратных скобках и нормализует регистр/пробелы."""
    text = _UNIT_RE.sub(" ", str(token))
    return " ".join(text.strip().lower().split())


def _build_alias_map() -> dict[str, str]:
    pairs = [
        (clean_header_token(alias), canonical)
        for canonical, aliases in COLUMN_ALIASES.items()
        for alias in aliases
    ]
    # Самые длинные алиасы имеют приоритет ("velocity u" раньше "velocity").
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return dict(pairs)


_ALIAS_TO_CANONICAL = _build_alias_map()


@dataclass
class RawCFD:
    """Сырые (физические) поля одного CFD-экспорта.

    Координаты и скорости — в метрах и м/с, давление — в паскалях
    (абсолютное или избыточное; фиксируется в манифесте через
    ``static_pressure``).
    """

    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    u: np.ndarray
    v: np.ndarray
    z: Optional[np.ndarray] = None
    u_mag: Optional[np.ndarray] = None
    source: str = ""
    skipped_rows: int = 0
    n_points: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for name in ("x", "y", "p", "u", "v"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.size == 0:
                raise ValueError(f"RawCFD.{name}: пустой массив")
            if not np.isfinite(array).all():
                raise ValueError(f"RawCFD.{name}: обнаружены NaN/Inf")
            setattr(self, name, array)

        n = int(self.x.size)
        for name in ("y", "p", "u", "v"):
            if int(getattr(self, name).size) != n:
                raise ValueError(
                    f"RawCFD.{name}: длина {getattr(self, name).size} "
                    f"не совпадает с x ({n})"
                )

        for name in ("z", "u_mag"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size != n:
                raise ValueError(f"RawCFD.{name}: длина {array.size} != {n}")
            setattr(self, name, array)

        self.n_points = n


def _locate_header(lines: Sequence[str]) -> int:
    """Индекс строки заголовков (первая текстовая строка после ``[Data]``)."""
    start = 0
    for index, line in enumerate(lines):
        if line.strip().lower() == "[data]":
            start = index + 1
            break

    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if not stripped or _SECTION_RE.match(stripped):
            continue
        if any(char.isalpha() for char in stripped):
            return index
    raise ValueError("не найдена строка заголовков CFD-экспорта")


def _map_columns(tokens: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, token in enumerate(tokens):
        canonical = _ALIAS_TO_CANONICAL.get(token)
        if canonical is not None and canonical not in mapping:
            mapping[canonical] = index
    return mapping


def parse_cfd_export(
    text: str,
    delimiter: str = DEFAULT_DELIMITER,
) -> RawCFD:
    """Разбирает текст CFD-экспорта (секции Fluent/CFX или плоский CSV)."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("пустой CFD-файл")

    header_index = _locate_header(lines)
    header_tokens = [
        clean_header_token(token)
        for token in lines[header_index].split(delimiter)
    ]
    columns = _map_columns(header_tokens)

    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise UnsetColumnError(
            "в заголовке CFD-экспорта нет колонок "
            f"{missing}; найденные колонки: {header_tokens}"
        )

    n_columns = len(header_tokens)
    rows: list[list[float]] = []
    skipped = 0
    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped or _SECTION_RE.match(stripped):
            continue
        parts = [part.strip() for part in stripped.split(delimiter)]
        if len(parts) != n_columns:
            skipped += 1
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            skipped += 1

    if not rows:
        raise ValueError("в CFD-файле не найдено ни одной числовой строки")
    if skipped:
        logger.warning("CFD: пропущено %d нечисловых/битых строк", skipped)

    data = np.asarray(rows, dtype=np.float64)
    optional: dict[str, np.ndarray] = {}
    if "z" in columns:
        optional["z"] = data[:, columns["z"]]
    if "velocity" in columns:
        optional["u_mag"] = data[:, columns["velocity"]]

    return RawCFD(
        x=data[:, columns["x"]],
        y=data[:, columns["y"]],
        p=data[:, columns["pressure"]],
        u=data[:, columns["u"]],
        v=data[:, columns["v"]],
        skipped_rows=skipped,
        **optional,
    )


def load_cfd_export(
    path: str | os.PathLike[str],
    delimiter: str = DEFAULT_DELIMITER,
) -> RawCFD:
    """Читает CFD-экспорт с диска."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CFD-файл не найден: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    raw = parse_cfd_export(text, delimiter=delimiter)
    raw.source = os.fspath(path)
    logger.info(
        "CFD: загружено %d точек из %s (пропущено %d строк)",
        raw.n_points, raw.source, raw.skipped_rows,
    )
    return raw


def iter_cfd_files(
    directory: str | os.PathLike[str],
    pattern: str = "*.csv",
) -> list[str]:
    """Возвращает отсортированный список CFD-файлов в каталоге."""
    import glob

    return sorted(glob.glob(os.path.join(os.fspath(directory), pattern)))

