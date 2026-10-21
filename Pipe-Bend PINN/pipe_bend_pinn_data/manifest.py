#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Манифест CFD-кейсов: сопоставление CSV -> входная скорость и масштабы.

Поддерживаются YAML и JSON. Минимальный пример::

    ref_scales:
      U_ref: 1.0
      pipe_width: 0.5
      density: 998.0
      dynamic_viscosity: 0.998
      static_pressure: 100000.0
    geometry:
      domain_file: workspace/geometry/domains/pipe_bend_domain.npy
    cases:
      - file: workspace/cfd/export.csv
        U_in: 1.0
        name: Re500

Краткая форма кейсов допускает отображение ``{file: U_in}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .normalization import ReferenceScales

logger = logging.getLogger(__name__)

# Корень «Pipe-Bend PINN»: родитель каталога пакета. Используется как
# fallback-корень для project-relative путей в манифесте.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

__all__ = ["CFDCase", "GeometrySpec", "CFDManifest", "load_manifest"]


@dataclass
class CFDCase:
    """Один CFD-кейс (файл + соответствующая входная скорость)."""

    file: str
    U_in: float
    name: str = ""
    delimiter: str = ","

    def __post_init__(self) -> None:
        self.file = os.fspath(self.file)
        self.U_in = float(self.U_in)
        if not np.isfinite(self.U_in) or self.U_in <= 0.0:
            raise ValueError(
                f"CFDCase.{self.file}: U_in должно быть положительным, "
                f"получено {self.U_in!r}"
            )
        if not self.name:
            self.name = os.path.splitext(os.path.basename(self.file))[0]


@dataclass
class GeometrySpec:
    """Источник контура стенки для расчёта SDF."""

    domain_file: Optional[str] = None
    vertices: Optional[np.ndarray] = None
    closed: bool = True

    def __post_init__(self) -> None:
        if self.domain_file is not None:
            self.domain_file = os.fspath(self.domain_file)
        if self.vertices is not None:
            vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 2)
            if vertices.shape[0] < 3:
                raise ValueError("geometry.vertices: нужно минимум 3 вершины")
            self.vertices = vertices

    @property
    def is_defined(self) -> bool:
        return self.domain_file is not None or self.vertices is not None


@dataclass
class CFDManifest:
    """Разобранный манифест."""

    cases: list[CFDCase]
    ref_scales: ReferenceScales
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    source: str = ""

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("манифест не содержит ни одного CFD-кейса")
        seen: set[str] = set()
        for case in self.cases:
            key = os.path.normcase(os.path.abspath(case.file))
            if key in seen:
                raise ValueError(f"дублирующийся файл в манифесте: {case.file}")
            seen.add(key)

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def velocities(self) -> list[float]:
        return [case.U_in for case in self.cases]


def _resolve_path(path: str, roots: Sequence[str]) -> str:
    """Разрешает относительный путь, перебирая корни по порядку."""
    candidate = os.fspath(path)
    if os.path.isabs(candidate):
        return candidate
    for root in roots:
        resolved = os.path.normpath(os.path.join(root, candidate))
        if os.path.exists(resolved):
            return resolved
    # Возвращаем первый вариант — ошибка всплывёт при чтении файла.
    return os.path.normpath(os.path.join(roots[0], candidate))


def _read_manifest_data(path: str) -> Mapping[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"манифест не найден: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(handle) or {}
        elif suffix == ".json":
            data = json.load(handle) or {}
        else:
            raise ValueError(
                f"неподдерживаемый формат манифеста: {suffix} (ожидается .yaml/.yml/.json)"
            )
    if not isinstance(data, Mapping):
        raise TypeError("корень манифеста должен быть отображением")
    return data


def _parse_cases(raw_cases: Any, roots: Sequence[str], delimiter: str) -> list[CFDCase]:
    if isinstance(raw_cases, Mapping):
        items = [
            {"file": key, "U_in": value}
            for key, value in raw_cases.items()
        ]
    elif isinstance(raw_cases, Sequence) and not isinstance(raw_cases, (str, bytes)):
        items = list(raw_cases)
    else:
        raise TypeError("cases должен быть списком или отображением {file: U_in}")

    cases: list[CFDCase] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"cases[{index}] должен быть отображением")
        file_value = item.get("file", item.get("path"))
        if file_value is None:
            raise ValueError(f"cases[{index}]: отсутствует поле 'file'")
        if "U_in" not in item:
            raise ValueError(f"cases[{index}] ({file_value}): отсутствует 'U_in'")
        resolved = _resolve_path(file_value, roots)
        cases.append(
            CFDCase(
                file=resolved,
                U_in=float(item["U_in"]),
                name=str(item.get("name", "") or ""),
                delimiter=str(item.get("delimiter", delimiter) or delimiter),
            )
        )
    return cases


def _parse_geometry(raw_geometry: Any, roots: Sequence[str]) -> GeometrySpec:
    if raw_geometry is None:
        return GeometrySpec()
    if isinstance(raw_geometry, str):
        return GeometrySpec(domain_file=_resolve_path(raw_geometry, roots))
    if not isinstance(raw_geometry, Mapping):
        raise TypeError("geometry должен быть отображением или путём к .npy")

    domain_file = raw_geometry.get("domain_file", raw_geometry.get("file"))
    vertices = raw_geometry.get("vertices")
    closed = bool(raw_geometry.get("closed", True))
    return GeometrySpec(
        domain_file=_resolve_path(domain_file, roots) if domain_file else None,
        vertices=np.asarray(vertices, dtype=np.float64) if vertices is not None else None,
        closed=closed,
    )


def load_manifest(
    path: str | os.PathLike[str],
    *,
    base_dir: Optional[str | os.PathLike[str]] = None,
    extra_roots: Optional[Sequence[str | os.PathLike[str]]] = None,
    delimiter: str = ",",
) -> CFDManifest:
    """Загружает и валидирует манифест CFD-кейсов.

    Args:
        path: Путь к YAML/JSON-манифесту.
        base_dir: Дополнительный корень для разрешения относительных путей
            (например, корень проекта). Имеет приоритет над каталогом манифеста.
        extra_roots: Дополнительные корни поиска (по порядку).
        delimiter: Разделитель колонок по умолчанию для всех кейсов.
    """
    manifest_path = os.fspath(path)
    data = _read_manifest_data(manifest_path)

    roots: list[str] = []
    if base_dir is not None:
        roots.append(os.fspath(base_dir))
    roots.append(os.path.dirname(os.path.abspath(manifest_path)))
    if extra_roots:
        roots.extend(os.fspath(root) for root in extra_roots)
    # Fallback: project-relative пути (workspace/..., docs/...) от корня проекта.
    roots.append(DEFAULT_PROJECT_ROOT)
    # Убираем дубликаты, сохраняя порядок.
    roots = list(dict.fromkeys(roots))

    scales_section = data.get("ref_scales", data.get("reference_scales"))
    if scales_section is None:
        raise ValueError("в манифесте отсутствует секция ref_scales")
    scales = ReferenceScales.from_mapping(scales_section)

    raw_cases = data.get("cases")
    if raw_cases is None:
        raise ValueError("в манифесте отсутствует секция cases")
    cases = _parse_cases(raw_cases, roots, delimiter)

    geometry = _parse_geometry(data.get("geometry"), roots)

    manifest = CFDManifest(
        cases=cases,
        ref_scales=scales,
        geometry=geometry,
        source=os.path.abspath(manifest_path),
    )
    logger.info(
        "Манифест '%s': %d кейсов, U_in=%s, U_ref=%.4g",
        manifest_path,
        manifest.n_cases,
        [case.U_in for case in manifest.cases],
        scales.velocity,
    )
    return manifest

