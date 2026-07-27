#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Единая карта рабочих директорий проекта Airfoil PINN.

Модуль отделяет код от физической раскладки файлов. Остальные части проекта
берут пути отсюда, поэтому перенос данных между папками больше не требует
точечных правок в геометрии, сэмплинге, инициализации или обучении.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkspacePaths:
    """Набор путей для одного корня проекта.

    UI-тесты создают временный корень проекта, поэтому пути собраны в dataclass:
    можно получить такую же структуру для реального репозитория или для
    временной папки без изменения глобального состояния.
    """

    project_root: Path
    workspace_dir: Path
    configs_dir: Path
    geometry_dir: Path
    geometry_foils_dir: Path
    geometry_domains_dir: Path
    geometry_metadata_dir: Path
    foil_geoms_path: Path
    sampling_dir: Path
    sampling_collocation_dir: Path
    sampling_mesh_dir: Path
    initial_dir: Path
    initial_fields_dir: Path
    initial_plots_dir: Path
    libraries_dir: Path
    fluids_dir: Path
    conditions_dir: Path
    experiments_dir: Path
    pictures_dir: Path
    backups_dir: Path
    logs_dir: Path


def for_project_root(project_root: str | os.PathLike[str] | None = None) -> WorkspacePaths:
    """Возвращает дерево рабочих путей для указанного корня проекта."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    workspace_dir = root / "workspace"
    geometry_dir = workspace_dir / "geometry"
    sampling_dir = workspace_dir / "sampling"
    initial_dir = workspace_dir / "initial"
    libraries_dir = workspace_dir / "libraries"

    return WorkspacePaths(
        project_root=root,
        workspace_dir=workspace_dir,
        configs_dir=workspace_dir / "configs",
        geometry_dir=geometry_dir,
        geometry_foils_dir=geometry_dir / "foils",
        geometry_domains_dir=geometry_dir / "domains",
        geometry_metadata_dir=geometry_dir / "metadata",
        foil_geoms_path=geometry_dir / "foil_geoms.yaml",
        sampling_dir=sampling_dir,
        sampling_collocation_dir=sampling_dir / "collocation",
        sampling_mesh_dir=sampling_dir / "mesh",
        initial_dir=initial_dir,
        initial_fields_dir=initial_dir / "fields",
        initial_plots_dir=initial_dir / "plots",
        libraries_dir=libraries_dir,
        fluids_dir=libraries_dir / "fluids",
        conditions_dir=libraries_dir / "conditions",
        experiments_dir=workspace_dir / "experiments",
        pictures_dir=workspace_dir / "pictures",
        backups_dir=workspace_dir / "backups",
        logs_dir=workspace_dir / "logs",
    )


PATHS = for_project_root()


def ensure_workspace_dirs(paths: WorkspacePaths = PATHS) -> None:
    """Создаёт базовые директории, которые нужны для работы приложения."""

    for directory in (
        paths.configs_dir,
        paths.geometry_foils_dir,
        paths.geometry_domains_dir,
        paths.geometry_metadata_dir,
        paths.sampling_collocation_dir,
        paths.sampling_mesh_dir,
        paths.initial_fields_dir,
        paths.initial_plots_dir,
        paths.fluids_dir,
        paths.conditions_dir,
        paths.experiments_dir,
        paths.pictures_dir,
        paths.backups_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_project_path(
    path: str | os.PathLike[str],
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Преобразует относительный путь из конфига в абсолютный путь проекта."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    return root / candidate


def project_relative(path: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> str:
    """Возвращает переносимый относительный путь для YAML-конфигов."""

    return Path(os.path.relpath(Path(path), root)).as_posix()


CONFIGS_DIR = str(PATHS.configs_dir)
FLUIDS_DIR = str(PATHS.fluids_dir)
CONDITIONS_DIR = str(PATHS.conditions_dir)
GEOMETRY_DOMAINS_DIR = str(PATHS.geometry_domains_dir)
GEOMETRY_FOILS_DIR = str(PATHS.geometry_foils_dir)
GEOMETRY_METADATA_DIR = str(PATHS.geometry_metadata_dir)
FOIL_GEOMS_PATH = str(PATHS.foil_geoms_path)
SAMPLING_COLLOCATION_DIR = str(PATHS.sampling_collocation_dir)
SAMPLING_MESH_DIR = str(PATHS.sampling_mesh_dir)
INITIAL_FIELDS_DIR = str(PATHS.initial_fields_dir)
INITIAL_PLOTS_DIR = str(PATHS.initial_plots_dir)
EXPERIMENTS_DIR = str(PATHS.experiments_dir)
PICTURES_DIR = str(PATHS.pictures_dir)
BACKUPS_DIR = str(PATHS.backups_dir)
LOGS_DIR = str(PATHS.logs_dir)
