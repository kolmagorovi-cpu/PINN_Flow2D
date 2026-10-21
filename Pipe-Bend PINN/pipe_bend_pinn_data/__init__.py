#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пакет подготовки CFD-данных для параметрической Pipe-Bend PINN.

Изолирует от обучающего ядра ``pipe_bend_pinn`` всю работу с CFD-экспортом:
разбор CSV (формат Fluent/CFX), манифест кейсов, единую нормировку по
диапазону входных скоростей, расчёт SDF, валидацию и сборку torch-датасета.

Публичный API::

    from pipe_bend_pinn_data import (
        CFDataHandler, ParametricCFDDataset,
        RawCFD, load_cfd_export,
        ReferenceScales, normalize_cfd,
        load_manifest, CFDManifest,
        validate_alignment,
    )

    handler = CFDataHandler("workspace/cfd/cfd_manifest.yaml").load()
    loader = handler.dataloader(batch_size=4096)
"""

from __future__ import annotations

from .cfd_loader import RawCFD, iter_cfd_files, load_cfd_export, parse_cfd_export
from .constants import (
    DATA_POINT_TYPE,
    NORMALIZED_DTYPE,
    REQUIRED_COLUMNS,
    UnsetColumnError,
)
from .dataset import ParametricCFDDataset
from .handler import CFDataHandler
from .manifest import CFDCase, CFDManifest, GeometrySpec, load_manifest
from .normalization import (
    NormalizedCFD,
    ReferenceScales,
    concat_normalized,
    normalize_cfd,
)
from .sdf import PolylineSDF, distance_to_polyline, load_domain_polyline, points_in_polygon
from .validation import (
    ValidationIssue,
    ValidationReport,
    validate_alignment,
    validate_manifest,
    validate_normalized,
    validate_raw_cfd,
)

__all__ = [
    "__version__",
    # constants
    "DATA_POINT_TYPE",
    "NORMALIZED_DTYPE",
    "REQUIRED_COLUMNS",
    "UnsetColumnError",
    # loader
    "RawCFD",
    "iter_cfd_files",
    "load_cfd_export",
    "parse_cfd_export",
    # manifest
    "CFDCase",
    "CFDManifest",
    "GeometrySpec",
    "load_manifest",
    # normalization
    "NormalizedCFD",
    "ReferenceScales",
    "concat_normalized",
    "normalize_cfd",
    # geometry / sdf
    "PolylineSDF",
    "distance_to_polyline",
    "load_domain_polyline",
    "points_in_polygon",
    # dataset / handler
    "ParametricCFDDataset",
    "CFDataHandler",
    # validation
    "ValidationIssue",
    "ValidationReport",
    "validate_alignment",
    "validate_manifest",
    "validate_normalized",
    "validate_raw_cfd",
]

__version__ = "0.1.0"
