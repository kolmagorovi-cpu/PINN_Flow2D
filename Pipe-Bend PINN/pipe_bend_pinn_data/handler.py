#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Высокоуровневый загрузчик CFD-кейсов: манифест -> нормализованный датасет.

Типичное использование::

    from pipe_bend_pinn_data import CFDataHandler

    handler = CFDataHandler("workspace/cfd/cfd_manifest.yaml").load()
    dataset = handler.dataset
    loader = handler.dataloader(batch_size=4096)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cfd_loader import load_cfd_export
from .constants import DEFAULT_DELIMITER
from .dataset import ParametricCFDDataset
from .manifest import CFDManifest, load_manifest
from .normalization import (
    NormalizedCFD,
    ReferenceScales,
    concat_normalized,
    normalize_cfd,
)
from .sdf import PolylineSDF, load_domain_polyline
from .validation import (
    ValidationReport,
    validate_alignment,
    validate_manifest,
    validate_normalized,
    validate_raw_cfd,
)

logger = logging.getLogger(__name__)

__all__ = ["CFDataHandler"]


class CFDataHandler:
    """Загружает все CFD-кейсы из манифеста и готовит параметрический датасет."""

    def __init__(
        self,
        manifest: str | os.PathLike[str] | CFDManifest,
        *,
        base_dir: Optional[str | os.PathLike[str]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        delimiter: str = DEFAULT_DELIMITER,
        strict: bool = True,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.strict = bool(strict)

        if isinstance(manifest, CFDManifest):
            self.manifest = manifest
            self._manifest_dir = os.path.dirname(manifest.source) if manifest.source else None
        else:
            self.manifest = load_manifest(
                manifest,
                base_dir=base_dir,
                delimiter=delimiter,
            )
            self._manifest_dir = os.path.dirname(os.path.abspath(os.fspath(manifest)))

        self.ref_scales: ReferenceScales = self.manifest.ref_scales
        self.reports: List[ValidationReport] = []
        self._dataset: Optional[ParametricCFDDataset] = None
        self._data: Optional[NormalizedCFD] = None
        self._sdf: Optional[PolylineSDF] = None
        self._polyline_norm: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ geometry
    def _resolve_geometry(self) -> Optional[PolylineSDF]:
        geometry = self.manifest.geometry
        if geometry.domain_file is not None:
            vertices = load_domain_polyline(geometry.domain_file)
        elif geometry.vertices is not None:
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
        else:
            logger.warning("CFD-геометрия не задана: SDF будет нулевой")
            return None

        self._polyline_norm = vertices / self.ref_scales.pipe_width
        logger.info("CFD-геометрия: %d вершин", self._polyline_norm.shape[0])
        return PolylineSDF(self._polyline_norm, closed=geometry.closed)

    # -------------------------------------------------------------------- loading
    def _load_case(
        self,
        case,
        case_index: int,
        sdf: Optional[PolylineSDF],
    ) -> NormalizedCFD:
        raw = load_cfd_export(case.file, delimiter=case.delimiter)
        raw_report = validate_raw_cfd(raw)
        self.reports.append(raw_report)

        width = self.ref_scales.pipe_width
        points_norm = np.column_stack((raw.x / width, raw.y / width))
        if sdf is not None:
            sdf_norm = sdf.evaluate(points_norm)
            alignment = validate_alignment(points_norm, sdf_norm, self._polyline_norm)
            self.reports.append(alignment)
        else:
            sdf_norm = np.zeros(raw.n_points, dtype=np.float64)

        return normalize_cfd(
            raw,
            self.ref_scales,
            case.U_in,
            case_name=case.name,
            case_id=case_index,
            sdf_norm=sdf_norm,
        )

    def load(self) -> "CFDataHandler":
        """Загружает, нормализует и валидирует все кейсы манифеста."""
        manifest_report = validate_manifest(self.manifest)
        self.reports.append(manifest_report)

        self._sdf = self._resolve_geometry()
        parts = [
            self._load_case(case, index, self._sdf)
            for index, case in enumerate(self.manifest.cases)
        ]
        self._data = concat_normalized(parts)

        normalized_report = validate_normalized(self._data)
        self.reports.append(normalized_report)

        if self.strict:
            for report in self.reports:
                report.raise_if_invalid()
        else:
            for report in self.reports:
                if not report.ok:
                    logger.warning("Валидация CFD:\n%s", report.summary())

        self._dataset = ParametricCFDDataset(
            self._data, device=self.device, dtype=self.dtype
        )
        return self

    # ------------------------------------------------------------------ properties
    @property
    def data(self) -> NormalizedCFD:
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data

    @property
    def dataset(self) -> ParametricCFDDataset:
        if self._dataset is None:
            self.load()
        assert self._dataset is not None
        return self._dataset

    def dataloader(
        self,
        *,
        batch_size: int = 4096,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> DataLoader:
        """``DataLoader`` с перемешиванием по всем кейсам (смесь Re в батче)."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    # ------------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, object]:
        data = self.data
        cases: List[Dict[str, object]] = []
        for index, (name, source) in enumerate(zip(data.case_names, data.sources)):
            mask = data.case_id == index
            cases.append(
                {
                    "name": name,
                    "source": source,
                    "n_points": int(mask.sum()),
                    "U_in": float(data.U_in[mask][0]),
                    "Re": float(data.re[mask][0]),
                    "re_norm": float(data.re_norm[mask][0]),
                    "p_norm": [
                        float(data.p_norm[mask].min()),
                        float(data.p_norm[mask].max()),
                    ],
                    "sdf_norm": [
                        float(data.sdf_norm[mask].min()),
                        float(data.sdf_norm[mask].max()),
                    ],
                }
            )
        return {
            "n_points": len(data),
            "n_cases": data.n_cases,
            "ref_scales": self.ref_scales.to_dict(),
            "Re_ref": self.ref_scales.reynolds,
            "cases": cases,
            "validation": [report.summary() for report in self.reports],
        }

    def to_structured_array(self) -> np.ndarray:
        return self.dataset.to_structured_array()

    def save_npy(self, path: str | os.PathLike[str]) -> str:
        return self.dataset.save_npy(path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        state = "loaded" if self._dataset is not None else "not loaded"
        return (
            f"CFDataHandler(state={state}, n_cases={self.manifest.n_cases}, "
            f"U_ref={self.ref_scales.velocity:g})"
        )

