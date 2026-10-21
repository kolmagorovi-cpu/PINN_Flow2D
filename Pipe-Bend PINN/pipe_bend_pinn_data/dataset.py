#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-датасет нормализованных CFD-точек для параметрической PINN."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DATA_POINT_TYPE, NORMALIZED_DTYPE
from .normalization import NormalizedCFD

logger = logging.getLogger(__name__)

__all__ = ["ParametricCFDDataset"]


class ParametricCFDDataset(Dataset):
    """Точки CFD с полями ``(x, y, sdf, re_norm) -> (u, v, p)``.

    ``__getitem__`` возвращает словарь тензоров; координаты/поля имеют форму
    ``(1,)``, ``re_norm`` — ``(1,)``, ``case_id`` — скаляр ``int``. Такой формат
    без доработок собирается стандартным ``torch.utils.data.default_collate``.
    """

    def __init__(
        self,
        data: NormalizedCFD,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(data, NormalizedCFD):
            raise TypeError("data должен быть NormalizedCFD")
        if len(data) == 0:
            raise ValueError("пустой набор CFD-точек")

        self.data = data
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                np.ascontiguousarray(values, dtype=np.float64),
                dtype=self.dtype,
            ).view(-1, 1)

        self.x = tensor(data.x_norm)
        self.y = tensor(data.y_norm)
        self.sdf = tensor(data.sdf_norm)
        self.u = tensor(data.u_norm)
        self.v = tensor(data.v_norm)
        self.p = tensor(data.p_norm)
        self.re_norm = tensor(data.re_norm)
        self.re = tensor(data.re)
        self.U_in = tensor(data.U_in)
        self.case_id = torch.as_tensor(data.case_id.astype(np.int64), dtype=torch.long)

        logger.info(
            "ParametricCFDDataset: %d точек, %d кейсов, Re_norm ∈ [%.4f, %.4f]",
            len(self),
            data.n_cases,
            float(data.re_norm.min()),
            float(data.re_norm.max()),
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[index].to(self.device),
            "y": self.y[index].to(self.device),
            "sdf": self.sdf[index].to(self.device),
            "re_norm": self.re_norm[index].to(self.device),
            "u": self.u[index].to(self.device),
            "v": self.v[index].to(self.device),
            "p": self.p[index].to(self.device),
            "case_id": self.case_id[index].to(self.device),
        }

    @property
    def n_cases(self) -> int:
        return self.data.n_cases

    def case_slices(self) -> Dict[str, slice]:
        """Диапазоны индексов точек для каждого кейса (по ``case_names``)."""
        result: Dict[str, slice] = {}
        ids = self.data.case_id
        for case_index, name in enumerate(self.data.case_names):
            mask = ids == case_index
            if not np.any(mask):
                continue
            positions = np.flatnonzero(mask)
            result[name] = slice(int(positions[0]), int(positions[-1]) + 1)
        return result

    def to_structured_array(self) -> np.ndarray:
        """Структурированный массив, совместимый по полям с ``PINNDataset``."""
        data = self.data
        structured = np.zeros(len(data), dtype=NORMALIZED_DTYPE)
        structured["x_norm"] = data.x_norm
        structured["y_norm"] = data.y_norm
        structured["sdf_norm"] = data.sdf_norm
        structured["p_norm"] = data.p_norm
        structured["u_norm"] = data.u_norm
        structured["v_norm"] = data.v_norm
        structured["re"] = data.re
        structured["re_norm"] = data.re_norm
        structured["U_in"] = data.U_in
        structured["case_id"] = data.case_id
        structured["boundary_type"] = DATA_POINT_TYPE
        structured["on_boundary"] = False
        return structured

    def save_npy(self, path: str | os.PathLike[str]) -> str:
        """Сохраняет структурированный массив в ``.npy`` и возвращает путь."""
        destination = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(destination))
        os.makedirs(directory, exist_ok=True)
        np.save(destination, self.to_structured_array())
        logger.info("CFD-датасет сохранён: %s", destination)
        return destination
