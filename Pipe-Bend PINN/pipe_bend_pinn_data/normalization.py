#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Безразмерная нормализация CFD-полей для параметрической PINN.

Единая опорная скорость ``U_ref`` используется для **всего** диапазона
входных скоростей: это принципиально, иначе зависимость решения от Re
«схлопывается» при покейсовой нормализации.

Обозначения::

    x_norm = x / W
    y_norm = y / W
    u_norm = u / U_ref
    v_norm = v / U_ref
    p_norm = (p - p_ref) / (0.5 * rho * U_ref**2)     # коэффициент давления Cp
    Re     = rho * U_in * W / mu
    re_norm = Re / Re_ref,   Re_ref = rho * U_ref * W / mu
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from typing import Any, Mapping, Optional

import numpy as np

from .cfd_loader import RawCFD

logger = logging.getLogger(__name__)

__all__ = [
    "ReferenceScales",
    "NormalizedCFD",
    "normalize_cfd",
    "concat_normalized",
]


def _positive_finite(name: str, value: Any) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            f"ReferenceScales.{name}: ожидается конечное положительное число, "
            f"получено {value!r}"
        )
    return numeric


@dataclass(frozen=True)
class ReferenceScales:
    """Единые физические масштабы для диапазона входных скоростей.

    Attributes:
        pipe_width: Ширина канала W [м] (длина обезразмеривания).
        velocity: Опорная скорость U_ref [м/с], общая для всех кейсов.
        density: Плотность rho [кг/м^3].
        dynamic_viscosity: Динамическая вязкость mu [Па·с].
        static_pressure: Опорное давление p_ref [Па] (абсолютное или
            избыточное — должно быть согласовано со значениями в CFD-файлах).
    """

    pipe_width: float
    velocity: float
    density: float
    dynamic_viscosity: float
    static_pressure: float = 0.0

    def __post_init__(self) -> None:
        for name in ("pipe_width", "velocity", "density", "dynamic_viscosity"):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))
        pressure = float(self.static_pressure)
        if not np.isfinite(pressure):
            raise ValueError(
                f"ReferenceScales.static_pressure: ожидается конечное число, "
                f"получено {self.static_pressure!r}"
            )
        object.__setattr__(self, "static_pressure", pressure)

    @property
    def dynamic_pressure(self) -> float:
        """``0.5 * rho * U_ref**2`` [Па] — масштаб давления для Cp."""
        return 0.5 * self.density * self.velocity**2

    @property
    def kinematic_viscosity(self) -> float:
        """Кинематическая вязкость ``nu = mu / rho`` [м^2/с]."""
        return self.dynamic_viscosity / self.density

    @property
    def reynolds(self) -> float:
        """Опорное число Рейнольдса ``rho * U_ref * W / mu``."""
        return self.reynolds_for(self.velocity)

    def reynolds_for(self, velocity: float) -> float:
        """Число Рейнольдса для произвольной входной скорости."""
        speed = _positive_finite("velocity", velocity)
        return self.density * speed * self.pipe_width / self.dynamic_viscosity

    def to_dict(self) -> dict[str, float]:
        return {
            "pipe_width": self.pipe_width,
            "velocity": self.velocity,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
            "static_pressure": self.static_pressure,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ReferenceScales":
        """Создаёт масштабы из секции манифеста (с алиасами полей)."""
        if not isinstance(mapping, Mapping):
            raise TypeError("ref_scales должен быть отображением")

        def pick(*keys: str, default: Optional[float] = None) -> Optional[float]:
            for key in keys:
                if key in mapping and mapping[key] is not None:
                    return float(mapping[key])
            return default

        width = pick("pipe_width", "W", "chord")
        velocity = pick("U_ref", "velocity", "U_in", "U_inf")
        density = pick("density", "rho")
        viscosity = pick("dynamic_viscosity", "mu")
        if viscosity is None:
            kinematic = pick("kinematic_viscosity", "nu")
            if kinematic is not None and density is not None:
                viscosity = kinematic * density
        pressure = pick("static_pressure", "p_ref", default=0.0)
        if None in (width, velocity, density, viscosity):
            raise ValueError(
                "ref_scales должен содержать pipe_width, U_ref, density и "
                "dynamic_viscosity (или kinematic_viscosity)"
            )
        return cls(
            pipe_width=width,
            velocity=velocity,
            density=density,
            dynamic_viscosity=viscosity,
            static_pressure=pressure or 0.0,
        )

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class NormalizedCFD:
    """Набор нормализованных CFD-точек (возможно, из нескольких кейсов)."""

    x_norm: np.ndarray
    y_norm: np.ndarray
    sdf_norm: np.ndarray
    p_norm: np.ndarray
    u_norm: np.ndarray
    v_norm: np.ndarray
    re: np.ndarray
    re_norm: np.ndarray
    U_in: np.ndarray
    case_id: np.ndarray
    case_names: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    ref_scales: Optional[ReferenceScales] = None

    _FIELDS = (
        "x_norm", "y_norm", "sdf_norm", "p_norm", "u_norm", "v_norm",
        "re", "re_norm", "U_in", "case_id",
    )

    def __post_init__(self) -> None:
        for name in self._FIELDS:
            array = np.asarray(getattr(self, name)).reshape(-1)
            setattr(self, name, array)
        sizes = {getattr(self, name).size for name in self._FIELDS}
        if len(sizes) != 1:
            raise ValueError(f"NormalizedCFD: несогласованные длины {sorted(sizes)}")

    def __len__(self) -> int:
        return int(self.x_norm.size)

    @property
    def n_points(self) -> int:
        return len(self)

    @property
    def n_cases(self) -> int:
        return int(np.unique(self.case_id).size) if len(self) else 0


def normalize_cfd(
    raw: RawCFD,
    scales: ReferenceScales,
    U_in: float,
    case_name: str = "",
    case_id: int = 0,
    sdf_norm: Optional[np.ndarray] = None,
) -> NormalizedCFD:
    """Обезразмеривает один CFD-кейс под единые опорные масштабы.

    Args:
        raw: Физические поля кейса.
        scales: Единые опорные масштабы (U_ref общий для диапазона).
        U_in: Входная скорость этого кейса [м/с] (максимум профиля на входе).
        case_name: Имя кейса для журналирования.
        case_id: Индекс кейса (для слияния нескольких файлов).
        sdf_norm: Предвычисленная нормированная SDF; если не задана — нули.
    """
    reynolds = scales.reynolds_for(U_in)
    re_norm = reynolds / scales.reynolds

    if sdf_norm is None:
        sdf_values = np.zeros(raw.n_points, dtype=np.float64)
    else:
        sdf_values = np.asarray(sdf_norm, dtype=np.float64).reshape(-1)
        if sdf_values.size != raw.n_points:
            raise ValueError(
                f"SDF: длина {sdf_values.size} не совпадает с числом точек "
                f"{raw.n_points}"
            )

    dynamic_pressure = scales.dynamic_pressure
    n = raw.n_points
    normalized = NormalizedCFD(
        x_norm=raw.x / scales.pipe_width,
        y_norm=raw.y / scales.pipe_width,
        sdf_norm=sdf_values,
        p_norm=(raw.p - scales.static_pressure) / dynamic_pressure,
        u_norm=raw.u / scales.velocity,
        v_norm=raw.v / scales.velocity,
        re=np.full(n, reynolds, dtype=np.float64),
        re_norm=np.full(n, re_norm, dtype=np.float64),
        U_in=np.full(n, float(U_in), dtype=np.float64),
        case_id=np.full(n, int(case_id), dtype=np.int32),
        case_names=[case_name or raw.source or f"case_{case_id}"],
        sources=[raw.source],
        ref_scales=scales,
    )
    logger.info(
        "CFD-кейс '%s': U_in=%.4g м/с, Re=%.1f, re_norm=%.4f, точек=%d",
        normalized.case_names[0], U_in, reynolds, re_norm, n,
    )
    return normalized


def _stack(name: str, parts: list[NormalizedCFD]) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(part, name)) for part in parts])


def concat_normalized(parts: list[NormalizedCFD]) -> NormalizedCFD:
    """Объединяет нормализованные кейсы, перенумеровывая ``case_id``."""
    if not parts:
        raise ValueError("concat_normalized: пустой список кейсов")

    renumbered: list[NormalizedCFD] = []
    case_names: list[str] = []
    sources: list[str] = []
    offset = 0
    for part in parts:
        unique_ids = np.unique(part.case_id)
        mapping = {int(old): offset + index for index, old in enumerate(unique_ids)}
        new_ids = np.array(
            [mapping[int(old)] for old in part.case_id], dtype=np.int32
        )
        renumbered.append(
            NormalizedCFD(
                x_norm=part.x_norm,
                y_norm=part.y_norm,
                sdf_norm=part.sdf_norm,
                p_norm=part.p_norm,
                u_norm=part.u_norm,
                v_norm=part.v_norm,
                re=part.re,
                re_norm=part.re_norm,
                U_in=part.U_in,
                case_id=new_ids,
                case_names=list(part.case_names),
                sources=list(part.sources),
                ref_scales=part.ref_scales,
            )
        )
        case_names.extend(part.case_names)
        sources.extend(part.sources)
        offset += len(unique_ids)

    return NormalizedCFD(
        x_norm=_stack("x_norm", renumbered),
        y_norm=_stack("y_norm", renumbered),
        sdf_norm=_stack("sdf_norm", renumbered),
        p_norm=_stack("p_norm", renumbered),
        u_norm=_stack("u_norm", renumbered),
        v_norm=_stack("v_norm", renumbered),
        re=_stack("re", renumbered),
        re_norm=_stack("re_norm", renumbered),
        U_in=_stack("U_in", renumbered),
        case_id=_stack("case_id", renumbered).astype(np.int32),
        case_names=case_names,
        sources=sources,
        ref_scales=parts[0].ref_scales,
    )


