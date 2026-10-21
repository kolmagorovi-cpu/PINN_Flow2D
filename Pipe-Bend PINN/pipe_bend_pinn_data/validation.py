#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверки корректности CFD-данных, нормализации и привязки к геометрии."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np

from .cfd_loader import RawCFD
from .normalization import NormalizedCFD
from .sdf import points_in_polygon

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_raw_cfd",
    "validate_normalized",
    "validate_alignment",
    "validate_manifest",
]


@dataclass
class ValidationIssue:
    level: str          # 'error' | 'warning' | 'info'
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - диагностика
        return f"[{self.level.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(level=level, code=code, message=message))

    def error(self, code: str, message: str) -> None:
        self.add("error", code, message)

    def warning(self, code: str, message: str) -> None:
        self.add("warning", code, message)

    def info(self, code: str, message: str) -> None:
        self.add("info", code, message)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.ok:
            details = "\n".join(str(issue) for issue in self.errors)
            raise ValueError(f"CFD-данные не прошли валидацию:\n{details}")

    def summary(self) -> str:
        lines = [
            f"Validation: {'OK' if self.ok else 'FAIL'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        ]
        lines.extend(f"  {issue}" for issue in self.issues)
        if self.metrics:
            lines.append(
                "  metrics: "
                + ", ".join(f"{key}={value:.6g}" for key, value in self.metrics.items())
            )
        return "\n".join(lines)


def validate_raw_cfd(raw: RawCFD) -> ValidationReport:
    """Проверяет физические поля одного кейса."""
    report = ValidationReport()
    report.metrics.update(
        n_points=float(raw.n_points),
        x_min=float(raw.x.min()), x_max=float(raw.x.max()),
        y_min=float(raw.y.min()), y_max=float(raw.y.max()),
        p_min=float(raw.p.min()), p_max=float(raw.p.max()),
    )
    if raw.n_points < 10:
        report.error("too_few_points", f"точек слишком мало: {raw.n_points}")

    speed = np.hypot(raw.u, raw.v)
    report.metrics["speed_max"] = float(speed.max())
    if raw.u_mag is not None:
        mismatch = float(np.max(np.abs(raw.u_mag - speed)))
        report.metrics["u_mag_max_mismatch"] = mismatch
        if mismatch > 1e-4 * max(1.0, float(speed.max())):
            report.warning(
                "velocity_magnitude_mismatch",
                f"|Velocity| не совпадает с hypot(u, v): max Δ={mismatch:.3e}",
            )

    if raw.z is not None:
        z_span = float(np.max(np.abs(raw.z - raw.z[0])))
        report.metrics["z_span"] = z_span
        if z_span > 1e-9:
            report.warning(
                "non_planar",
                f"Z-координата меняется на {z_span:.3e} м (ожидалось плоское сечение)",
            )
    return report


def validate_normalized(norm: NormalizedCFD) -> ValidationReport:
    """Проверяет нормализованные поля на конечность и разумные диапазоны."""
    report = ValidationReport()
    for name in ("x_norm", "y_norm", "sdf_norm", "p_norm", "u_norm", "v_norm",
                 "re", "re_norm"):
        values = np.asarray(getattr(norm, name), dtype=np.float64)
        if not np.isfinite(values).all():
            report.error("non_finite", f"{name}: обнаружены NaN/Inf")
        report.metrics[f"{name}_min"] = float(np.nanmin(values))
        report.metrics[f"{name}_max"] = float(np.nanmax(values))

    if np.any(norm.sdf_norm < -1e-9):
        report.error("negative_sdf", "SDF принимает отрицательные значения")
    if np.any(norm.re <= 0.0):
        report.error("non_positive_re", "Re должно быть положительным")
    return report


def validate_alignment(
    points_norm: np.ndarray,
    sdf_norm: np.ndarray,
    polyline_norm: np.ndarray,
    near_wall_threshold: float = 0.02,
) -> ValidationReport:
    """Проверяет, что CFD-точки лежат внутри заданной геометрии.

    Это ключевая проверка соответствия: если CSV экспортирован с другой
    геометрии, доля точек внутри домена будет близка к нулю.
    """
    report = ValidationReport()
    points = np.asarray(points_norm, dtype=np.float64).reshape(-1, 2)
    sdf_values = np.asarray(sdf_norm, dtype=np.float64).reshape(-1)
    polyline = np.asarray(polyline_norm, dtype=np.float64).reshape(-1, 2)

    contains = points_in_polygon(points, polyline)
    coverage = float(contains.mean()) if contains.size else 0.0
    report.metrics.update(
        coverage=coverage,
        sdf_min=float(sdf_values.min()),
        sdf_mean=float(sdf_values.mean()),
        sdf_max=float(sdf_values.max()),
        near_wall_fraction=float((sdf_values < near_wall_threshold).mean()),
    )

    if coverage < 0.99:
        report.error(
            "outside_domain",
            f"только {coverage * 100:.2f}% CFD-точек внутри заданной геометрии; "
            "проверьте, что CSV экспортирован с этой же геометрии",
        )
    if report.metrics["sdf_min"] < -1e-9:
        report.error("negative_sdf", "SDF принимает отрицательные значения")
    if report.metrics["sdf_max"] > 10.0:
        report.warning(
            "far_points",
            f"максимальное расстояние до стенки {report.metrics['sdf_max']:.3g} "
            "в нормированных единицах — часть точек далеко от домена",
        )
    if report.metrics["near_wall_fraction"] > 0.5:
        report.warning(
            "dense_wall_points",
            "более половины точек лежат ближе 0.02·W к стенке",
        )
    return report


def validate_manifest(manifest, *, check_files: bool = True) -> ValidationReport:
    """Проверяет манифест: наличие файлов, уникальность скоростей, геометрию."""
    report = ValidationReport()
    report.metrics["n_cases"] = float(manifest.n_cases)
    report.metrics["U_min"] = float(min(manifest.velocities()))
    report.metrics["U_max"] = float(max(manifest.velocities()))

    if check_files:
        for case in manifest.cases:
            if not os.path.exists(case.file):
                report.error("missing_file", f"CFD-файл не найден: {case.file}")

    velocities = manifest.velocities()
    if len(set(velocities)) != len(velocities):
        report.warning(
            "duplicate_velocities",
            f"повторяющиеся U_in в манифесте: {sorted(velocities)}",
        )
    if manifest.n_cases < 2:
        report.warning(
            "single_case",
            "в манифесте один кейс — параметрическая интерполяция по скорости "
            "невозможна",
        )
    if not manifest.geometry.is_defined:
        report.error(
            "geometry_missing",
            "не задана геометрия (geometry.domain_file или geometry.vertices) — "
            "SDF вычислить нельзя",
        )
    return report


