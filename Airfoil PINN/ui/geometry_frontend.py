"""Standalone geometry workflow used by the UI prototype.

This module intentionally does not import the project backend. It mirrors the
Geometry.py workflow closely enough for UI testing: load airfoil coordinates,
compute stats, build a domain preview and save mock artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class GeometryStats:
    point_count: int
    is_closed: bool
    chord: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class GeometryArtifacts:
    foil_path: Path
    domain_path: Path
    metadata_path: Path


class GeometryReadError(RuntimeError):
    """Raised when standalone UI geometry loading fails."""


class GeometryFrontendSession:
    """Small independent geometry model for the frontend-only loop."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.foils_dir = self.project_root / "Geometry" / "Foils"
        self.output_dir = self.project_root / "ui" / "mock_outputs" / "Geometry" / "Data"
        self.metadata_dir = self.project_root / "ui" / "mock_outputs" / "Geometry" / "Metadata"

        self.name = "mock_naca0012"
        self.source = "mock"
        self.foil_points: list[Point] = []
        self.domain_points: list[Point] = []
        self.stats: GeometryStats | None = None
        self.load_sample("closed_naca")
        self.create_domain(front=1.5, rear=3.0, top=1.2, bottom=1.2)

    @property
    def is_loaded(self) -> bool:
        return bool(self.foil_points)

    @property
    def is_closed(self) -> bool:
        return bool(self.stats and self.stats.is_closed)

    def available_foil_files(self) -> list[Path]:
        allowed = {".txt", ".csv", ".dat", ".npy"}
        if not self.foils_dir.exists():
            return []
        return sorted(path for path in self.foils_dir.iterdir() if path.suffix.lower() in allowed)

    def load_sample(self, sample: str) -> None:
        if sample == "open_plate":
            points = [Point(x / 80.0, 0.0) for x in range(81)]
            self.name = "mock_open_plate"
        else:
            points = self._make_naca0012()
            self.name = "mock_naca0012"

        self.source = "mock"
        self.foil_points = points
        self.domain_points = []
        self.stats = self._compute_stats(points)

    def load_file(self, filepath: str | Path) -> None:
        path = Path(filepath)
        if not path.exists():
            raise GeometryReadError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".npy":
            points = self._load_npy(path)
        elif ext == ".csv":
            points = self._load_csv(path)
        else:
            points = self._load_text(path)

        if len(points) < 2:
            raise GeometryReadError("Airfoil file must contain at least two coordinate points")

        self.name = path.stem
        self.source = str(path)
        self.foil_points = points
        self.domain_points = []
        self.stats = self._compute_stats(points)

    def load_text_block(self, text: str, name: str = "manual_profile") -> None:
        points = self._parse_coordinate_text(text)
        if len(points) < 2:
            raise GeometryReadError("Manual coordinates must contain at least two valid points")

        self.name = self._clean_name(name or "manual_profile")
        self.source = "manual input"
        self.foil_points = points
        self.domain_points = []
        self.stats = self._compute_stats(points)

    def create_domain(self, front: float, top: float, rear: float | None = None, bottom: float | None = None) -> None:
        if not self.stats:
            raise GeometryReadError("Load an airfoil before creating a domain")

        if self.stats.is_closed:
            rear = 3.0 if rear is None else rear
            bottom = 1.0 if bottom is None else bottom
            self.domain_points = [
                Point(-front, -bottom),
                Point(-front, top),
                Point(rear, top),
                Point(rear, -bottom),
                Point(-front, -bottom),
            ]
            return

        x_min = self.stats.x_min
        x_max = self.stats.x_max
        y_trailing = self.y_at_x(x_max)
        first = self.foil_points[0]
        points = list(self.foil_points)
        points.extend(
            [
                Point(x_max, top),
                Point(-front, top),
                Point(-front, 0.0),
                Point(x_min, 0.0),
                Point(first.x, first.y),
            ]
        )
        if math.isnan(y_trailing):
            points[-2] = Point(x_min, 0.0)
        self.domain_points = points

    def y_at_x(self, x_target: float) -> float:
        points = sorted(self.foil_points, key=lambda point: point.x)
        unique: list[Point] = []
        seen: set[float] = set()
        for point in points:
            key = round(point.x, 12)
            if key not in seen:
                unique.append(point)
                seen.add(key)

        if not unique:
            return float("nan")
        if len(unique) == 1:
            return unique[0].y

        for left, right in zip(unique, unique[1:]):
            if left.x <= x_target <= right.x:
                span = right.x - left.x
                if abs(span) < 1e-12:
                    return left.y
                t = (x_target - left.x) / span
                return left.y + t * (right.y - left.y)

        first, last = unique[0], unique[-1]
        span = last.x - first.x
        if abs(span) < 1e-12:
            return first.y
        t = (x_target - first.x) / span
        return first.y + t * (last.y - first.y)

    def save(self, base_name: str, fmt: str) -> GeometryArtifacts:
        if not self.foil_points or not self.domain_points or not self.stats:
            raise GeometryReadError("Load an airfoil and build a domain before saving")

        fmt = fmt.lower().lstrip(".")
        if fmt not in {"npy", "csv", "txt"}:
            raise GeometryReadError(f"Unsupported save format: {fmt}")

        clean_name = self._clean_name(base_name or self.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        foil_path = self.metadata_dir / f"{clean_name}_foil.{fmt}"
        domain_path = self.output_dir / f"{clean_name}_domain.{fmt}"
        metadata_path = self.metadata_dir / f"{clean_name}_metadata.json"

        self._save_points(foil_path, self.foil_points, fmt)
        self._save_points(domain_path, self.domain_points, fmt)

        metadata = {
            "mode": "frontend_standalone",
            "geom_type": "airfoil",
            "name": clean_name,
            "source": self.source,
            "chord": self.stats.chord,
            "is_closed": self.stats.is_closed,
            "num_foil_points": len(self.foil_points),
            "num_domain_points": len(self.domain_points),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return GeometryArtifacts(foil_path=foil_path, domain_path=domain_path, metadata_path=metadata_path)

    def canvas_bounds(self) -> tuple[float, float, float, float]:
        points = self.domain_points or self.foil_points
        if not points:
            return -1.5, 3.0, -1.2, 1.2
        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        return min(x_values), max(x_values), min(y_values), max(y_values)

    def _load_npy(self, path: Path) -> list[Point]:
        np = self._numpy()
        data = np.load(path, allow_pickle=False)
        return self._points_from_rows(data.tolist())

    def _load_csv(self, path: Path) -> list[Point]:
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            rows.extend(row for row in reader)
        return self._points_from_rows(rows)

    def _load_text(self, path: Path) -> list[Point]:
        return self._parse_coordinate_text(path.read_text(encoding="utf-8-sig"))

    def _parse_coordinate_text(self, text: str) -> list[Point]:
        rows: list[list[str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stripped = stripped.replace(",", " ").replace(";", " ")
            rows.append(stripped.split())
        return self._points_from_rows(rows)

    def _points_from_rows(self, rows: Iterable[Sequence[object]]) -> list[Point]:
        points: list[Point] = []
        for row in rows:
            if len(row) < 2:
                continue
            try:
                points.append(Point(float(row[0]), float(row[1])))
            except (TypeError, ValueError):
                continue
        return points

    def _compute_stats(self, points: list[Point]) -> GeometryStats:
        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        first = points[0]
        last = points[-1]
        is_closed = math.hypot(first.x - last.x, first.y - last.y) <= 1e-6
        x_min = min(x_values)
        x_max = max(x_values)
        return GeometryStats(
            point_count=len(points),
            is_closed=is_closed,
            chord=x_max - x_min,
            x_min=x_min,
            x_max=x_max,
            y_min=min(y_values),
            y_max=max(y_values),
        )

    def _save_points(self, path: Path, points: list[Point], fmt: str) -> None:
        if fmt == "npy":
            np = self._numpy()
            np.save(path, np.array([[point.x, point.y] for point in points], dtype=float))
        elif fmt == "csv":
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "y"])
                writer.writerows((point.x, point.y) for point in points)
        else:
            lines = [f"{point.x:.8f}\t{point.y:.8f}" for point in points]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _make_naca0012(self, count: int = 96) -> list[Point]:
        half = max(12, count // 2)
        thickness = 0.12
        upper: list[Point] = []
        lower: list[Point] = []

        for index in range(half):
            beta = math.pi * index / (half - 1)
            x = 0.5 * (1.0 - math.cos(beta))
            yt = 5.0 * thickness * (
                0.2969 * math.sqrt(max(x, 0.0))
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
            upper.append(Point(x, yt))
            lower.append(Point(x, -yt))

        points = upper + list(reversed(lower))
        points.append(points[0])
        return points

    def _clean_name(self, value: str) -> str:
        allowed = []
        for char in value.strip():
            if char.isalnum() or char in {"_", "-"}:
                allowed.append(char)
            elif char in {" ", "."}:
                allowed.append("_")
        return "".join(allowed).strip("_") or "airfoil"

    def _numpy(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise GeometryReadError("NumPy is required for .npy geometry files") from exc
        return np
