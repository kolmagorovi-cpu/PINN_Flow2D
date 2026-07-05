"""Geometry workflow used by the UI.

This module keeps geometry editing independent from the heavy backend modules:
it loads airfoil coordinates, computes stats, builds a domain preview and saves
artifacts into the workspace geometry folders.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from airfoil_pinn.app.paths import for_project_root


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
    """Raised when UI geometry loading fails."""


class GeometryFrontendSession:
    """Independent geometry model for the UI loop."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        paths = for_project_root(self.project_root)
        self.foils_dir = paths.geometry_foils_dir
        self.output_dir = paths.geometry_domains_dir
        self.metadata_dir = paths.geometry_metadata_dir
        self.geom_config_path = paths.geom_config_path
        self.foil_library_path = paths.foil_geoms_path

        self.name = "airfoil"
        self.source = "not loaded"
        self.default_save_format = "npy"
        self.last_domain_params: dict[str, float] = {"front": 1.5, "rear": 3.0, "top": 1.2, "bottom": 1.2}
        self.foil_points: list[Point] = []
        self.domain_points: list[Point] = []
        self.stats: GeometryStats | None = None

        files = self.available_foil_files()
        if files:
            self.load_file(files[0])
        else:
            self.load_sample("closed_naca")
        self.create_domain(**self.last_domain_params)

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

    def available_library_names(self) -> list[str]:
        if not self.foil_library_path.exists():
            return []
        data = self._load_yaml_file(self.foil_library_path)
        return sorted(item.get("name", "") for item in data.get("foils", []) if item.get("name"))

    def load_sample(self, sample: str) -> None:
        if sample == "open_plate":
            points = [Point(x / 80.0, 0.0) for x in range(81)]
            self.name = "mock_open_plate"
        elif sample in {"naca2412", "NACA 2412"}:
            points = self._make_naca4(max_camber=0.02, camber_position=0.4, thickness=0.12)
            self.name = "NACA_2412"
        elif sample in {"rae2822", "RAE 2822"}:
            points = self._make_rae2822()
            self.name = "RAE_2822"
        else:
            points = self._make_naca0012()
            self.name = "NACA_0012"

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

    def load_from_config(self, config_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(config_path) if config_path else self.geom_config_path
        if not path.exists():
            raise GeometryReadError(f"Geometry config not found: {path}")

        config = self._load_yaml_file(path)
        geom_name = config.get("geom_name") or "airfoil"
        common_params = config.get("common_params", {}) or {}
        airfoil_params = config.get("airfoil_params", {}) or {}

        self._apply_common_params(common_params)
        source_type = airfoil_params.get("source_type", "inline")

        if source_type == "file":
            file_path = airfoil_params.get("file_path")
            if not file_path:
                raise GeometryReadError("geom_config.yaml: airfoil_params.file_path is empty")
            self.load_file(self._resolve_path(file_path, base_dir=path.parent))
            self.name = self._clean_name(geom_name)
        elif source_type == "inline":
            if "X" in airfoil_params and "Y" in airfoil_params:
                self.load_inline_xy(airfoil_params.get("X", []), airfoil_params.get("Y", []), geom_name)
            else:
                self.load_from_library(geom_name)
        else:
            raise GeometryReadError(f"Unsupported airfoil source_type: {source_type}")

        self.last_domain_params = self._domain_params_from_config(airfoil_params.get("domain_params", {}) or {})
        return config

    def load_from_library(self, geom_name: str, library_path: str | Path | None = None) -> None:
        path = Path(library_path) if library_path else self.foil_library_path
        if not path.exists():
            raise GeometryReadError(f"Foil library not found: {path}")

        data = self._load_yaml_file(path)
        for item in data.get("foils", []):
            if item.get("name") == geom_name:
                self._load_from_library_item(item, path.parent)
                return
        raise GeometryReadError(f"Airfoil '{geom_name}' not found in {path}")

    def load_inline_xy(self, x_values: Sequence[object], y_values: Sequence[object], name: str) -> None:
        if len(x_values) != len(y_values):
            raise GeometryReadError("Inline X and Y arrays must have the same length")
        points: list[Point] = []
        for x_value, y_value in zip(x_values, y_values):
            try:
                points.append(Point(float(x_value), float(y_value)))
            except (TypeError, ValueError):
                continue
        if len(points) < 2:
            raise GeometryReadError("Inline geometry must contain at least two coordinate points")
        self.name = self._clean_name(name)
        self.source = "inline"
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

    def replace_foil_points(
        self,
        points: Iterable[Point | Sequence[object]],
        name: str | None = None,
        source: str = "canvas edit",
    ) -> None:
        normalized: list[Point] = []
        for point in points:
            if isinstance(point, Point):
                normalized.append(point)
                continue
            if len(point) < 2:
                continue
            try:
                normalized.append(Point(float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue

        if len(normalized) < 2:
            raise GeometryReadError("Edited geometry must contain at least two valid points")

        self.name = self._clean_name(name or self.name)
        self.source = source
        self.foil_points = normalized
        self.domain_points = []
        self.stats = self._compute_stats(normalized)

    def create_domain(self, front: float, top: float, rear: float | None = None, bottom: float | None = None) -> None:
        if not self.stats:
            raise GeometryReadError("Load an airfoil before creating a domain")

        self.last_domain_params = {
            "front": float(front),
            "top": float(top),
            "rear": float(rear if rear is not None else self.last_domain_params.get("rear", 3.0)),
            "bottom": float(bottom if bottom is not None else self.last_domain_params.get("bottom", 1.0)),
        }

        if self.stats.is_closed:
            rear = self.last_domain_params["rear"]
            bottom = self.last_domain_params["bottom"]
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
        metadata_path = self.metadata_dir / f"{clean_name}_metadata.npy"

        self._save_points(foil_path, self.foil_points, fmt)
        self._save_points(domain_path, self.domain_points, fmt)

        metadata = {
            "geom_type": "airfoil",
            "chord": self.stats.chord,
            "is_closed": self.stats.is_closed,
            "num_foil_points": len(self.foil_points),
            "num_domain_points": len(self.domain_points),
        }
        metadata_path = self._save_metadata(metadata_path, metadata)
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

    def _load_from_library_item(self, item: dict[str, Any], base_dir: Path) -> None:
        source_type = item.get("source_type", "inline")
        name = item.get("name") or "airfoil"
        if source_type == "file":
            file_path = item.get("file_path")
            if not file_path:
                raise GeometryReadError(f"Library item '{name}' has no file_path")
            self.load_file(self._resolve_path(file_path, base_dir=base_dir))
            self.name = self._clean_name(name)
            return
        if source_type == "inline":
            self.load_inline_xy(item.get("X", []), item.get("Y", []), name)
            return
        raise GeometryReadError(f"Unsupported library source_type: {source_type}")

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

    def _save_metadata(self, path: Path, metadata: dict[str, Any]) -> Path:
        try:
            np = self._numpy()
            np.save(path, metadata)
            return path
        except GeometryReadError:
            fallback = path.with_suffix(".json")
            fallback.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return fallback

    def _make_naca0012(self, count: int = 96) -> list[Point]:
        return self._make_naca4(max_camber=0.0, camber_position=0.0, thickness=0.12, count=count)

    def _make_naca4(
        self,
        max_camber: float,
        camber_position: float,
        thickness: float,
        count: int = 96,
    ) -> list[Point]:
        half = max(12, count // 2)
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
            if max_camber > 0.0 and camber_position > 0.0:
                if x < camber_position:
                    yc = max_camber / camber_position**2 * (2 * camber_position * x - x**2)
                    dyc = 2 * max_camber / camber_position**2 * (camber_position - x)
                else:
                    denom = (1 - camber_position) ** 2
                    yc = max_camber / denom * ((1 - 2 * camber_position) + 2 * camber_position * x - x**2)
                    dyc = 2 * max_camber / denom * (camber_position - x)
            else:
                yc = 0.0
                dyc = 0.0
            theta = math.atan(dyc)
            upper.append(Point(x - yt * math.sin(theta), yc + yt * math.cos(theta)))
            lower.append(Point(x + yt * math.sin(theta), yc - yt * math.cos(theta)))

        points = upper + list(reversed(lower))
        points.append(points[0])
        return points

    def _make_rae2822(self, count: int = 112) -> list[Point]:
        points = self._make_naca4(max_camber=0.018, camber_position=0.55, thickness=0.12, count=count)
        shaped: list[Point] = []
        for point in points:
            x = point.x
            if point.y >= 0:
                y = point.y * (1.0 - 0.22 * x) + 0.010 * x * (1.0 - x)
            else:
                y = point.y * (1.0 + 0.10 * x) - 0.004 * x * (1.0 - x)
            shaped.append(Point(x, y))
        shaped[-1] = shaped[0]
        return shaped

    def _clean_name(self, value: str) -> str:
        allowed = []
        for char in value.strip():
            if char.isalnum() or char in {"_", "-"}:
                allowed.append(char)
            elif char in {" ", "."}:
                allowed.append("_")
        return "".join(allowed).strip("_") or "airfoil"

    def _apply_common_params(self, common_params: dict[str, Any]) -> None:
        output_dir = common_params.get("output_dir")
        if output_dir:
            self.output_dir = self._resolve_path(output_dir, base_dir=self.project_root)
        save_format = common_params.get("save_format")
        if save_format:
            self.default_save_format = str(save_format).lower().lstrip(".")

    def _domain_params_from_config(self, params: dict[str, Any]) -> dict[str, float]:
        result = dict(self.last_domain_params)
        for key in ("front", "rear", "top", "bottom"):
            if key in params and params[key] is not None:
                try:
                    result[key] = float(params[key])
                except (TypeError, ValueError):
                    pass
        return result

    def _resolve_path(self, value: str | Path, base_dir: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [
            self.project_root / path,
            base_dir / path,
            self.foils_dir / path,
            self.foils_dir / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_yaml_file(self, path: Path) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as exc:
            raise GeometryReadError("PyYAML is required to read geometry YAML files") from exc
        with path.open("r", encoding="utf-8-sig") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise GeometryReadError(f"YAML file must contain a mapping: {path}")
        return data

    def _numpy(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise GeometryReadError("NumPy is required for .npy geometry files") from exc
        return np
