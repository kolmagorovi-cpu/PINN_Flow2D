"""Mock data generators used by the standalone UI prototype."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import QPointF, QRectF


@dataclass(frozen=True)
class Domain:
    front: float = 1.5
    rear: float = 3.0
    top: float = 1.2
    bottom: float = 1.2

    @property
    def rect(self) -> QRectF:
        return QRectF(-self.front, -self.bottom, self.front + self.rear, self.top + self.bottom)


@dataclass(frozen=True)
class CollocationPoint:
    x: float
    y: float
    kind: str


def naca_0012_points(count: int = 92, angle_deg: float = 0.0) -> list[QPointF]:
    """Create a smooth NACA 0012-like closed contour."""
    upper: list[QPointF] = []
    lower: list[QPointF] = []
    half = max(12, count // 2)
    thickness = 0.12

    for i in range(half):
        beta = math.pi * i / (half - 1)
        x = 0.5 * (1 - math.cos(beta))
        yt = 5 * thickness * (
            0.2969 * math.sqrt(max(x, 0.0))
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
        upper.append(QPointF(x, yt))
        lower.append(QPointF(x, -yt))

    points = upper + list(reversed(lower[1:-1]))
    return rotate_points(points, angle_deg, pivot=QPointF(0.25, 0.0))


def rotate_points(points: Iterable[QPointF], angle_deg: float, pivot: QPointF) -> list[QPointF]:
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    rotated: list[QPointF] = []
    for point in points:
        x = point.x() - pivot.x()
        y = point.y() - pivot.y()
        rotated.append(QPointF(pivot.x() + x * ca - y * sa, pivot.y() + x * sa + y * ca))
    return rotated


def generate_collocation_points(
    domain: Domain,
    count: int = 850,
    foil_points: Iterable[QPointF] | None = None,
) -> list[CollocationPoint]:
    """Generate deterministic-looking mock collocation points."""
    rng = random.Random(42 + count)
    rect = domain.rect
    points: list[CollocationPoint] = []
    boundary_count = max(32, count // 7)

    for _ in range(count):
        x = rng.uniform(rect.left(), rect.right())
        y = rng.uniform(rect.top(), rect.bottom())
        points.append(CollocationPoint(x, y, "interior"))

    for i in range(boundary_count):
        t = i / max(1, boundary_count - 1)
        y = rect.top() + t * rect.height()
        points.append(CollocationPoint(rect.left(), y, "inlet"))
        points.append(CollocationPoint(rect.right(), y, "outlet"))

    for i in range(boundary_count):
        t = i / max(1, boundary_count - 1)
        x = rect.left() + t * rect.width()
        points.append(CollocationPoint(x, rect.top(), "symmetry"))
        points.append(CollocationPoint(x, rect.bottom(), "wall"))

    foil = list(foil_points) if foil_points is not None else naca_0012_points(72)
    for point in foil:
        points.append(CollocationPoint(point.x(), point.y(), "wall"))

    return points


def next_metric_value(previous: float, floor: float, jitter: float = 0.045) -> float:
    decay = 0.965 + random.random() * 0.025
    noise = 1.0 + random.uniform(-jitter, jitter)
    return max(floor, previous * decay * noise)
