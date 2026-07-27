"""Small live metric plot drawn with QPainter."""

from __future__ import annotations

import math
from typing import Iterable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ui.theme import PALETTE


class MetricPlot(QWidget):
    """Compact plot widget for mock training curves."""

    def __init__(self, title: str, color: str = PALETTE.accent, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.values: list[float] = []
        self.setMinimumHeight(140)

    def set_values(self, values: Iterable[float]) -> None:
        self.values = list(values)
        self.update()

    def append_value(self, value: float, max_points: int = 180) -> None:
        self.values.append(value)
        if len(self.values) > max_points:
            self.values = self.values[-max_points:]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        plot_rect = QRectF(42, 30, self.width() - 58, self.height() - 48)

        painter.setPen(QPen(QColor("#d6dee6"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        painter.setPen(QColor("#14202b"))
        painter.drawText(14, 20, self.title)

        painter.setPen(QPen(QColor("#edf1f5"), 1))
        for i in range(5):
            y = plot_rect.top() + i * plot_rect.height() / 4
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)

        if len(self.values) < 2:
            painter.setPen(QColor(PALETTE.muted))
            painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "waiting")
            return

        safe_values = [max(v, 1e-12) for v in self.values]
        min_log = min(math.log10(v) for v in safe_values)
        max_log = max(math.log10(v) for v in safe_values)
        if abs(max_log - min_log) < 1e-9:
            max_log += 1.0
            min_log -= 1.0

        path = QPainterPath()
        for idx, value in enumerate(safe_values):
            x = plot_rect.left() + idx * plot_rect.width() / (len(safe_values) - 1)
            log_value = math.log10(value)
            y = plot_rect.bottom() - (log_value - min_log) * plot_rect.height() / (max_log - min_log)
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(self.color, 2.2))
        painter.drawPath(path)
        painter.setPen(QColor(PALETTE.muted))
        painter.drawText(10, int(plot_rect.top()) + 5, f"1e{max_log:.0f}")
        painter.drawText(10, int(plot_rect.bottom()), f"1e{min_log:.0f}")

