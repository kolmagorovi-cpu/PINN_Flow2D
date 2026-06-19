"""Interactive geometry preview with mock flow animation."""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QBrush, QWheelEvent
from PySide6.QtWidgets import QWidget

from ui.mock_data import Domain, generate_collocation_points, naca_0012_points
from ui.theme import PALETTE


class GeometryCanvas(QWidget):
    """Canvas for testing pan, zoom, mock airfoil editing and animated streamlines."""

    cursorChanged = Signal(float, float)
    selectionChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(620, 440)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.domain = Domain()
        self.angle_deg = 0.0
        self.geometry_name = "mock_naca0012"
        self.is_closed = True
        self.foil_points = naca_0012_points(angle_deg=self.angle_deg)
        self.domain_points = self._domain_polygon_from_rect()
        self.collocation_points = generate_collocation_points(self.domain, foil_points=self.foil_points)
        self._view_rect_cache = self._compute_view_rect()

        self.show_points = True
        self.show_streamlines = False
        self.show_grid = True

        self._scale = 150.0
        self._pan = QPointF(0.0, 0.0)
        self._last_mouse: Optional[QPointF] = None
        self._dragging_vertex: Optional[int] = None
        self._phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._tick)

    def rebuild_geometry(self, domain: Domain, angle_deg: float) -> None:
        self.domain = domain
        self.angle_deg = angle_deg
        self.geometry_name = "mock_naca0012"
        self.is_closed = True
        self.foil_points = naca_0012_points(angle_deg=angle_deg)
        self.domain_points = self._domain_polygon_from_rect()
        self._view_rect_cache = self._compute_view_rect()
        self.collocation_points = generate_collocation_points(self.domain, len(self.collocation_points), self.foil_points)
        self.update()

    def set_geometry(
        self,
        foil_points: list[QPointF],
        domain_points: list[QPointF],
        name: str,
        is_closed: bool,
    ) -> None:
        self.foil_points = list(foil_points)
        self.domain_points = list(domain_points) if domain_points else self._domain_polygon_from_rect()
        self.geometry_name = name
        self.is_closed = is_closed
        self.domain = self._domain_from_points(self.domain_points or self.foil_points)
        self._view_rect_cache = self._compute_view_rect()
        self.collocation_points = generate_collocation_points(self.domain, len(self.collocation_points), self.foil_points)
        self.update()

    def regenerate_points(self, count: int) -> None:
        self.collocation_points = generate_collocation_points(self.domain, count, self.foil_points)
        self.update()

    def reset_view(self) -> None:
        self._scale = self._fit_scale()
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def set_points_visible(self, visible: bool) -> None:
        self.show_points = visible
        self.update()

    def set_streamlines_visible(self, visible: bool) -> None:
        self.show_streamlines = visible
        self._sync_animation_timer()
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        self.show_grid = visible
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#fbfcfe"))

        if self._scale <= 0:
            self._scale = self._fit_scale()

        if self.show_grid:
            self._draw_grid(painter)
        self._draw_domain(painter)
        if self.show_streamlines:
            self._draw_streamlines(painter)
        if self.show_points:
            self._draw_points(painter)
        self._draw_airfoil(painter)
        self._draw_overlay(painter)

    def resizeEvent(self, event) -> None:  # noqa: N802
        if event.oldSize().width() <= 0:
            self.reset_view()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        self._sync_animation_timer()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            vertex = self._nearest_vertex(event.position())
            if vertex is not None:
                self._dragging_vertex = vertex
                self.selectionChanged.emit(f"Airfoil vertex #{vertex + 1}")
            else:
                self._last_mouse = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        world = self._screen_to_world(pos)
        self.cursorChanged.emit(world.x(), world.y())

        if self._dragging_vertex is not None:
            self.foil_points[self._dragging_vertex] = world
            self._view_rect_cache = self._compute_view_rect()
            self.update()
        elif self._last_mouse is not None:
            delta = pos - self._last_mouse
            self._pan += QPointF(delta.x(), delta.y())
            self._last_mouse = pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._last_mouse = None
        self._dragging_vertex = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        old_world = self._screen_to_world(event.position())
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._scale = max(35.0, min(620.0, self._scale * factor))
        new_screen = self._world_to_screen(old_world)
        delta = event.position() - new_screen
        self._pan += QPointF(delta.x(), delta.y())
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.018) % 1.0
        if self.show_streamlines:
            self.update()

    def _fit_scale(self) -> float:
        rect = self._view_rect().adjusted(-0.15, -0.15, 0.15, 0.15)
        sx = max(1.0, self.width() - 60) / max(0.1, rect.width())
        sy = max(1.0, self.height() - 60) / max(0.1, rect.height())
        return min(sx, sy)

    def _world_to_screen(self, point: QPointF) -> QPointF:
        rect = self._view_rect()
        cx = rect.center().x()
        cy = rect.center().y()
        x = (point.x() - cx) * self._scale + self.width() / 2 + self._pan.x()
        y = -(point.y() - cy) * self._scale + self.height() / 2 + self._pan.y()
        return QPointF(x, y)

    def _screen_to_world(self, point: QPointF) -> QPointF:
        rect = self._view_rect()
        cx = rect.center().x()
        cy = rect.center().y()
        x = (point.x() - self.width() / 2 - self._pan.x()) / self._scale + cx
        y = -(point.y() - self.height() / 2 - self._pan.y()) / self._scale + cy
        return QPointF(x, y)

    def _draw_grid(self, painter: QPainter) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#e7edf3"), 1))
        rect = self._view_rect().adjusted(-0.6, -0.6, 0.6, 0.6)
        step = 0.25
        x = math.floor(rect.left() / step) * step
        while x <= rect.right():
            p1 = self._world_to_screen(QPointF(x, rect.top()))
            p2 = self._world_to_screen(QPointF(x, rect.bottom()))
            painter.drawLine(p1, p2)
            x += step
        y = math.floor(rect.top() / step) * step
        while y <= rect.bottom():
            p1 = self._world_to_screen(QPointF(rect.left(), y))
            p2 = self._world_to_screen(QPointF(rect.right(), y))
            painter.drawLine(p1, p2)
            y += step
        painter.restore()

    def _draw_domain(self, painter: QPainter) -> None:
        painter.save()
        if self.domain_points:
            path = QPainterPath()
            path.moveTo(self._world_to_screen(self.domain_points[0]))
            for point in self._decimated_points(self.domain_points[1:], limit=1000):
                path.lineTo(self._world_to_screen(point))
            painter.setBrush(QColor(235, 244, 250, 68))
            painter.setPen(QPen(QColor(PALETTE.accent), 2))
            painter.drawPath(path)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor(PALETTE.accent_dark), 1))
            step = max(1, math.ceil(len(self.domain_points) / 18))
            for point in self.domain_points[::step]:
                screen = self._world_to_screen(point)
                painter.drawRect(QRectF(screen - QPointF(3, 3), screen + QPointF(3, 3)))
        else:
            rect = self.domain.rect
            top_left = self._world_to_screen(QPointF(rect.left(), rect.top()))
            bottom_right = self._world_to_screen(QPointF(rect.right(), rect.bottom()))
            screen_rect = QRectF(top_left, bottom_right).normalized()
            painter.setBrush(QColor(235, 244, 250, 68))
            painter.setPen(QPen(QColor(PALETTE.accent), 2))
            painter.drawRoundedRect(screen_rect, 4, 4)
        painter.restore()

    def _draw_streamlines(self, painter: QPainter) -> None:
        painter.save()
        rect = self._view_rect()
        pen = QPen(QColor(18, 100, 163, 120), 1.5)
        painter.setPen(pen)
        lines = 8
        for row in range(lines):
            y = rect.top() + (row + 0.5) * rect.height() / lines
            path = QPainterPath()
            start = QPointF(rect.left(), y)
            path.moveTo(self._world_to_screen(start))
            segments = 44
            for i in range(1, segments + 1):
                t = i / segments
                x = rect.left() + t * rect.width()
                disturbance = 0.09 * math.sin((t * 3.5 + self._phase) * math.tau + row * 0.35)
                foil_wake = 0.08 * math.exp(-((x - 0.55) ** 2) / 0.22) * math.sin(row * 0.9 + self._phase * math.tau)
                point = QPointF(x, y + disturbance + foil_wake)
                path.lineTo(self._world_to_screen(point))
            painter.drawPath(path)

            dot_t = (self._phase + row * 0.071) % 1.0
            dot_x = rect.left() + dot_t * rect.width()
            dot_y = y + 0.09 * math.sin((dot_t * 3.5 + self._phase) * math.tau + row * 0.35)
            screen = self._world_to_screen(QPointF(dot_x, dot_y))
            painter.setBrush(QColor(18, 100, 163, 170))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(screen, 3, 3)
            painter.setPen(pen)
        painter.restore()

    def _draw_points(self, painter: QPainter) -> None:
        painter.save()
        colors = {
            "inlet": QColor(PALETTE.inlet),
            "outlet": QColor(PALETTE.outlet),
            "wall": QColor(PALETTE.wall),
            "symmetry": QColor(PALETTE.symmetry),
            "interior": QColor(PALETTE.interior),
        }
        painter.setPen(Qt.PenStyle.NoPen)
        stride = max(1, math.ceil(len(self.collocation_points) / 900))
        for idx, point in enumerate(self.collocation_points):
            if idx % stride:
                continue
            color = colors.get(point.kind, QColor(PALETTE.interior))
            color.setAlpha(140 if point.kind == "interior" else 210)
            painter.setBrush(color)
            screen = self._world_to_screen(QPointF(point.x, point.y))
            radius = 2 if point.kind == "interior" else 3
            painter.drawEllipse(screen, radius, radius)
        painter.restore()

    def _draw_airfoil(self, painter: QPainter) -> None:
        painter.save()
        path = QPainterPath()
        if self.foil_points:
            path.moveTo(self._world_to_screen(self.foil_points[0]))
            for point in self._decimated_points(self.foil_points[1:], limit=900):
                path.lineTo(self._world_to_screen(point))
            if self.is_closed:
                path.closeSubpath()
        painter.setBrush(QBrush(QColor("#f8fafc") if self.is_closed else QColor(248, 250, 252, 0)))
        painter.setPen(QPen(QColor("#0f172a"), 2.2))
        painter.drawPath(path)

        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor(PALETTE.accent_dark), 1))
        for idx in range(0, len(self.foil_points), max(1, len(self.foil_points) // 18)):
            painter.drawEllipse(self._world_to_screen(self.foil_points[idx]), 3, 3)
        painter.restore()

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#475569"), 1))
        painter.setBrush(QColor(255, 255, 255, 225))
        box = QRectF(14, 14, 224, 82)
        painter.drawRoundedRect(box, 6, 6)
        rect = self._view_rect()
        painter.drawText(28, 40, f"Geometry: {self.geometry_name}")
        painter.drawText(28, 62, f"Domain: {rect.width():.2f} x {rect.height():.2f} m")
        painter.drawText(28, 84, f"Points: {len(self.foil_points)} / {len(self.collocation_points)}")
        painter.restore()

    def _nearest_vertex(self, screen_pos: QPointF) -> Optional[int]:
        best_idx: Optional[int] = None
        best_dist = 10.0
        for idx, point in enumerate(self.foil_points):
            screen = self._world_to_screen(point)
            dist = math.hypot(screen.x() - screen_pos.x(), screen.y() - screen_pos.y())
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _domain_polygon_from_rect(self) -> list[QPointF]:
        rect = self.domain.rect
        return [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.left(), rect.bottom()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.left(), rect.top()),
        ]

    def _domain_from_points(self, points: list[QPointF]) -> Domain:
        if not points:
            return Domain()
        min_x = min(point.x() for point in points)
        max_x = max(point.x() for point in points)
        min_y = min(point.y() for point in points)
        max_y = max(point.y() for point in points)
        return Domain(
            front=max(0.1, -min_x),
            rear=max(0.1, max_x),
            top=max(0.1, max_y),
            bottom=max(0.0, -min_y),
        )

    def _view_rect(self) -> QRectF:
        return self._view_rect_cache

    def _compute_view_rect(self) -> QRectF:
        points = self.domain_points or self.foil_points
        if not points:
            return self.domain.rect
        min_x = min(point.x() for point in points)
        max_x = max(point.x() for point in points)
        min_y = min(point.y() for point in points)
        max_y = max(point.y() for point in points)
        if abs(max_x - min_x) < 1e-9:
            max_x += 0.5
            min_x -= 0.5
        if abs(max_y - min_y) < 1e-9:
            max_y += 0.5
            min_y -= 0.5
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def _sync_animation_timer(self) -> None:
        if self.show_streamlines and self.isVisible():
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _decimated_points(self, points: list[QPointF], limit: int) -> list[QPointF]:
        if len(points) <= limit:
            return points
        step = max(1, math.ceil(len(points) / limit))
        selected = points[::step]
        if selected[-1] != points[-1]:
            selected.append(points[-1])
        return selected
