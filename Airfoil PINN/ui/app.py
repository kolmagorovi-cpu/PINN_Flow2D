"""Entry point for the Airfoil PINN PySide6 UI."""

from __future__ import annotations

import sys

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "PySide6 отсутствует в общем окружении D:\\Python\\BaseEnv. "
        "Проект не устанавливает GUI-зависимости автоматически."
    ) from exc

from ui.main_window import MainWindow
from ui.theme import app_stylesheet


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("AirfoilPINN")
    app.setApplicationName("Airfoil PINN UI")
    app.setStyleSheet(app_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
