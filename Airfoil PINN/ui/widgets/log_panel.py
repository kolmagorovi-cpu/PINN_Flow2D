"""Log panel for mock UI events."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QTextEdit


class LogPanel(QTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)

    def append_event(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.append(f"[{stamp}] {message}")

