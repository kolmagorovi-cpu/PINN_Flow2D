"""Plain-text log panel for UI and backend output."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit


class LogPanel(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)

    def append_event(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.appendPlainText(f"[{stamp}] {message}")

    def append_raw(self, message: str) -> None:
        self.appendPlainText(message)

    def append_raw_lines(self, messages: Iterable[str]) -> None:
        text = "\n".join(message for message in messages if message)
        if text:
            self.appendPlainText(text)
