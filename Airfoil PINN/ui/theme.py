"""Qt stylesheet and shared visual constants for the UI prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    window: str = "#f4f6f8"
    panel: str = "#ffffff"
    panel_alt: str = "#eef3f7"
    border: str = "#d6dee6"
    text: str = "#14202b"
    muted: str = "#607080"
    accent: str = "#1264a3"
    accent_dark: str = "#0b4f82"
    success: str = "#178a52"
    warning: str = "#b36b00"
    danger: str = "#b42318"
    inlet: str = "#2f80ed"
    outlet: str = "#8b5cf6"
    wall: str = "#ef4444"
    symmetry: str = "#14b8a6"
    interior: str = "#334155"


PALETTE = Palette()


def app_stylesheet() -> str:
    """Return a restrained engineering-style Qt stylesheet."""
    return f"""
    QMainWindow, QWidget {{
        background: {PALETTE.window};
        color: {PALETTE.text};
        font-family: Segoe UI, Arial, sans-serif;
        font-size: 10pt;
    }}

    QFrame#Sidebar {{
        background: #152231;
        border: none;
    }}

    QLabel#AppTitle {{
        color: #ffffff;
        font-size: 14pt;
        font-weight: 700;
    }}

    QLabel#AppSubtitle {{
        color: #a9b8c8;
        font-size: 8.5pt;
    }}

    QListWidget#StepList {{
        background: transparent;
        border: none;
        color: #dbe7f2;
        outline: none;
    }}

    QListWidget#StepList::item {{
        min-height: 40px;
        padding: 8px 10px;
        border-radius: 6px;
        margin: 2px 8px;
    }}

    QListWidget#StepList::item:selected {{
        background: #244563;
        color: #ffffff;
    }}

    QListWidget#StepList::item:hover {{
        background: #1d344a;
    }}

    QFrame#Panel, QWidget#Panel {{
        background: {PALETTE.panel};
        border: 1px solid {PALETTE.border};
        border-radius: 8px;
    }}

    QLabel#SectionTitle {{
        font-size: 13pt;
        font-weight: 700;
    }}

    QLabel#MetricValue {{
        font-size: 16pt;
        font-weight: 700;
        color: {PALETTE.accent_dark};
    }}

    QLabel#Muted {{
        color: {PALETTE.muted};
    }}

    QToolButton, QPushButton {{
        background: {PALETTE.panel};
        border: 1px solid {PALETTE.border};
        border-radius: 6px;
        padding: 7px 10px;
    }}

    QToolButton:hover, QPushButton:hover {{
        background: {PALETTE.panel_alt};
        border-color: #b7c5d2;
    }}

    QToolButton#ModuleButton:checked {{
        background: #e2f0fb;
        color: {PALETTE.accent_dark};
        border-color: {PALETTE.accent};
        font-weight: 700;
    }}

    QPushButton#PrimaryButton {{
        background: {PALETTE.accent};
        color: #ffffff;
        border-color: {PALETTE.accent_dark};
        font-weight: 600;
    }}

    QPushButton#PrimaryButton:hover {{
        background: {PALETTE.accent_dark};
    }}

    QPushButton#DangerButton {{
        color: {PALETTE.danger};
    }}

    QGroupBox {{
        background: {PALETTE.panel};
        border: 1px solid {PALETTE.border};
        border-radius: 8px;
        margin-top: 18px;
        padding: 14px 10px 10px 10px;
        font-weight: 700;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }}

    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
        background: #ffffff;
        border: 1px solid {PALETTE.border};
        border-radius: 5px;
        padding: 5px 7px;
        min-height: 22px;
    }}

    QCheckBox {{
        spacing: 8px;
    }}

    QProgressBar {{
        background: #e7edf3;
        border: 1px solid {PALETTE.border};
        border-radius: 5px;
        height: 12px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background: {PALETTE.accent};
        border-radius: 4px;
    }}

    QTableWidget {{
        background: #ffffff;
        border: 1px solid {PALETTE.border};
        border-radius: 6px;
        gridline-color: #edf1f5;
        selection-background-color: #d9ecff;
    }}

    QTextEdit, QPlainTextEdit {{
        background: #101822;
        color: #dbe7f2;
        border: none;
        border-radius: 8px;
        font-family: Consolas, monospace;
        font-size: 9pt;
    }}
    """
