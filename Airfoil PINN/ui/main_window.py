"""Main window for the PySide6 UI branch."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QPointF, QSettings, QSize, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from airfoil_pinn.app.paths import for_project_root
from ui.geometry_frontend import GeometryFrontendSession, GeometryReadError
from ui.theme import PALETTE
from ui.workers import BackendWorker, ResultsScanWorker
from ui.widgets.geometry_canvas import GeometryCanvas
from ui.widgets.log_panel import LogPanel
from ui.widgets.metric_plot import MetricPlot
from ui.widgets.result_canvas import ResultImageCanvas


class MainWindow(QMainWindow):
    """Main desktop window for geometry setup, training launch, and results."""

    GEOMETRY_FILE_EXTENSIONS = {".txt", ".dat", ".csv", ".npy"}
    MODULES = (
        ("geometry", "Geometry", QStyle.StandardPixmap.SP_FileDialogDetailedView),
        ("collocation", "Collocation", QStyle.StandardPixmap.SP_DriveHDIcon),
        ("fluid", "Fluid", QStyle.StandardPixmap.SP_ComputerIcon),
        ("boundaries", "Boundaries", QStyle.StandardPixmap.SP_DirLinkIcon),
        ("initialization", "Initialization", QStyle.StandardPixmap.SP_BrowserReload),
        ("training", "Training", QStyle.StandardPixmap.SP_MediaPlay),
        ("results", "Results", QStyle.StandardPixmap.SP_DialogSaveButton),
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Airfoil PINN - Backend Training UI")
        self.resize(1320, 840)
        self.setAcceptDrops(True)
        self.project_root = Path(__file__).resolve().parents[1]
        self.workspace_paths = for_project_root(self.project_root)
        self.default_results_root = self.workspace_paths.experiments_dir / "ui_runs"
        self.current_results_dir = self.default_results_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.settings = QSettings("AirfoilPINN", "AirfoilPINNUI")
        self.backend_worker: BackendWorker | None = None
        self.training_thread: QThread | None = None
        self.training_running = False
        self.final_metrics: dict[str, Any] = {}
        self._pending_log_lines: list[str] = []
        self._pending_metrics: dict[str, Any] = {}
        self._active_results_scans: list[QThread] = []
        self._results_scan_workers: dict[QThread, ResultsScanWorker] = {}
        self._results_scan_generation = 0
        self.geometry_session = GeometryFrontendSession(project_root=self.project_root)
        self.external_geometry_path: Path | None = None
        self.manual_coordinates_text = ""
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(75)
        self._log_flush_timer.timeout.connect(self._flush_backend_logs)
        self._metrics_flush_timer = QTimer(self)
        self._metrics_flush_timer.setInterval(120)
        self._metrics_flush_timer.timeout.connect(self._flush_backend_metrics)

        self._create_actions()
        self._create_menu()
        self._create_toolbar()
        self._build_layout()
        self._connect_signals()
        self._apply_widget_hints()
        self._refresh_foil_files()
        self._refresh_library_names()
        self._apply_domain_params_to_controls()
        self._apply_save_format_to_controls()
        self._sync_geometry_ui(reset_view=True, update_name=True)
        self._restore_ui_settings()
        initial_module = str(self.settings.value("window/active_module", "training"))
        self._set_active_module(initial_module if initial_module in self.module_index else "training")
        self._log_flush_timer.start()
        self._metrics_flush_timer.start()

        self.log_panel.append_event("Backend Training UI started")
        self.statusBar().showMessage("Backend UI ready")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._mime_has_geometry_payload(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if self._handle_geometry_drop(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802
        if event.type() in {QEvent.Type.DragEnter, QEvent.Type.DragMove} and hasattr(event, "mimeData"):
            if self._mime_has_geometry_payload(event.mimeData()):
                event.acceptProposedAction()
                return True
        if event.type() == QEvent.Type.Drop and hasattr(event, "mimeData"):
            if self._handle_geometry_drop(event.mimeData()):
                event.acceptProposedAction()
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_ui_settings()
        self._flush_backend_logs()
        self._flush_backend_metrics()
        if self.training_running:
            reply = QMessageBox.question(
                self,
                "Training is running",
                "Stop the backend process and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_training()
            if self.training_thread is not None:
                self.training_thread.quit()
                self.training_thread.wait(3000)
        for thread in list(self._active_results_scans):
            thread.quit()
            thread.wait(1000)
        super().closeEvent(event)

    def _create_actions(self) -> None:
        style = QApplication.style()
        self.open_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open", self)
        self.save_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self.load_config_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Load training config",
            self,
        )
        self.save_config_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save training config",
            self,
        )
        self.open_results_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            "Open results folder",
            self,
        )
        self.load_results_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Load results folder",
            self,
        )
        self.refresh_results_action = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
            "Refresh results",
            self,
        )
        self.exit_action = QAction("Exit", self)
        self.reset_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Reset view", self)
        self.start_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Start", self)
        self.stop_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop", self)
        self.info_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation), "About", self)

        self.open_action.setToolTip("Load airfoil file")
        self.save_action.setToolTip("Save geometry artifacts")
        self.load_config_action.setToolTip("Load training configuration JSON")
        self.save_config_action.setToolTip("Save training configuration JSON")
        self.open_results_action.setToolTip("Open current results folder")
        self.load_results_action.setToolTip("Load an existing results folder")
        self.refresh_results_action.setToolTip("Refresh result images")
        self.reset_action.setToolTip("Reset geometry view")
        self.start_action.setToolTip("Start backend training")
        self.stop_action.setToolTip("Stop backend training")
        self.info_action.setToolTip("Show application details")
        self.stop_action.setEnabled(False)

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.load_config_action)
        file_menu.addAction(self.save_config_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_results_action)
        file_menu.addAction(self.open_results_action)
        file_menu.addAction(self.refresh_results_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(self.info_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.reset_action)
        toolbar.addSeparator()
        toolbar.addAction(self.start_action)
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        toolbar.addAction(self.info_action)
        self.addToolBar(toolbar)

    def _build_layout(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        root.addWidget(self._build_module_nav())

        body = QSplitter(Qt.Orientation.Horizontal)
        self.left_module_stack = QStackedWidget()
        self.right_module_stack = QStackedWidget()
        self.module_index: dict[str, int] = {}

        for module_key, left_panel, right_panel in (
            ("geometry", self._build_geometry_left_panel(), self._build_geometry_right_panel()),
            ("collocation", self._build_collocation_left_panel(), self._build_collocation_right_panel()),
            ("fluid", self._build_fluid_left_panel(), self._build_fluid_right_panel()),
            ("boundaries", self._build_boundary_left_panel(), self._build_boundary_right_panel()),
            ("initialization", self._build_initialization_left_panel(), self._build_initialization_right_panel()),
            ("training", self._build_training_left_panel(), self._build_training_right_panel()),
            ("results", self._build_results_left_panel(), self._build_results_right_panel()),
        ):
            self.module_index[module_key] = self.left_module_stack.addWidget(left_panel)
            self.right_module_stack.addWidget(right_panel)

        body.addWidget(self.left_module_stack)
        body.addWidget(self._build_geometry_center())
        body.addWidget(self.right_module_stack)
        body.setSizes([310, 780, 330])
        body.setCollapsible(1, False)

        workspace = QSplitter(Qt.Orientation.Vertical)
        workspace.addWidget(body)
        self.log_panel = LogPanel()
        workspace.addWidget(self.log_panel)
        workspace.setSizes([690, 140])
        root.addWidget(workspace, 1)
        self.setCentralWidget(central)

    def _build_module_nav(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Panel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        title = QLabel("AEROPINN Backend Training UI")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        mode = QLabel("subprocess runner")
        mode.setObjectName("Muted")
        layout.addWidget(mode)

        self.module_button_group = QButtonGroup(self)
        self.module_button_group.setExclusive(True)
        self.module_buttons: dict[str, QToolButton] = {}
        style = QApplication.style()

        for module_key, label, icon_id in self.MODULES:
            button = QToolButton()
            button.setObjectName("ModuleButton")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIcon(style.standardIcon(icon_id))
            button.setText(label)
            button.setMinimumWidth(118)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked=False, key=module_key: self._set_active_module(key))
            self.module_button_group.addButton(button)
            self.module_buttons[module_key] = button
            layout.addWidget(button, 1)

        return frame

    def _set_active_module(self, module_key: str) -> None:
        index = self.module_index[module_key]
        self.left_module_stack.setCurrentIndex(index)
        self.right_module_stack.setCurrentIndex(index)
        self.module_buttons[module_key].setChecked(True)
        self.statusBar().showMessage(f"{self.module_buttons[module_key].text()} settings")

    def _build_geometry_center(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_geometry_header())
        self.geometry_canvas = GeometryCanvas()
        layout.addWidget(self.geometry_canvas, 1)
        return page

    def _new_side_panel(self):
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        panel.setWidget(content)
        panel.setMinimumWidth(285)
        return panel, layout

    def _build_geometry_left_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()

        domain_box = QGroupBox("Domain")
        form = QFormLayout(domain_box)
        self.domain_mode_label = QLabel("Detected: closed airfoil")
        self.domain_mode_label.setObjectName("Muted")
        self.front_spin = self._double_spin(0.2, 8.0, 1.5, 0.1)
        self.rear_spin = self._double_spin(0.2, 10.0, 3.0, 0.1)
        self.top_spin = self._double_spin(0.2, 5.0, 1.2, 0.1)
        self.bottom_spin = self._double_spin(0.2, 5.0, 1.2, 0.1)
        form.addRow("Profile", self.domain_mode_label)
        form.addRow("Front, m", self.front_spin)
        form.addRow("Rear, m", self.rear_spin)
        form.addRow("Top, m", self.top_spin)
        form.addRow("Bottom, m", self.bottom_spin)
        layout.addWidget(domain_box)

        display_box = QGroupBox("Display")
        display_layout = QVBoxLayout(display_box)
        self.points_check = QCheckBox("Collocation points")
        self.points_check.setChecked(False)
        self.flow_check = QCheckBox("Flow animation")
        self.flow_check.setChecked(False)
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        display_layout.addWidget(self.points_check)
        display_layout.addWidget(self.flow_check)
        display_layout.addWidget(self.grid_check)
        layout.addWidget(display_box)

        points_box = QGroupBox("Preview Points")
        points_form = QFormLayout(points_box)
        self.points_spin = QSpinBox()
        self.points_spin.setRange(100, 6000)
        self.points_spin.setSingleStep(100)
        self.points_spin.setValue(850)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Uniform", "Boundary focused", "RAR preview"])
        points_form.addRow("Points", self.points_spin)
        points_form.addRow("Method", self.method_combo)
        layout.addWidget(points_box)

        button_row = QHBoxLayout()
        self.rebuild_button = QPushButton("Build Domain")
        self.rebuild_button.setObjectName("PrimaryButton")
        self.sample_button = QPushButton("Sample")
        self.rebuild_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.sample_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        button_row.addWidget(self.rebuild_button)
        button_row.addWidget(self.sample_button)
        layout.addLayout(button_row)

        self.geometry_info = QLabel("x: 0.000, y: 0.000")
        self.geometry_info.setObjectName("Muted")
        layout.addWidget(self.geometry_info)
        layout.addStretch(1)
        return panel

    def _build_geometry_right_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()

        self.stats_table = QTableWidget(8, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._setup_compact_table(self.stats_table, height=236)
        layout.addWidget(self._wrap_group("Airfoil Stats", self.stats_table))

        self.vertices_table = QTableWidget(0, 2)
        self.vertices_table.setHorizontalHeaderLabels(["x, m", "y, m"])
        self._setup_compact_table(self.vertices_table, height=180)
        layout.addWidget(self._wrap_group("Domain Vertices", self.vertices_table))

        export_box = QGroupBox("Export")
        export_form = QFormLayout(export_box)
        self.geometry_name_edit = QLineEdit(self.geometry_session.name)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["npy", "csv", "txt"])
        self.export_geometry_button = QPushButton("Save")
        self.export_geometry_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.export_geometry_button.setObjectName("PrimaryButton")
        self.output_dir_label = QLabel(f"Output: {self._short_path(self.geometry_session.output_dir)}")
        self.output_dir_label.setObjectName("Muted")
        self.output_dir_label.setWordWrap(True)
        export_form.addRow("Name", self.geometry_name_edit)
        export_form.addRow("Format", self.save_format_combo)
        export_form.addRow("Folder", self.output_dir_label)
        export_form.addRow("", self.export_geometry_button)
        layout.addWidget(export_box)
        layout.addStretch(1)
        return panel

    def _build_form_side_panel(self, title: str, fields: list[tuple[str, QWidget]], note: str = "") -> QWidget:
        panel, layout = self._new_side_panel()
        box = QGroupBox(title)
        form = QFormLayout(box)
        for label, widget in fields:
            form.addRow(label, widget)
        layout.addWidget(box)
        if note:
            note_label = QLabel(note)
            note_label.setObjectName("Muted")
            note_label.setWordWrap(True)
            layout.addWidget(note_label)
        layout.addStretch(1)
        return panel

    def _build_status_side_panel(self, title: str, lines: list[str]) -> QWidget:
        panel, layout = self._new_side_panel()
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        for line in lines:
            label = QLabel(line)
            label.setObjectName("Muted")
            label.setWordWrap(True)
            box_layout.addWidget(label)
        layout.addWidget(box)
        layout.addStretch(1)
        return panel

    def _build_collocation_left_panel(self) -> QWidget:
        return self._build_form_side_panel(
            "Collocation",
            [
                ("Grid step, m", self._double_spin(0.001, 1.0, 0.04, 0.005)),
                ("Boundary density", self._double_spin(0.1, 10.0, 2.5, 0.1)),
                ("Boundary mode", self._combo(["inlet/outlet/wall/symmetry", "axisymmetric", "custom labels"])),
                ("Multiprocessing", self._combo(["auto", "1 process", "4 processes", "8 processes"])),
            ],
        )

    def _build_collocation_right_panel(self) -> QWidget:
        return self._build_status_side_panel(
            "Collocation Preview",
            ["Geometry remains visible in the center.", "Sampler settings will be connected to backend later."],
        )

    def _build_fluid_left_panel(self) -> QWidget:
        return self._build_form_side_panel(
            "Fluid",
            [
                ("Preset", self._combo(["Air 20C", "Water 20C", "Custom"])),
                ("Density, kg/m3", self._double_spin(0.1, 5000.0, 1.204, 0.01)),
                ("Kinematic viscosity, m2/s", self._double_spin(1e-8, 1e-2, 1.5e-5, 1e-6, decimals=8)),
                ("Dynamic viscosity, Pa*s", self._double_spin(1e-8, 1e-1, 1.8e-5, 1e-6, decimals=8)),
            ],
        )

    def _build_fluid_right_panel(self) -> QWidget:
        return self._build_status_side_panel("Fluid Status", ["Fluid properties panel.", "Geometry view stays active."])

    def _build_boundary_left_panel(self) -> QWidget:
        return self._build_form_side_panel(
            "Boundaries",
            [
                ("Inlet velocity, m/s", self._double_spin(0.0, 400.0, 30.0, 0.5)),
                ("Outlet pressure, Pa", self._double_spin(-1e6, 1e6, 0.0, 100.0)),
                ("Wall", self._combo(["no slip", "slip", "moving wall"])),
                ("Top boundary", self._combo(["symmetry", "outlet", "wall"])),
                ("Bottom boundary", self._combo(["wall", "symmetry", "axis symmetry"])),
            ],
        )

    def _build_boundary_right_panel(self) -> QWidget:
        return self._build_status_side_panel("Boundary Summary", ["Boundary labels will map onto the visible geometry domain."])

    def _build_initialization_left_panel(self) -> QWidget:
        return self._build_form_side_panel(
            "Initialization",
            [
                ("Pressure mode", self._combo(["constant", "dynamic pressure", "linear"])),
                ("Velocity mode", self._combo(["uniform", "potential preview", "noisy uniform"])),
                ("Velocity noise", self._double_spin(0.0, 1.0, 0.01, 0.01)),
                ("Pressure noise", self._double_spin(0.0, 1.0, 0.0, 0.01)),
                ("Temperature, K", self._double_spin(100.0, 600.0, 293.15, 0.5)),
            ],
        )

    def _build_initialization_right_panel(self) -> QWidget:
        return self._build_status_side_panel("Initialization Status", ["Initial fields will be previewed over the geometry."])

    def _build_training_left_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()

        profile_box = QGroupBox("Airfoil")
        profile_form = QFormLayout(profile_box)
        self.training_profile_combo = QComboBox()
        self.training_profile_combo.addItems([
            "Current UI geometry",
            "NACA 0012",
            "NACA 2412",
            "RAE 2822",
            "Custom coordinate file",
        ])
        self.custom_profile_edit = QLineEdit()
        self.custom_profile_edit.setPlaceholderText("Optional coordinate file")
        self.browse_profile_button = QPushButton("Browse")
        self.browse_profile_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        profile_row = QWidget()
        profile_row_layout = QHBoxLayout(profile_row)
        profile_row_layout.setContentsMargins(0, 0, 0, 0)
        profile_row_layout.addWidget(self.custom_profile_edit, 1)
        profile_row_layout.addWidget(self.browse_profile_button)
        profile_form.addRow("Preset", self.training_profile_combo)
        profile_form.addRow("Coordinates", profile_row)
        layout.addWidget(profile_box)

        backend_box = QGroupBox("Backend")
        backend_form = QFormLayout(backend_box)
        self.backend_mode_combo = QComboBox()
        self.backend_mode_combo.addItems(["Airfoil_PINN.py", "Mock backend"])
        self.base_config_combo = QComboBox()
        for config_path in self._available_backend_configs():
            self.base_config_combo.addItem(self._short_project_path(config_path), str(config_path))
        if self.base_config_combo.count() == 0:
            self.base_config_combo.addItem("No YAML configs found", "")
        self.backend_script_edit = QLineEdit(str(self.project_root / "Airfoil_PINN.py"))
        self.browse_backend_button = QPushButton("Browse")
        self.browse_backend_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        backend_script_row = QWidget()
        backend_script_layout = QHBoxLayout(backend_script_row)
        backend_script_layout.setContentsMargins(0, 0, 0, 0)
        backend_script_layout.addWidget(self.backend_script_edit, 1)
        backend_script_layout.addWidget(self.browse_backend_button)
        backend_form.addRow("Mode", self.backend_mode_combo)
        backend_form.addRow("Base config", self.base_config_combo)
        backend_form.addRow("Script", backend_script_row)
        layout.addWidget(backend_box)

        training_box = QGroupBox("Training")
        training_form = QFormLayout(training_box)
        self.epochs_spin = self._spin(1, 1_000_000, 500, 100)
        self.batch_size_spin = self._spin(1, 1_000_000, 1000, 100)
        self.learning_rate_spin = self._double_spin(1e-8, 1.0, 0.001, 0.0001, decimals=8)
        self.collocation_points_spin = self._spin(100, 10_000_000, 5000, 100)
        self.device_combo = self._combo(["auto", "cuda", "cpu"])
        training_form.addRow("Epochs", self.epochs_spin)
        training_form.addRow("Learning rate", self.learning_rate_spin)
        training_form.addRow("Batch size", self.batch_size_spin)
        training_form.addRow("Collocation points", self.collocation_points_spin)
        training_form.addRow("Device", self.device_combo)
        layout.addWidget(training_box)

        loss_box = QGroupBox("Loss Weights")
        loss_form = QFormLayout(loss_box)
        self.pde_weight_spin = self._double_spin(0.0, 1e8, 1.0, 0.1, decimals=4)
        self.bc_weight_spin = self._double_spin(0.0, 1e8, 1.0, 0.1, decimals=4)
        self.data_weight_spin = self._double_spin(0.0, 1e8, 1.0, 0.1, decimals=4)
        self.fvm_checkbox = QCheckBox("Use FVM residuals")
        self.fvm_checkbox.setChecked(True)
        loss_form.addRow("PDE", self.pde_weight_spin)
        loss_form.addRow("BC", self.bc_weight_spin)
        loss_form.addRow("Data", self.data_weight_spin)
        loss_form.addRow("", self.fvm_checkbox)
        layout.addWidget(loss_box)

        result_box = QGroupBox("Results")
        result_form = QFormLayout(result_box)
        self.results_dir_edit = QLineEdit(str(self.current_results_dir))
        self.browse_results_button = QPushButton("Browse")
        self.browse_results_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        result_row = QWidget()
        result_layout = QHBoxLayout(result_row)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(self.results_dir_edit, 1)
        result_layout.addWidget(self.browse_results_button)
        result_form.addRow("Folder", result_row)
        layout.addWidget(result_box)

        layout.addStretch(1)
        return panel

    def _build_training_right_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()
        metrics = QGridLayout()
        self.epoch_value = self._metric_label("0 / 0")
        self.loss_value = self._metric_label("n/a")
        self.pde_value = self._metric_label("n/a")
        self.bc_value = self._metric_label("n/a")
        self.data_value = self._metric_label("n/a")
        metrics.addWidget(self._metric_card("Epoch", self.epoch_value), 0, 0)
        metrics.addWidget(self._metric_card("Total loss", self.loss_value), 0, 1)
        metrics.addWidget(self._metric_card("PDE", self.pde_value), 1, 0)
        metrics.addWidget(self._metric_card("BC", self.bc_value), 1, 1)
        metrics.addWidget(self._metric_card("Data", self.data_value), 2, 0, 1, 2)
        layout.addLayout(metrics)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        layout.addWidget(self.progress)
        self.loss_plot = MetricPlot("Total loss", PALETTE.accent)
        self.pde_plot = MetricPlot("PDE", PALETTE.success)
        self.bc_plot = MetricPlot("Boundary", PALETTE.warning)
        layout.addWidget(self.loss_plot)
        layout.addWidget(self.pde_plot)
        layout.addWidget(self.bc_plot)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("PrimaryButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("DangerButton")
        self.refresh_results_button_train = QPushButton("Results")
        style = QApplication.style()
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.refresh_results_button_train.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.refresh_results_button_train)
        layout.addLayout(controls)
        layout.addStretch(1)
        return panel

    def _build_results_left_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()

        folder_box = QGroupBox("Results Folder")
        folder_layout = QVBoxLayout(folder_box)
        self.results_folder_label = QLabel(self._short_project_path(self.current_results_dir))
        self.results_folder_label.setObjectName("Muted")
        self.results_folder_label.setWordWrap(True)
        folder_layout.addWidget(self.results_folder_label)

        button_row = QHBoxLayout()
        self.open_results_button = QPushButton("Open")
        self.load_results_button = QPushButton("Load")
        self.refresh_results_button = QPushButton("Refresh")
        style = QApplication.style()
        self.open_results_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.load_results_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.refresh_results_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        button_row.addWidget(self.open_results_button)
        button_row.addWidget(self.load_results_button)
        button_row.addWidget(self.refresh_results_button)
        folder_layout.addLayout(button_row)
        layout.addWidget(folder_box)

        self.result_files_list = QListWidget()
        self.result_files_list.setMinimumHeight(180)
        layout.addWidget(self._wrap_group("Plots", self.result_files_list))

        self.final_metrics_table = QTableWidget(0, 2)
        self.final_metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._setup_compact_table(self.final_metrics_table, height=220)
        layout.addWidget(self._wrap_group("Final Losses", self.final_metrics_table))

        layout.addStretch(1)
        return panel

    def _build_results_right_panel(self) -> QWidget:
        panel, layout = self._new_side_panel()
        self.result_canvas = ResultImageCanvas()
        self.result_status_label = QLabel("No results loaded")
        self.result_status_label.setObjectName("Muted")
        self.result_status_label.setWordWrap(True)
        layout.addWidget(self.result_canvas)
        layout.addWidget(self.result_status_label)
        layout.addStretch(1)
        return panel

    def _build_geometry_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Panel")
        layout = QGridLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(5)

        title_label = QLabel("Geometry")
        title_label.setObjectName("SectionTitle")
        source_title = QLabel("Airfoil Source")
        source_title.setObjectName("Muted")

        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "workspace/geometry/foils",
            "External file",
            "Manual input",
            "workspace/geometry/geom_config.yaml",
            "Foil library",
            "Built-in closed NACA 0012",
            "Built-in NACA 2412",
            "Built-in RAE 2822",
            "Built-in open plate",
        ])
        self.foil_file_combo = QComboBox()
        self.library_combo = QComboBox()
        self.browse_geometry_button = QPushButton("Browse")
        self.manual_geometry_button = QPushButton("Manual Input")
        self.load_geometry_button = QPushButton("Load")
        self.load_geometry_button.setObjectName("PrimaryButton")
        self.manual_geometry_button.setObjectName("PrimaryButton")
        self.browse_geometry_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.manual_geometry_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.load_geometry_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))

        self.source_path_label = QLabel("Source: not loaded")
        self.source_path_label.setObjectName("Muted")
        self.source_path_label.setWordWrap(True)

        self.source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.foil_file_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.library_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_combo.setMinimumWidth(170)
        self.foil_file_combo.setMinimumWidth(180)
        self.library_combo.setMinimumWidth(180)
        self.browse_geometry_button.setMinimumWidth(96)
        self.manual_geometry_button.setMinimumWidth(128)
        self.load_geometry_button.setMinimumWidth(86)

        layout.addWidget(title_label, 0, 0)
        layout.addWidget(source_title, 0, 1)
        layout.addWidget(self.source_combo, 0, 2)
        layout.addWidget(self.foil_file_combo, 0, 3)
        layout.addWidget(self.library_combo, 0, 3)
        layout.addWidget(self.browse_geometry_button, 0, 4)
        layout.addWidget(self.manual_geometry_button, 0, 4)
        layout.addWidget(self.load_geometry_button, 0, 5)
        layout.addWidget(self.source_path_label, 1, 2, 1, 4)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 2)
        layout.setColumnStretch(3, 3)
        layout.setColumnStretch(4, 1)
        layout.setColumnStretch(5, 1)
        return frame

    def _metric_card(self, title: str, value_label: QLabel) -> QWidget:
        card = QFrame()
        card.setObjectName("Panel")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        label = QLabel(title)
        label.setObjectName("Muted")
        layout.addWidget(label)
        layout.addWidget(value_label)
        return card

    def _metric_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("MetricValue")
        return label

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 3,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _spin(self, minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _combo(self, values: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        return combo

    def _connect_signals(self) -> None:
        for drop_target in (self.geometry_canvas, self.left_module_stack, self.right_module_stack):
            drop_target.setAcceptDrops(True)
            drop_target.installEventFilter(self)

        self.geometry_canvas.cursorChanged.connect(self._update_cursor_label)
        self.geometry_canvas.selectionChanged.connect(self.statusBar().showMessage)
        self.geometry_canvas.geometryEdited.connect(self._handle_canvas_geometry_edited)

        self.source_combo.currentIndexChanged.connect(self._update_source_controls)
        self.browse_geometry_button.clicked.connect(self._browse_geometry_file)
        self.load_geometry_button.clicked.connect(self._load_geometry_source)
        self.manual_geometry_button.clicked.connect(self._open_manual_coordinates_dialog)
        self.rebuild_button.clicked.connect(self._rebuild_geometry)
        self.sample_button.clicked.connect(self._sample_points)
        self.export_geometry_button.clicked.connect(self._export_geometry)
        self.reset_action.triggered.connect(self.geometry_canvas.reset_view)
        self.points_check.toggled.connect(self.geometry_canvas.set_points_visible)
        self.flow_check.toggled.connect(self.geometry_canvas.set_streamlines_visible)
        self.grid_check.toggled.connect(self.geometry_canvas.set_grid_visible)

        self.open_action.triggered.connect(self._mock_open)
        self.save_action.triggered.connect(self._mock_save)
        self.info_action.triggered.connect(self._show_about)
        self.load_config_action.triggered.connect(self._load_training_config)
        self.save_config_action.triggered.connect(self._save_training_config)
        self.exit_action.triggered.connect(self.close)
        self.open_results_action.triggered.connect(self._open_results_folder)
        self.load_results_action.triggered.connect(self._choose_results_folder)
        self.refresh_results_action.triggered.connect(self._refresh_results)
        self.start_action.triggered.connect(self._start_training)
        self.stop_action.triggered.connect(self._stop_training)
        self.start_button.clicked.connect(self._start_training)
        self.stop_button.clicked.connect(self._stop_training)
        self.refresh_results_button_train.clicked.connect(self._refresh_results)
        self.browse_profile_button.clicked.connect(self._browse_training_profile_file)
        self.browse_backend_button.clicked.connect(self._browse_backend_script)
        self.browse_results_button.clicked.connect(self._browse_results_dir)
        self.backend_mode_combo.currentIndexChanged.connect(self._sync_backend_controls)
        self.training_profile_combo.currentIndexChanged.connect(self._sync_training_profile_controls)
        self.open_results_button.clicked.connect(self._open_results_folder)
        self.load_results_button.clicked.connect(self._choose_results_folder)
        self.refresh_results_button.clicked.connect(self._refresh_results)
        self.result_files_list.currentItemChanged.connect(self._show_selected_result_image)

        self._sync_backend_controls()
        self._sync_training_profile_controls()

    def _apply_widget_hints(self) -> None:
        self.front_spin.setToolTip("Upstream domain length before the airfoil")
        self.rear_spin.setToolTip("Downstream domain length after a closed airfoil")
        self.top_spin.setToolTip("Upper domain clearance")
        self.bottom_spin.setToolTip("Lower domain clearance for closed airfoils")
        self.points_spin.setToolTip("Number of preview points drawn in the UI canvas")
        self.collocation_points_spin.setToolTip("Saved in the UI config; real backend uses data.points_file from YAML")
        self.base_config_combo.setToolTip("YAML config passed to Airfoil_PINN.py")
        self.backend_script_edit.setToolTip("Console backend script launched through subprocess")
        self.results_dir_edit.setToolTip("Run folder for backend config, UI config, logs, plots, and checkpoints")
        self.fvm_checkbox.setToolTip("Requires data.mesh_file in the selected backend YAML")
        self.custom_profile_edit.setToolTip("Coordinate file for UI geometry preview")

    def _restore_ui_settings(self) -> None:
        self.resize(
            self._settings_int("window/width", self.width()),
            self._settings_int("window/height", self.height()),
        )

        self.front_spin.setValue(self._settings_float("geometry/front", self.front_spin.value()))
        self.rear_spin.setValue(self._settings_float("geometry/rear", self.rear_spin.value()))
        self.top_spin.setValue(self._settings_float("geometry/top", self.top_spin.value()))
        self.bottom_spin.setValue(self._settings_float("geometry/bottom", self.bottom_spin.value()))
        self.points_spin.setValue(self._settings_int("geometry/preview_points", self.points_spin.value()))
        self.points_check.setChecked(self._settings_bool("geometry/show_points", self.points_check.isChecked()))
        self.flow_check.setChecked(self._settings_bool("geometry/show_flow", self.flow_check.isChecked()))
        self.grid_check.setChecked(self._settings_bool("geometry/show_grid", self.grid_check.isChecked()))
        self._set_combo_text(self.save_format_combo, self.settings.value("geometry/save_format", None))
        try:
            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=True, update_name=False)
        except GeometryReadError:
            pass

        self._set_combo_text(self.backend_mode_combo, self.settings.value("training/backend_mode", None))
        self._set_combo_data(self.base_config_combo, self.settings.value("training/base_config", None))
        backend_script = self.settings.value("training/backend_script", None)
        if backend_script:
            self.backend_script_edit.setText(str(backend_script))
        self._set_combo_text(self.training_profile_combo, self.settings.value("training/profile", None))
        custom_profile = self.settings.value("training/custom_profile", None)
        if custom_profile:
            self.custom_profile_edit.setText(str(custom_profile))
        results_dir = self.settings.value("training/results_dir", None)
        if results_dir:
            self.results_dir_edit.setText(str(results_dir))
        self.epochs_spin.setValue(self._settings_int("training/epochs", self.epochs_spin.value()))
        self.batch_size_spin.setValue(self._settings_int("training/batch_size", self.batch_size_spin.value()))
        self.collocation_points_spin.setValue(
            self._settings_int("training/collocation_points", self.collocation_points_spin.value())
        )
        self.learning_rate_spin.setValue(
            self._settings_float("training/learning_rate", self.learning_rate_spin.value())
        )
        self._set_combo_text(self.device_combo, self.settings.value("training/device", None))
        self.pde_weight_spin.setValue(self._settings_float("training/pde_weight", self.pde_weight_spin.value()))
        self.bc_weight_spin.setValue(self._settings_float("training/bc_weight", self.bc_weight_spin.value()))
        self.data_weight_spin.setValue(self._settings_float("training/data_weight", self.data_weight_spin.value()))
        self.fvm_checkbox.setChecked(self._settings_bool("training/use_fvm", self.fvm_checkbox.isChecked()))

        self._sync_backend_controls()
        self._sync_training_profile_controls()

    def _save_ui_settings(self) -> None:
        self.settings.setValue("window/width", self.width())
        self.settings.setValue("window/height", self.height())
        active_module = next(
            (key for key, button in self.module_buttons.items() if button.isChecked()),
            "training",
        )
        self.settings.setValue("window/active_module", active_module)

        self.settings.setValue("geometry/front", self.front_spin.value())
        self.settings.setValue("geometry/rear", self.rear_spin.value())
        self.settings.setValue("geometry/top", self.top_spin.value())
        self.settings.setValue("geometry/bottom", self.bottom_spin.value())
        self.settings.setValue("geometry/preview_points", self.points_spin.value())
        self.settings.setValue("geometry/show_points", self.points_check.isChecked())
        self.settings.setValue("geometry/show_flow", self.flow_check.isChecked())
        self.settings.setValue("geometry/show_grid", self.grid_check.isChecked())
        self.settings.setValue("geometry/save_format", self.save_format_combo.currentText())

        self.settings.setValue("training/backend_mode", self.backend_mode_combo.currentText())
        self.settings.setValue("training/base_config", self.base_config_combo.currentData() or "")
        self.settings.setValue("training/backend_script", self.backend_script_edit.text())
        self.settings.setValue("training/profile", self.training_profile_combo.currentText())
        self.settings.setValue("training/custom_profile", self.custom_profile_edit.text())
        self.settings.setValue("training/results_dir", self.results_dir_edit.text())
        self.settings.setValue("training/epochs", self.epochs_spin.value())
        self.settings.setValue("training/batch_size", self.batch_size_spin.value())
        self.settings.setValue("training/collocation_points", self.collocation_points_spin.value())
        self.settings.setValue("training/learning_rate", self.learning_rate_spin.value())
        self.settings.setValue("training/device", self.device_combo.currentText())
        self.settings.setValue("training/pde_weight", self.pde_weight_spin.value())
        self.settings.setValue("training/bc_weight", self.bc_weight_spin.value())
        self.settings.setValue("training/data_weight", self.data_weight_spin.value())
        self.settings.setValue("training/use_fvm", self.fvm_checkbox.isChecked())
        self.settings.sync()

    def _settings_int(self, key: str, default: int) -> int:
        try:
            return int(self.settings.value(key, default))
        except (TypeError, ValueError):
            return default

    def _settings_float(self, key: str, default: float) -> float:
        try:
            return float(self.settings.value(key, default))
        except (TypeError, ValueError):
            return default

    def _settings_bool(self, key: str, default: bool) -> bool:
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

    def _setup_compact_table(self, table: QTableWidget, height: int) -> None:
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        table.setFixedHeight(height)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def _wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _refresh_foil_files(self) -> None:
        self.foil_file_combo.clear()
        files = self.geometry_session.available_foil_files()
        if not files:
            self.foil_file_combo.addItem("No files found", "")
        for path in files:
            self.foil_file_combo.addItem(path.name, str(path))
        self._update_source_controls()

    def _refresh_library_names(self) -> None:
        self.library_combo.clear()
        try:
            names = self.geometry_session.available_library_names()
        except GeometryReadError as exc:
            self.library_combo.addItem(str(exc), "")
            return
        if not names:
            self.library_combo.addItem("No workspace/geometry/foil_geoms.yaml entries", "")
        for name in names:
            self.library_combo.addItem(name, name)
        self._update_source_controls()

    def _update_source_controls(self) -> None:
        mode = self.source_combo.currentText()
        file_mode = mode == "workspace/geometry/foils"
        library_mode = mode == "Foil library"
        external_mode = mode == "External file"
        manual_mode = mode == "Manual input"

        self.foil_file_combo.setVisible(file_mode)
        self.library_combo.setVisible(library_mode)
        self.browse_geometry_button.setVisible(external_mode)
        self.manual_geometry_button.setVisible(manual_mode)
        self.load_geometry_button.setVisible(not manual_mode)

        self.foil_file_combo.setEnabled(file_mode)
        self.library_combo.setEnabled(library_mode)
        self.browse_geometry_button.setEnabled(external_mode)
        self.manual_geometry_button.setEnabled(manual_mode)
        self.load_geometry_button.setEnabled(not manual_mode)
        if mode != "External file":
            self.external_geometry_path = None

    def _mime_has_geometry_payload(self, mime_data) -> bool:
        if mime_data.hasUrls():
            return self._first_geometry_file_from_urls(mime_data.urls()) is not None
        return mime_data.hasText() and self._looks_like_coordinate_text(mime_data.text())

    def _handle_geometry_drop(self, mime_data) -> bool:
        if mime_data.hasUrls():
            path = self._first_geometry_file_from_urls(mime_data.urls())
            if path is None:
                return False
            self._load_geometry_file_path(path)
            return True

        if mime_data.hasText() and self._looks_like_coordinate_text(mime_data.text()):
            self.manual_coordinates_text = mime_data.text()
            self.source_combo.setCurrentText("Manual input")
            self._load_manual_coordinates(self.manual_coordinates_text)
            return True

        return False

    def _first_geometry_file_from_urls(self, urls) -> Path | None:
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in self.GEOMETRY_FILE_EXTENSIONS:
                return path
        return None

    def _looks_like_coordinate_text(self, text: str) -> bool:
        valid_rows = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").replace(";", " ").split()
            if len(parts) < 2:
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            valid_rows += 1
            if valid_rows >= 2:
                return True
        return False

    def _load_geometry_source(self) -> None:
        mode = self.source_combo.currentText()
        try:
            if mode == "Built-in closed NACA 0012":
                self.geometry_session.load_sample("closed_naca")
            elif mode == "Built-in NACA 2412":
                self.geometry_session.load_sample("naca2412")
            elif mode == "Built-in RAE 2822":
                self.geometry_session.load_sample("rae2822")
            elif mode == "Built-in open plate":
                self.geometry_session.load_sample("open_plate")
            elif mode == "workspace/geometry/foils":
                path = self.foil_file_combo.currentData()
                if not path:
                    raise GeometryReadError("workspace/geometry/foils does not contain supported files")
                self.geometry_session.load_file(path)
            elif mode == "Manual input":
                self._open_manual_coordinates_dialog()
                return
            elif mode == "workspace/geometry/geom_config.yaml":
                self.geometry_session.load_from_config()
                self._apply_domain_params_to_controls()
                self._apply_save_format_to_controls()
            elif mode == "Foil library":
                geom_name = self.library_combo.currentData()
                if not geom_name:
                    raise GeometryReadError("workspace/geometry/foil_geoms.yaml does not contain profiles")
                self.geometry_session.load_from_library(geom_name)
            else:
                if self.external_geometry_path is None:
                    self._browse_geometry_file()
                if self.external_geometry_path is None:
                    return
                self.geometry_session.load_file(self.external_geometry_path)

            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=True, update_name=True)
            self.log_panel.append_event(f"Loaded geometry: {self.geometry_session.name}")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Geometry load failed", str(exc))
            self.log_panel.append_event(f"Geometry load failed: {exc}")

    def _open_manual_coordinates_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Coordinates")
        dialog.resize(520, 380)

        layout = QVBoxLayout(dialog)
        hint = QLabel("Enter one point per line: x y, x,y, or x; y")
        hint.setObjectName("Muted")
        edit = QTextEdit()
        edit.setAcceptRichText(False)
        edit.setPlaceholderText("0.000 0.000\n0.500 0.060\n1.000 0.000")
        edit.setPlainText(self.manual_coordinates_text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(hint)
        layout.addWidget(edit, 1)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.manual_coordinates_text = edit.toPlainText()
            self._load_manual_coordinates(self.manual_coordinates_text)

    def _load_manual_coordinates(self, text: str) -> None:
        try:
            self.geometry_session.load_text_block(
                text=text,
                name=self.geometry_name_edit.text() or "manual_profile",
            )
            self._create_domain_from_controls()
            self.source_combo.setCurrentText("Manual input")
            self._sync_geometry_ui(reset_view=True, update_name=True)
            self.log_panel.append_event(f"Loaded manual coordinates: {self.geometry_session.name}")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Manual coordinates failed", str(exc))
            self.log_panel.append_event(f"Manual coordinates failed: {exc}")

    def _browse_geometry_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load airfoil",
            str(self.geometry_session.foils_dir),
            "Airfoil files (*.txt *.dat *.csv *.npy);;All files (*.*)",
        )
        if path:
            self._select_external_geometry_path(Path(path))

    def _select_external_geometry_path(self, path: Path) -> None:
        self.external_geometry_path = path
        self.source_combo.setCurrentText("External file")
        self.source_path_label.setText(f"Source: {self._short_path(self.external_geometry_path)}")

    def _load_geometry_file_path(self, path: Path) -> None:
        try:
            self._select_external_geometry_path(path)
            self.geometry_session.load_file(path)
            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=True, update_name=True)
            self.log_panel.append_event(f"Loaded dropped file: {self._short_path(path)}")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Geometry load failed", str(exc))
            self.log_panel.append_event(f"Geometry load failed: {exc}")

    def _create_domain_from_controls(self) -> None:
        self.geometry_session.create_domain(
            front=self.front_spin.value(),
            rear=self.rear_spin.value(),
            top=self.top_spin.value(),
            bottom=self.bottom_spin.value(),
        )

    def _apply_domain_params_to_controls(self) -> None:
        params = self.geometry_session.last_domain_params
        self.front_spin.setValue(params.get("front", self.front_spin.value()))
        self.rear_spin.setValue(params.get("rear", self.rear_spin.value()))
        self.top_spin.setValue(params.get("top", self.top_spin.value()))
        self.bottom_spin.setValue(params.get("bottom", self.bottom_spin.value()))

    def _apply_save_format_to_controls(self) -> None:
        fmt = self.geometry_session.default_save_format
        index = self.save_format_combo.findText(fmt)
        if index >= 0:
            self.save_format_combo.setCurrentIndex(index)

    def _sync_geometry_ui(self, reset_view: bool = False, update_name: bool = False) -> None:
        foil_points = [QPointF(point.x, point.y) for point in self.geometry_session.foil_points]
        domain_points = [QPointF(point.x, point.y) for point in self.geometry_session.domain_points]
        self.geometry_canvas.set_geometry(
            foil_points=foil_points,
            domain_points=domain_points,
            name=self.geometry_session.name,
            is_closed=self.geometry_session.is_closed,
        )
        if reset_view:
            self.geometry_canvas.reset_view()

        if update_name:
            self.geometry_name_edit.setText(self.geometry_session.name)

        self.domain_mode_label.setText(
            "Detected: closed airfoil" if self.geometry_session.is_closed else "Detected: open airfoil"
        )
        self.rear_spin.setEnabled(self.geometry_session.is_closed)
        self.bottom_spin.setEnabled(self.geometry_session.is_closed)
        self.source_path_label.setText(f"Source: {self._short_path(self.geometry_session.source)}")
        self.output_dir_label.setText(f"Output: {self._short_path(self.geometry_session.output_dir)}")
        self._update_stats_table()
        self._update_vertices_table()

    def _update_stats_table(self) -> None:
        stats = self.geometry_session.stats
        if stats is None:
            self.stats_table.setRowCount(0)
            return

        rows = [
            ("Name", self.geometry_session.name),
            ("Source", self._short_path(self.geometry_session.source)),
            ("Points", str(stats.point_count)),
            ("Closed", "yes" if stats.is_closed else "no"),
            ("Chord, m", f"{stats.chord:.6f}"),
            ("X range, m", f"{stats.x_min:.6f} .. {stats.x_max:.6f}"),
            ("Y range, m", f"{stats.y_min:.6f} .. {stats.y_max:.6f}"),
            ("Size, m", f"{stats.width:.6f} x {stats.height:.6f}"),
        ]
        self.stats_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.stats_table.setItem(row, 0, QTableWidgetItem(key))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))

    def _update_vertices_table(self) -> None:
        points = self.geometry_session.domain_points
        max_rows = 220
        if len(points) <= max_rows:
            visible_rows = [(point.x, point.y) for point in points]
        else:
            head = [(point.x, point.y) for point in points[:110]]
            tail = [(point.x, point.y) for point in points[-109:]]
            visible_rows = head + [(float("nan"), float("nan"))] + tail

        self.vertices_table.setRowCount(len(visible_rows))
        for row, (x_value, y_value) in enumerate(visible_rows):
            if math.isnan(x_value):
                self.vertices_table.setItem(row, 0, QTableWidgetItem("..."))
                self.vertices_table.setItem(row, 1, QTableWidgetItem(f"{len(points) - len(visible_rows) + 1} hidden"))
            else:
                self.vertices_table.setItem(row, 0, QTableWidgetItem(f"{x_value:.6f}"))
                self.vertices_table.setItem(row, 1, QTableWidgetItem(f"{y_value:.6f}"))

    def _export_geometry(self) -> None:
        try:
            artifacts = self.geometry_session.save(
                base_name=self.geometry_name_edit.text(),
                fmt=self.save_format_combo.currentText(),
            )
            self.log_panel.append_event(f"Saved foil: {self._short_path(artifacts.foil_path)}")
            self.log_panel.append_event(f"Saved domain: {self._short_path(artifacts.domain_path)}")
            self.log_panel.append_event(f"Saved metadata: {self._short_path(artifacts.metadata_path)}")
            self.statusBar().showMessage("Geometry artifacts saved")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Geometry save failed", str(exc))
            self.log_panel.append_event(f"Geometry save failed: {exc}")

    def _short_path(self, value: str | Path) -> str:
        if value == "mock":
            return "mock"
        path = Path(value)
        try:
            return str(path.relative_to(self.geometry_session.project_root))
        except (ValueError, RuntimeError):
            return str(path)

    def _update_cursor_label(self, x: float, y: float) -> None:
        self.geometry_info.setText(f"x: {x:.3f}, y: {y:.3f}")

    def _handle_canvas_geometry_edited(self, points: list[tuple[float, float]]) -> None:
        try:
            self.geometry_session.replace_foil_points(
                points,
                name=self.geometry_name_edit.text() or self.geometry_session.name,
                source="canvas edit",
            )
            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=False, update_name=False)
            self.log_panel.append_event("Geometry updated from canvas edit")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Geometry edit failed", str(exc))
            self.log_panel.append_event(f"Geometry edit failed: {exc}")

    def _rebuild_geometry(self) -> None:
        try:
            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=False)
            self.log_panel.append_event("Domain rebuilt in UI geometry model")
        except GeometryReadError as exc:
            QMessageBox.warning(self, "Domain build failed", str(exc))
            self.log_panel.append_event(f"Domain build failed: {exc}")

    def _sample_points(self) -> None:
        self.geometry_canvas.regenerate_points(self.points_spin.value())
        self.log_panel.append_event(f"Generated {self.points_spin.value()} mock collocation points")

    def _mock_open(self) -> None:
        self.source_combo.setCurrentText("External file")
        self._browse_geometry_file()
        if self.external_geometry_path:
            self._load_geometry_source()

    def _mock_save(self) -> None:
        self._export_geometry()

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "Airfoil PINN UI",
            "PySide6 desktop UI for Airfoil PINN.\n\n"
            "Training runs through subprocess.Popen in a QThread. "
            "Progress is parsed from backend stdout in real time.",
        )

    def _start_training(self) -> None:
        if self.training_running:
            return

        try:
            ui_config = self._collect_training_config()
            command, run_dir = self._build_backend_command(ui_config)
        except Exception as exc:
            QMessageBox.warning(self, "Training setup failed", str(exc))
            self.statusBar().showMessage("Training setup error")
            return

        self.current_results_dir = run_dir
        self.results_dir_edit.setText(str(run_dir))
        try:
            self._write_json(run_dir / "ui_training_config.json", ui_config)
        except OSError as exc:
            QMessageBox.warning(self, "Training setup failed", f"Could not write UI config: {exc}")
            self.statusBar().showMessage("Training setup error")
            return
        self._reset_training_monitor(ui_config["training"]["epochs"])

        self.training_thread = QThread(self)
        self.backend_worker = BackendWorker(
            command=command,
            cwd=self.project_root,
            expected_epochs=ui_config["training"]["epochs"],
        )
        self.backend_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.backend_worker.run)
        self.backend_worker.log_message.connect(self._handle_backend_log)
        self.backend_worker.progress.connect(self._handle_backend_progress)
        self.backend_worker.metrics_updated.connect(self._handle_backend_metrics)
        self.backend_worker.error.connect(self._handle_backend_error)
        self.backend_worker.finished.connect(self._handle_training_finished)
        self.backend_worker.finished.connect(self.training_thread.quit)
        self.backend_worker.finished.connect(self.backend_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.finished.connect(self._handle_training_thread_finished)

        self._set_training_running(True)
        self._set_active_module("training")
        self.log_panel.clear()
        self.log_panel.append_event(f"Training started: {ui_config['backend']['mode']}")
        self.log_panel.append_event(f"Results folder: {self._short_project_path(run_dir)}")
        self.training_thread.start()

    def _stop_training(self) -> None:
        if not self.backend_worker:
            return
        self.statusBar().showMessage("Stopping training...")
        self.stop_button.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.log_panel.append_event("Stop requested")
        self.backend_worker.request_stop()

    def _reset_training_monitor(self, total_epochs: int) -> None:
        self.final_metrics = {}
        self._pending_metrics.clear()
        self._pending_log_lines.clear()
        self.progress.setValue(0)
        self.epoch_value.setText(f"0 / {total_epochs}")
        self.loss_value.setText("n/a")
        self.pde_value.setText("n/a")
        self.bc_value.setText("n/a")
        self.data_value.setText("n/a")
        self.loss_plot.set_values([])
        self.pde_plot.set_values([])
        self.bc_plot.set_values([])

    def _set_training_running(self, running: bool) -> None:
        self.training_running = running
        self.start_button.setEnabled(not running)
        self.start_action.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.stop_action.setEnabled(running)
        self._set_training_inputs_enabled(not running)
        self.statusBar().showMessage("Training..." if running else "Ready")

    def _set_training_inputs_enabled(self, enabled: bool) -> None:
        widgets = (
            self.training_profile_combo,
            self.custom_profile_edit,
            self.browse_profile_button,
            self.backend_mode_combo,
            self.base_config_combo,
            self.backend_script_edit,
            self.browse_backend_button,
            self.epochs_spin,
            self.learning_rate_spin,
            self.batch_size_spin,
            self.collocation_points_spin,
            self.device_combo,
            self.pde_weight_spin,
            self.bc_weight_spin,
            self.data_weight_spin,
            self.fvm_checkbox,
            self.results_dir_edit,
            self.browse_results_button,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        if enabled:
            self._sync_backend_controls()
            self._sync_training_profile_controls()

    def _handle_backend_log(self, message: str) -> None:
        self._pending_log_lines.append(message)
        if len(self._pending_log_lines) >= 200:
            self._flush_backend_logs()

    def _handle_backend_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def _handle_backend_metrics(self, metrics: dict) -> None:
        self.final_metrics.update(metrics)
        self._pending_metrics.update(metrics)

    def _flush_backend_logs(self) -> None:
        if not self._pending_log_lines:
            return
        lines = self._pending_log_lines
        self._pending_log_lines = []
        self.log_panel.append_raw_lines(lines)

    def _flush_backend_metrics(self) -> None:
        if not self._pending_metrics:
            return
        metrics = dict(self._pending_metrics)
        self._pending_metrics.clear()
        self._apply_backend_metrics(metrics)

    def _apply_backend_metrics(self, metrics: dict[str, Any]) -> None:
        epoch = metrics.get("epoch")
        total_epochs = metrics.get("total_epochs") or self.epochs_spin.value()
        if epoch is not None:
            self.epoch_value.setText(f"{int(epoch)} / {int(total_epochs)}")

        if "loss" in metrics:
            loss = float(metrics["loss"])
            self.loss_value.setText(self._format_metric(loss))
            self.loss_plot.append_value(loss)
        if "pde_loss" in metrics:
            pde_loss = float(metrics["pde_loss"])
            self.pde_value.setText(self._format_metric(pde_loss))
            self.pde_plot.append_value(pde_loss)
        if "bc_loss" in metrics:
            bc_loss = float(metrics["bc_loss"])
            self.bc_value.setText(self._format_metric(bc_loss))
            self.bc_plot.append_value(bc_loss)
        if "data_loss" in metrics:
            self.data_value.setText(self._format_metric(float(metrics["data_loss"])))

    def _handle_backend_error(self, message: str) -> None:
        self.statusBar().showMessage("Training error")
        self.log_panel.append_event(f"ERROR: {message}")

    def _handle_training_finished(self, success: bool, message: str) -> None:
        self._flush_backend_logs()
        self._flush_backend_metrics()
        self._set_training_running(False)
        self.log_panel.append_event(message)
        self._refresh_results()
        self._set_active_module("results")

        if success:
            self.statusBar().showMessage("Training completed")
            return

        self.statusBar().showMessage("Training stopped or failed")
        if "stopped by user" not in message.lower():
            QMessageBox.warning(self, "Backend finished with error", message)

    def _handle_training_thread_finished(self) -> None:
        self.backend_worker = None
        self.training_thread = None

    def _collect_training_config(self) -> dict[str, Any]:
        result_dir = Path(self.results_dir_edit.text()).expanduser()
        if not result_dir.is_absolute():
            result_dir = self.project_root / result_dir
        result_dir.mkdir(parents=True, exist_ok=True)

        return {
            "profile": {
                "preset": self.training_profile_combo.currentText(),
                "coordinate_file": self.custom_profile_edit.text().strip(),
                "current_geometry": self.geometry_session.name,
            },
            "backend": {
                "mode": self.backend_mode_combo.currentText(),
                "script": self.backend_script_edit.text().strip(),
                "base_config": self.base_config_combo.currentData() or "",
            },
            "training": {
                "epochs": self.epochs_spin.value(),
                "learning_rate": self.learning_rate_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "collocation_points": self.collocation_points_spin.value(),
                "device": self.device_combo.currentText(),
            },
            "loss_weights": {
                "pde": self.pde_weight_spin.value(),
                "bc": self.bc_weight_spin.value(),
                "data": self.data_weight_spin.value(),
                "use_fvm": self.fvm_checkbox.isChecked(),
            },
            "results_dir": str(result_dir),
        }

    def _apply_training_config(self, config: dict[str, Any]) -> None:
        profile = config.get("profile", {})
        backend = config.get("backend", {})
        training = config.get("training", {})
        weights = config.get("loss_weights", {})

        self._set_combo_text(self.training_profile_combo, profile.get("preset"))
        self.custom_profile_edit.setText(str(profile.get("coordinate_file") or ""))
        self._set_combo_text(self.backend_mode_combo, backend.get("mode"))
        backend_script = Path(str(backend.get("script") or self.project_root / "Airfoil_PINN.py"))
        self.backend_script_edit.setText(str(backend_script))
        self._set_combo_data(self.base_config_combo, backend.get("base_config"))
        self.epochs_spin.setValue(int(training.get("epochs", self.epochs_spin.value())))
        self.learning_rate_spin.setValue(float(training.get("learning_rate", self.learning_rate_spin.value())))
        self.batch_size_spin.setValue(int(training.get("batch_size", self.batch_size_spin.value())))
        self.collocation_points_spin.setValue(
            int(training.get("collocation_points", self.collocation_points_spin.value()))
        )
        self._set_combo_text(self.device_combo, training.get("device"))
        self.pde_weight_spin.setValue(float(weights.get("pde", self.pde_weight_spin.value())))
        self.bc_weight_spin.setValue(float(weights.get("bc", self.bc_weight_spin.value())))
        self.data_weight_spin.setValue(float(weights.get("data", self.data_weight_spin.value())))
        self.fvm_checkbox.setChecked(bool(weights.get("use_fvm", self.fvm_checkbox.isChecked())))
        if config.get("results_dir"):
            self.results_dir_edit.setText(str(config["results_dir"]))
        self._sync_backend_controls()
        self._sync_training_profile_controls()

    def _build_backend_command(self, ui_config: dict[str, Any]) -> tuple[list[str], Path]:
        run_dir = Path(ui_config["results_dir"]).resolve()
        mode = ui_config["backend"]["mode"]

        if mode == "Mock backend":
            command = [
                sys.executable,
                "-u",
                "-m",
                "ui.mock_backend",
                "--epochs",
                str(ui_config["training"]["epochs"]),
                "--output-dir",
                str(run_dir),
            ]
            return command, run_dir

        script = Path(ui_config["backend"]["script"] or self.project_root / "Airfoil_PINN.py")
        if not script.is_absolute():
            script = self.project_root / script
        if not script.exists():
            raise FileNotFoundError(f"Backend script not found: {script}")

        backend_config_path = run_dir / "backend_config.yaml"
        self._write_backend_yaml_config(ui_config, backend_config_path)
        command = [
            sys.executable,
            "-u",
            str(script),
            "train",
            "--config",
            str(backend_config_path),
            "--experiment-dir",
            str(run_dir),
        ]
        return command, run_dir

    def _write_backend_yaml_config(self, ui_config: dict[str, Any], path: Path) -> None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to write backend YAML configs") from exc

        config = self._load_base_backend_config(ui_config["backend"].get("base_config"))
        training = config.setdefault("training", {})
        training["epochs"] = ui_config["training"]["epochs"]
        training["batch_size"] = ui_config["training"]["batch_size"]

        optimizer = config.setdefault("optimizer_params", {})
        optimizer["lr"] = ui_config["training"]["learning_rate"]

        loss_cfg = config.setdefault("loss_weights", {})
        pde_weights = loss_cfg.setdefault("pde_weights", {})
        for key in list(pde_weights) or ["cont", "mom_x", "mom_y", "mom_r", "mom_z"]:
            pde_weights[key] = ui_config["loss_weights"]["pde"]
        bc_weights = loss_cfg.setdefault("bc_weights", {})
        for key in list(bc_weights) or ["inlet", "outlet", "wall", "symmetry", "axis_symmetry"]:
            bc_weights[key] = ui_config["loss_weights"]["bc"]
        loss_cfg["data_weight"] = ui_config["loss_weights"]["data"]
        fvm_cfg = loss_cfg.setdefault("fvm", {})
        if isinstance(fvm_cfg, dict):
            fvm_cfg["enabled"] = ui_config["loss_weights"]["use_fvm"]
            fvm_cfg.setdefault("weight", 25.0)
            fvm_cfg.setdefault("batch_interval", 1)
            fvm_cfg.setdefault("reduction", "divergence_l2_volume")
            fvm_cfg.setdefault("schedule", "every_optimizer_step")
        else:
            loss_cfg["fvm"] = {
                "enabled": ui_config["loss_weights"]["use_fvm"],
                "weight": 25.0,
                "batch_interval": 1,
                "reduction": "divergence_l2_volume",
                "schedule": "every_optimizer_step",
            }

        process = config.setdefault("process", {})
        if ui_config["training"]["device"] != "auto":
            process["device"] = ui_config["training"]["device"]

        output = config.setdefault("output", {})
        output["console_only"] = False
        output.setdefault("plot_frequency", 10)

        diagnostics = config.setdefault("diagnostics", {})
        epochs = max(1, int(ui_config["training"]["epochs"]))
        diagnostics["interval"] = min(int(diagnostics.get("interval", epochs) or epochs), epochs)

        config["enable_key_listener"] = False
        config["project_name"] = self._sanitize_name(
            ui_config["profile"]["preset"].replace(" ", "_").lower()
        )
        config["ui"] = ui_config
        self._validate_backend_data_paths(config, ui_config["loss_weights"]["use_fvm"])

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _validate_backend_data_paths(self, config: dict[str, Any], require_mesh: bool) -> None:
        data = config.setdefault("data", {})
        points_file = data.get("points_file")
        if not points_file:
            raise RuntimeError(
                "Selected backend config has no data.points_file. "
                "Create collocation/initial data in the backend workflow or choose another YAML config."
            )
        points_path = self._resolve_backend_path(points_file)
        if not points_path.exists():
            raise FileNotFoundError(f"Backend data.points_file not found: {points_path}")

        if not require_mesh:
            return
        mesh_file = data.get("mesh_file")
        if not mesh_file:
            raise RuntimeError("FVM is enabled, but selected backend config has no data.mesh_file")
        mesh_path = self._resolve_backend_path(mesh_file)
        if not mesh_path.exists():
            raise FileNotFoundError(f"FVM mesh file not found: {mesh_path}")

    def _resolve_backend_path(self, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def _load_base_backend_config(self, config_path: str | None) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read backend YAML configs") from exc

        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = self.project_root / path
            if path.exists():
                data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
                if isinstance(data, dict):
                    return deepcopy(data)
        from airfoil_pinn.config.manager import ConfigManager

        return deepcopy(ConfigManager().config)

    def _save_training_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save training config",
            str(self.workspace_paths.configs_dir / "ui_training_config.json"),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        config = self._collect_training_config()
        self._write_json(Path(path), config)
        self.statusBar().showMessage("Training configuration saved")

    def _load_training_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load training config",
            str(self.workspace_paths.configs_dir),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Configuration JSON must contain an object")
            self._apply_training_config(config)
            self.statusBar().showMessage("Training configuration loaded")
        except Exception as exc:
            QMessageBox.warning(self, "Config load failed", str(exc))

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _refresh_results(self) -> None:
        result_dir = Path(self.results_dir_edit.text()).expanduser()
        if not result_dir.is_absolute():
            result_dir = self.project_root / result_dir
        self.current_results_dir = result_dir
        self.results_folder_label.setText(self._short_project_path(result_dir))
        self.result_files_list.clear()
        self.result_canvas.show_message("Scanning results...")
        self.result_status_label.setText("Scanning results folder...")

        self._results_scan_generation += 1
        generation = self._results_scan_generation
        thread = QThread(self)
        worker = ResultsScanWorker(result_dir)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            lambda image_paths, report_metrics, scan_generation=generation, scan_dir=result_dir: (
                self._handle_results_scan_finished(scan_generation, scan_dir, image_paths, report_metrics)
            )
        )
        worker.error.connect(self._handle_results_scan_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda scan_thread=thread: self._forget_results_scan(scan_thread))
        self._active_results_scans.append(thread)
        self._results_scan_workers[thread] = worker
        thread.start()

    def _handle_results_scan_finished(
        self,
        generation: int,
        result_dir: Path,
        image_paths: list[str],
        report_metrics: dict[str, str],
    ) -> None:
        if generation != self._results_scan_generation:
            return

        images = [Path(path) for path in image_paths]
        self.result_files_list.clear()
        for image_path in images:
            item = QListWidgetItem(self._short_path_from(result_dir, image_path))
            item.setData(Qt.ItemDataRole.UserRole, str(image_path))
            self.result_files_list.addItem(item)

        if images:
            self.result_files_list.setCurrentRow(0)
            self.result_status_label.setText(f"{len(images)} image(s) found")
        else:
            self.result_canvas.show_message("No saved plots found")
            self.result_status_label.setText("No PNG/JPG files found in results folder")

        merged = dict(self.final_metrics)
        merged.update(report_metrics)
        self._update_final_metrics_table(merged)

    def _handle_results_scan_error(self, message: str) -> None:
        self.result_status_label.setText("Results scan failed")
        self.log_panel.append_event(f"Results scan failed: {message}")

    def _forget_results_scan(self, thread: QThread) -> None:
        if thread in self._active_results_scans:
            self._active_results_scans.remove(thread)
        self._results_scan_workers.pop(thread, None)

    def _show_selected_result_image(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        self.result_canvas.load_image(path)
        self.result_status_label.setText(self._short_project_path(path))

    def _open_results_folder(self) -> None:
        result_dir = Path(self.results_dir_edit.text()).expanduser()
        if not result_dir.is_absolute():
            result_dir = self.project_root / result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result_dir)))

    def _choose_results_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Load results folder", str(self.current_results_dir))
        if not folder:
            return
        self.results_dir_edit.setText(folder)
        self._refresh_results()
        self._set_active_module("results")

    def _browse_training_profile_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load airfoil coordinates",
            str(self.geometry_session.foils_dir),
            "Airfoil files (*.txt *.dat *.csv *.npy);;All files (*.*)",
        )
        if not path:
            return
        self.custom_profile_edit.setText(path)
        self.training_profile_combo.setCurrentText("Custom coordinate file")
        self._load_geometry_file_path(Path(path))

    def _browse_backend_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select backend script",
            str(self.project_root),
            "Python files (*.py);;All files (*.*)",
        )
        if path:
            self.backend_script_edit.setText(path)

    def _browse_results_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select results folder", str(self.current_results_dir))
        if folder:
            self.results_dir_edit.setText(folder)

    def _sync_backend_controls(self) -> None:
        actual_backend = self.backend_mode_combo.currentText() != "Mock backend" and not self.training_running
        self.base_config_combo.setEnabled(actual_backend)
        self.backend_script_edit.setEnabled(actual_backend)
        self.browse_backend_button.setEnabled(actual_backend)

    def _sync_training_profile_controls(self) -> None:
        custom = self.training_profile_combo.currentText() == "Custom coordinate file" and not self.training_running
        self.custom_profile_edit.setEnabled(custom)
        self.browse_profile_button.setEnabled(custom)
        self._load_selected_preset_geometry()

    def _load_selected_preset_geometry(self) -> None:
        preset = self.training_profile_combo.currentText()
        sample_map = {
            "NACA 0012": "closed_naca",
            "NACA 2412": "naca2412",
            "RAE 2822": "rae2822",
        }
        sample = sample_map.get(preset)
        if sample is None:
            return
        try:
            self.geometry_session.load_sample(sample)
            self._create_domain_from_controls()
            self._sync_geometry_ui(reset_view=True, update_name=True)
            self.log_panel.append_event(f"Loaded preset profile: {preset}")
        except GeometryReadError as exc:
            self.log_panel.append_event(f"Preset profile failed: {exc}")

    def _available_backend_configs(self) -> list[Path]:
        configs_dir = self.workspace_paths.configs_dir
        if not configs_dir.exists():
            return []
        return sorted(configs_dir.glob("*.yaml"))

    def _update_final_metrics_table(self, metrics: dict[str, Any]) -> None:
        preferred = [
            "epoch",
            "total_epochs",
            "loss",
            "pde_loss",
            "bc_loss",
            "data_loss",
            "final_loss",
            "final_pde_loss",
            "final_bc_loss",
            "best_epoch",
        ]
        rows: list[tuple[str, Any]] = []
        for key in preferred:
            if key in metrics:
                rows.append((key, metrics[key]))
        for key in sorted(metrics):
            if key not in preferred and key.startswith(("final_", "best_", "mass_")):
                rows.append((key, metrics[key]))
        self.final_metrics_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.final_metrics_table.setItem(row, 0, QTableWidgetItem(key))
            self.final_metrics_table.setItem(row, 1, QTableWidgetItem(self._format_metric(value)))

    def _set_combo_text(self, combo: QComboBox, value: object) -> None:
        if value is None:
            return
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
        if value is None:
            return
        value_text = str(value)
        for index in range(combo.count()):
            if str(combo.itemData(index)) == value_text:
                combo.setCurrentIndex(index)
                return

    def _short_project_path(self, value: str | Path) -> str:
        return self._short_path_from(self.project_root, Path(value))

    def _short_path_from(self, base: Path, value: Path) -> str:
        try:
            return str(value.resolve().relative_to(base.resolve()))
        except (ValueError, RuntimeError, OSError):
            return str(value)

    def _format_metric(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(number):
            return f"{number:.3e}" if abs(number) < 0.001 or abs(number) >= 1000 else f"{number:.6g}"
        return str(value)

    def _sanitize_name(self, value: str) -> str:
        cleaned = []
        for char in value:
            if char.isalnum() or char in {"_", "-"}:
                cleaned.append(char)
            elif char in {" ", "."}:
                cleaned.append("_")
        return "".join(cleaned).strip("_") or "airfoil_pinn"
