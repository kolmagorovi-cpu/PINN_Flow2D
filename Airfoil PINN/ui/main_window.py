"""Main window for the PySide6 UI branch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QVBoxLayout,
    QWidget,
)

from ui.geometry_frontend import GeometryFrontendSession, GeometryReadError
from ui.mock_data import next_metric_value
from ui.theme import PALETTE
from ui.widgets.geometry_canvas import GeometryCanvas
from ui.widgets.log_panel import LogPanel
from ui.widgets.metric_plot import MetricPlot


@dataclass
class MockTrainingState:
    running: bool = False
    epoch: int = 0
    total_epochs: int = 500
    loss: float = 2.5
    pde: float = 0.9
    bc: float = 0.45


class MainWindow(QMainWindow):
    """Autonomous GUI prototype; no imports from project backend modules."""

    GEOMETRY_FILE_EXTENSIONS = {".txt", ".dat", ".csv", ".npy"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Airfoil PINN - UI Prototype")
        self.resize(1320, 840)
        self.setAcceptDrops(True)
        self.training = MockTrainingState()
        self.geometry_session = GeometryFrontendSession()
        self.external_geometry_path: Path | None = None
        self.manual_coordinates_text = ""

        self._create_actions()
        self._create_toolbar()
        self._build_layout()
        self._connect_signals()
        self._refresh_foil_files()
        self._refresh_library_names()
        self._apply_domain_params_to_controls()
        self._apply_save_format_to_controls()
        self._sync_geometry_ui(reset_view=True, update_name=True)

        self.training_timer = QTimer(self)
        self.training_timer.setInterval(220)
        self.training_timer.timeout.connect(self._training_tick)

        self.log_panel.append_event("UI branch started")
        self.statusBar().showMessage("Ready")

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

    def _create_actions(self) -> None:
        style = QApplication.style()
        self.open_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open", self)
        self.save_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self.reset_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Reset view", self)
        self.start_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Start", self)
        self.stop_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop", self)
        self.info_action = QAction(style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation), "About", self)

        self.open_action.setToolTip("Load airfoil file")
        self.save_action.setToolTip("Save mock project")
        self.reset_action.setToolTip("Reset geometry view")
        self.start_action.setToolTip("Start mock training")
        self.stop_action.setToolTip("Stop mock training")
        self.info_action.setToolTip("Show prototype details")

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
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = self._build_sidebar()
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.geometry_page = self._build_geometry_page()
        self.collocation_page = self._build_collocation_page()
        self.fluid_page = self._build_fluid_page()
        self.boundary_page = self._build_boundary_page()
        self.initialization_page = self._build_initialization_page()
        self.training_page = self._build_training_page()
        self.results_page = self._build_results_page()

        for page in (
            self.geometry_page,
            self.collocation_page,
            self.fluid_page,
            self.boundary_page,
            self.initialization_page,
            self.training_page,
            self.results_page,
        ):
            self.stack.addWidget(page)

        workspace = QSplitter(Qt.Orientation.Vertical)
        workspace.addWidget(self.stack)
        self.log_panel = LogPanel()
        workspace.addWidget(self.log_panel)
        workspace.setSizes([640, 150])
        root.addWidget(workspace, 1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setFixedWidth(236)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 18, 10, 18)
        layout.setSpacing(14)

        title = QLabel("AEROPINN")
        title.setObjectName("AppTitle")
        subtitle = QLabel("UI prototype")
        subtitle.setObjectName("AppSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepList")
        self.step_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for text in (
            "Geometry",
            "Collocation",
            "Fluid",
            "Boundaries",
            "Initialization",
            "Training",
            "Results",
        ):
            self.step_list.addItem(QListWidgetItem(text))
        self.step_list.setCurrentRow(0)
        layout.addWidget(self.step_list, 1)

        mode = QLabel("UI branch")
        mode.setObjectName("AppSubtitle")
        layout.addWidget(mode)
        return frame

    def _build_geometry_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)

        header = self._build_geometry_header()
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.geometry_canvas = GeometryCanvas()
        splitter.addWidget(self.geometry_canvas)
        splitter.addWidget(self._build_geometry_controls())
        splitter.setSizes([900, 340])
        layout.addWidget(splitter, 1)
        return page

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
            "Geometry/Foils file",
            "External file",
            "Manual input",
            "geom_config.yaml",
            "Foil library",
            "Built-in closed NACA 0012",
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

    def _build_geometry_controls(self) -> QWidget:
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

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

        self.stats_table = QTableWidget(8, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._setup_compact_table(self.stats_table, height=236)
        layout.addWidget(self._wrap_group("Airfoil Stats", self.stats_table))

        self.vertices_table = QTableWidget(0, 2)
        self.vertices_table.setHorizontalHeaderLabels(["x, m", "y, m"])
        self._setup_compact_table(self.vertices_table, height=180)
        layout.addWidget(self._wrap_group("Domain Vertices", self.vertices_table))

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

        panel.setWidget(content)
        panel.setMinimumWidth(320)
        return panel

    def _build_collocation_page(self) -> QWidget:
        return self._form_page(
            "Collocation",
            "Sampler preview",
            [
                ("Grid step, m", self._double_spin(0.001, 1.0, 0.04, 0.005)),
                ("Boundary density", self._double_spin(0.1, 10.0, 2.5, 0.1)),
                ("Boundary mode", self._combo(["inlet/outlet/wall/symmetry", "axisymmetric", "custom labels"])),
                ("Multiprocessing", self._combo(["auto", "1 process", "4 processes", "8 processes"])),
            ],
        )

    def _build_fluid_page(self) -> QWidget:
        return self._form_page(
            "Fluid",
            "Material preset",
            [
                ("Preset", self._combo(["Air 20C", "Water 20C", "Custom"])),
                ("Density, kg/m3", self._double_spin(0.1, 5000.0, 1.204, 0.01)),
                ("Kinematic viscosity, m2/s", self._double_spin(1e-8, 1e-2, 1.5e-5, 1e-6, decimals=8)),
                ("Dynamic viscosity, Pa*s", self._double_spin(1e-8, 1e-1, 1.8e-5, 1e-6, decimals=8)),
            ],
        )

    def _build_boundary_page(self) -> QWidget:
        return self._form_page(
            "Boundaries",
            "Condition set",
            [
                ("Inlet velocity, m/s", self._double_spin(0.0, 400.0, 30.0, 0.5)),
                ("Outlet pressure, Pa", self._double_spin(-1e6, 1e6, 0.0, 100.0)),
                ("Wall", self._combo(["no slip", "slip", "moving wall"])),
                ("Top boundary", self._combo(["symmetry", "outlet", "wall"])),
                ("Bottom boundary", self._combo(["wall", "symmetry", "axis symmetry"])),
            ],
        )

    def _build_initialization_page(self) -> QWidget:
        return self._form_page(
            "Initialization",
            "Field setup",
            [
                ("Pressure mode", self._combo(["constant", "dynamic pressure", "linear"])),
                ("Velocity mode", self._combo(["uniform", "potential preview", "noisy uniform"])),
                ("Velocity noise", self._double_spin(0.0, 1.0, 0.01, 0.01)),
                ("Pressure noise", self._double_spin(0.0, 1.0, 0.0, 0.01)),
                ("Temperature, K", self._double_spin(100.0, 600.0, 293.15, 0.5)),
            ],
        )

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        layout.addWidget(self._page_header("Training", "Mock live run"))

        top = QGridLayout()
        self.epoch_value = self._metric_label("0 / 500")
        self.loss_value = self._metric_label("2.500e+00")
        self.pde_value = self._metric_label("9.000e-01")
        self.bc_value = self._metric_label("4.500e-01")
        top.addWidget(self._metric_card("Epoch", self.epoch_value), 0, 0)
        top.addWidget(self._metric_card("Total loss", self.loss_value), 0, 1)
        top.addWidget(self._metric_card("PDE loss", self.pde_value), 0, 2)
        top.addWidget(self._metric_card("BC loss", self.bc_value), 0, 3)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 500)
        layout.addWidget(self.progress)

        plots = QHBoxLayout()
        self.loss_plot = MetricPlot("Total loss", PALETTE.accent)
        self.pde_plot = MetricPlot("PDE", PALETTE.success)
        self.bc_plot = MetricPlot("Boundary", PALETTE.warning)
        plots.addWidget(self.loss_plot)
        plots.addWidget(self.pde_plot)
        plots.addWidget(self.bc_plot)
        layout.addLayout(plots)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("PrimaryButton")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("DangerButton")
        style = QApplication.style()
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.pause_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addStretch(1)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        layout.addWidget(self._page_header("Results", "Mock artifacts"))

        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["Artifact", "Status", "Path"])
        rows = [
            ("geometry_preview.png", "ready", "mock/results/geometry_preview.png"),
            ("collocation_points.npz", "ready", "mock/data/collocation_points.npz"),
            ("training_metrics.csv", "waiting", "mock/logs/training_metrics.csv"),
            ("final_fields.png", "waiting", "mock/results/final_fields.png"),
            ("checkpoint.pt", "waiting", "mock/checkpoints/checkpoint.pt"),
        ]
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        return page

    def _form_page(self, title: str, subtitle: str, fields: list[tuple[str, QWidget]]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        layout.addWidget(self._page_header(title, subtitle))

        box = QGroupBox(title)
        form = QFormLayout(box)
        for label, widget in fields:
            form.addRow(label, widget)
        layout.addWidget(box)

        preview = QLabel("Status: mock data")
        preview.setObjectName("Muted")
        layout.addWidget(preview)
        layout.addStretch(1)
        return page

    def _page_header(self, title: str, subtitle: str) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Panel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        text = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("Muted")
        text.addWidget(title_label)
        text.addWidget(subtitle_label)
        layout.addLayout(text)
        layout.addStretch(1)
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

    def _combo(self, values: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        return combo

    def _connect_signals(self) -> None:
        for drop_target in (self.geometry_page, self.geometry_canvas, self.stack):
            drop_target.setAcceptDrops(True)
            drop_target.installEventFilter(self)

        self.step_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.geometry_canvas.cursorChanged.connect(self._update_cursor_label)
        self.geometry_canvas.selectionChanged.connect(self.statusBar().showMessage)

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
        self.start_action.triggered.connect(self._start_training)
        self.stop_action.triggered.connect(self._stop_training)
        self.start_button.clicked.connect(self._start_training)
        self.pause_button.clicked.connect(self._pause_training)
        self.stop_button.clicked.connect(self._stop_training)

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
            self.library_combo.addItem("No foil_geoms.yaml entries", "")
        for name in names:
            self.library_combo.addItem(name, name)
        self._update_source_controls()

    def _update_source_controls(self) -> None:
        mode = self.source_combo.currentText()
        file_mode = mode == "Geometry/Foils file"
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
            elif mode == "Built-in open plate":
                self.geometry_session.load_sample("open_plate")
            elif mode == "Geometry/Foils file":
                path = self.foil_file_combo.currentData()
                if not path:
                    raise GeometryReadError("Geometry/Foils does not contain supported files")
                self.geometry_session.load_file(path)
            elif mode == "Manual input":
                self._open_manual_coordinates_dialog()
                return
            elif mode == "geom_config.yaml":
                self.geometry_session.load_from_config()
                self._apply_domain_params_to_controls()
                self._apply_save_format_to_controls()
            elif mode == "Foil library":
                geom_name = self.library_combo.currentData()
                if not geom_name:
                    raise GeometryReadError("Geometry/foil_geoms.yaml does not contain profiles")
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
            "PySide6 UI branch. Geometry is implemented in UI code without modifying backend files.",
        )

    def _start_training(self) -> None:
        self.training.running = True
        if self.training.epoch >= self.training.total_epochs:
            self.training = MockTrainingState(total_epochs=self.training.total_epochs)
            self.progress.setValue(0)
            self.loss_plot.set_values([])
            self.pde_plot.set_values([])
            self.bc_plot.set_values([])
        self.training_timer.start()
        self.stack.setCurrentWidget(self.training_page)
        self.step_list.setCurrentRow(5)
        self.log_panel.append_event("Mock training started")

    def _pause_training(self) -> None:
        self.training.running = False
        self.training_timer.stop()
        self.log_panel.append_event("Mock training paused")

    def _stop_training(self) -> None:
        self.training.running = False
        self.training_timer.stop()
        self.log_panel.append_event("Mock training stopped")

    def _training_tick(self) -> None:
        if not self.training.running:
            return

        self.training.epoch += 1
        self.training.loss = next_metric_value(self.training.loss, 3e-4)
        self.training.pde = next_metric_value(self.training.pde, 2e-4)
        self.training.bc = next_metric_value(self.training.bc, 1e-4)

        self.epoch_value.setText(f"{self.training.epoch} / {self.training.total_epochs}")
        self.loss_value.setText(f"{self.training.loss:.3e}")
        self.pde_value.setText(f"{self.training.pde:.3e}")
        self.bc_value.setText(f"{self.training.bc:.3e}")
        self.progress.setValue(self.training.epoch)
        self.loss_plot.append_value(self.training.loss)
        self.pde_plot.append_value(self.training.pde)
        self.bc_plot.append_value(self.training.bc)

        if self.training.epoch % 50 == 0:
            self.log_panel.append_event(f"Epoch {self.training.epoch}: loss={self.training.loss:.3e}")

        if self.training.epoch >= self.training.total_epochs:
            self.training.running = False
            self.training_timer.stop()
            self.log_panel.append_event("Mock training finished")
