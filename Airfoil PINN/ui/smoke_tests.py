"""Smoke checks for the PySide6 UI branch.

Run from the project root:
    python -m ui.smoke_tests
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.geometry_frontend import GeometryFrontendSession  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.workers import BackendWorker  # noqa: E402


def test_worker_metric_parser() -> None:
    worker = BackendWorker(["python"], cwd=".")
    seen: list[dict] = []
    worker.metrics_updated.connect(lambda metrics: seen.append(dict(metrics)))
    worker._parse_metrics("Epoch 3/10, Loss: 1.23e-4, PDE: 5.0e-5, BC: 2.0e-5, Data: 1.0e-5")

    assert seen, "worker did not emit parsed metrics"
    latest = seen[-1]
    assert latest["epoch"] == 3
    assert latest["total_epochs"] == 10
    assert latest["loss"] == 1.23e-4
    assert latest["pde_loss"] == 5.0e-5
    assert latest["bc_loss"] == 2.0e-5
    assert latest["data_loss"] == 1.0e-5


def test_geometry_save() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        foils_dir = root / "Geometry" / "Foils"
        foils_dir.mkdir(parents=True)
        (foils_dir / "sample.txt").write_text("0 0\n1 0\n", encoding="utf-8")

        session = GeometryFrontendSession(project_root=root)
        session.create_domain(front=1.5, rear=3.0, top=1.2, bottom=1.2)
        artifacts = session.save("smoke", "csv")

        assert artifacts.foil_path.exists()
        assert artifacts.domain_path.exists()
        assert artifacts.metadata_path.exists()


def test_main_window_mock_backend(app: QApplication) -> None:
    settings = QSettings("AirfoilPINN", "AirfoilPINNUI")
    settings_backup = {key: settings.value(key) for key in settings.allKeys()}
    with tempfile.TemporaryDirectory(prefix="airfoil_ui_smoke_") as tmp:
        window = MainWindow()
        try:
            run_dir = Path(tmp)
            window.backend_mode_combo.setCurrentText("Mock backend")
            window.epochs_spin.setValue(3)
            window.results_dir_edit.setText(str(run_dir))

            timed_out = {"value": False}

            def poll_done() -> None:
                if not window.training_running:
                    app.quit()
                    return
                QTimer.singleShot(100, poll_done)

            def timeout() -> None:
                timed_out["value"] = True
                if window.training_running:
                    window._stop_training()
                QTimer.singleShot(1500, app.quit)

            QTimer.singleShot(0, window._start_training)
            QTimer.singleShot(100, poll_done)
            QTimer.singleShot(20_000, timeout)
            app.exec()

            assert not timed_out["value"], "mock backend did not finish in time"
            assert not window.training_running
            assert window.final_metrics.get("epoch") == 3
            assert len(list(run_dir.rglob("*.png"))) == 3
        finally:
            window.close()
            settings.clear()
            for key, value in settings_backup.items():
                settings.setValue(key, value)
            settings.sync()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    test_worker_metric_parser()
    test_geometry_save()
    test_main_window_mock_backend(app)
    print("UI smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
