"""Subprocess worker for running the training backend from the Qt UI."""

from __future__ import annotations

import os
import re
import signal
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Mapping, Sequence

from PySide6.QtCore import QObject, Signal, Slot


class BackendWorker(QObject):
    """Run a backend command in a worker thread and stream its output."""

    log_message = Signal(str)
    progress = Signal(int)
    metrics_updated = Signal(dict)
    finished = Signal(bool, str)
    error = Signal(str)

    _EPOCH_RE = re.compile(r"\bEpoch\s*(?:=|:)?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    _EPOCH_ONLY_RE = re.compile(r"\bEpoch\s+(\d+)\b", re.IGNORECASE)
    _LOSS_RE = re.compile(
        r"\b(?:Total\s+Loss|Loss|loss)\b\s*[:=]?\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    _PDE_RE = re.compile(r"\bPDE\b\s*[:=]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.IGNORECASE)
    _BC_RE = re.compile(r"\bBC\b\s*[:=]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.IGNORECASE)
    _DATA_RE = re.compile(r"\bData\b\s*[:=]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", re.IGNORECASE)

    def __init__(
        self,
        command: Sequence[str],
        cwd: str | Path,
        expected_epochs: int | None = None,
        env: Mapping[str, str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.command = [str(part) for part in command]
        self.cwd = Path(cwd)
        self.expected_epochs = expected_epochs
        self.env = dict(env) if env is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._stop_requested = False
        self._last_metrics: dict[str, float | int] = {}
        self._state_lock = threading.Lock()

    @Slot()
    def run(self) -> None:
        """Start the configured command and block until it exits."""
        if not self.command:
            self._emit_failure("Backend command is empty")
            return
        if not self.cwd.exists():
            self._emit_failure(f"Working directory not found: {self.cwd}")
            return

        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("MPLBACKEND", "Agg")

        creationflags = 0
        preexec_fn = None
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            preexec_fn = os.setsid

        try:
            process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            with self._state_lock:
                self._process = process
        except FileNotFoundError as exc:
            self._emit_failure(f"Backend executable not found: {self.command[0]}")
            self.error.emit(str(exc))
            return
        except OSError as exc:
            self._emit_failure(f"Backend failed to start: {exc}")
            self.error.emit(str(exc))
            return

        self.log_message.emit(f"$ {self._format_command(self.command)}")

        threads = [
            threading.Thread(target=self._read_stream, args=(process.stdout, ""), daemon=True),
            threading.Thread(target=self._read_stream, args=(process.stderr, "stderr: "), daemon=True),
        ]
        for thread in threads:
            thread.start()

        return_code = process.wait()
        for thread in threads:
            thread.join(timeout=1.0)

        if self._is_stop_requested():
            self.finished.emit(False, "Training stopped by user")
            return

        if return_code == 0:
            self.progress.emit(100)
            self.finished.emit(True, "Training completed")
            return

        message = f"Backend exited with code {return_code}"
        self.error.emit(message)
        self.finished.emit(False, message)

    @Slot()
    def request_stop(self) -> None:
        """Ask the running backend process to stop."""
        with self._state_lock:
            self._stop_requested = True
            process = self._process
        if process is None or process.poll() is not None:
            return

        self.log_message.emit("Stopping backend process...")
        threading.Thread(
            target=self._terminate_process,
            args=(process,),
            name="backend-terminator",
            daemon=True,
        ).start()

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        try:
            if sys.platform == "win32":
                self._terminate_windows_process_tree(process)
            else:
                self._terminate_posix_process_group(process)
        except Exception as exc:  # pragma: no cover - platform/process race
            self.error.emit(f"Could not terminate backend gracefully: {exc}")
            try:
                process.terminate()
            except OSError:
                pass

    def _read_stream(self, stream, prefix: str) -> None:
        if stream is None:
            return
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                message = f"{prefix}{line}" if prefix else line
                self.log_message.emit(message)
                self._parse_metrics(line)
        except (OSError, ValueError) as exc:
            if not self._is_stop_requested():
                self.error.emit(f"Backend stream read failed: {exc}")
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _parse_metrics(self, line: str) -> None:
        epoch_match = self._EPOCH_RE.search(line)
        epoch: int | None = None
        total_epochs: int | None = None
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
        elif self.expected_epochs:
            epoch_only = self._EPOCH_ONLY_RE.search(line)
            if epoch_only:
                epoch = int(epoch_only.group(1))
                total_epochs = self.expected_epochs

        loss = self._extract_float(self._LOSS_RE, line)
        pde = self._extract_float(self._PDE_RE, line)
        bc = self._extract_float(self._BC_RE, line)
        data = self._extract_float(self._DATA_RE, line)
        table_name, table_value = self._extract_table_metric(line)
        if table_name == "total loss" and loss is None:
            loss = table_value
        elif table_name == "total pde" and pde is None:
            pde = table_value
        elif table_name == "total bc" and bc is None:
            bc = table_value

        metrics: dict[str, float | int] = {}
        if epoch is not None:
            metrics["epoch"] = epoch
        if total_epochs is not None:
            metrics["total_epochs"] = total_epochs
        if loss is not None:
            metrics["loss"] = loss
        if pde is not None:
            metrics["pde_loss"] = pde
        if bc is not None:
            metrics["bc_loss"] = bc
        if data is not None:
            metrics["data_loss"] = data

        if not metrics:
            return

        with self._state_lock:
            self._last_metrics.update(metrics)
            snapshot = dict(self._last_metrics)
        if epoch is not None and total_epochs:
            percent = max(0, min(100, round(epoch / total_epochs * 100)))
            self.progress.emit(percent)
        self.metrics_updated.emit(snapshot)

    @staticmethod
    def _extract_float(pattern: re.Pattern[str], line: str) -> float | None:
        match = pattern.search(line)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_table_metric(line: str) -> tuple[str | None, float | None]:
        if "|" not in line:
            return None, None
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) < 3:
            return None, None
        name = cells[0].lower()
        if name not in {"total loss", "total pde", "total bc"}:
            return None, None
        try:
            return name, float(cells[2])
        except ValueError:
            return None, None

    def _emit_failure(self, message: str) -> None:
        self.error.emit(message)
        self.finished.emit(False, message)

    def _is_stop_requested(self) -> bool:
        with self._state_lock:
            return self._stop_requested

    @staticmethod
    def _format_command(command: Sequence[str]) -> str:
        if sys.platform == "win32":
            return subprocess.list2cmdline([str(part) for part in command])
        return shlex.join(str(part) for part in command)

    @staticmethod
    def _terminate_windows_process_tree(process: subprocess.Popen[str]) -> None:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    @staticmethod
    def _terminate_posix_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return


class ResultsScanWorker(QObject):
    """Scan a results folder away from the GUI thread."""

    finished = Signal(list, dict)
    error = Signal(str)

    def __init__(self, result_dir: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.result_dir = Path(result_dir)

    @Slot()
    def run(self) -> None:
        try:
            images = self._result_image_paths(self.result_dir)
            report_metrics = self._read_final_report(self.result_dir)
        except Exception as exc:  # pragma: no cover - filesystem race
            self.error.emit(str(exc))
            self.finished.emit([], {})
            return
        self.finished.emit([str(path) for path in images], report_metrics)

    @staticmethod
    def _result_image_paths(result_dir: Path) -> list[Path]:
        if not result_dir.exists():
            return []
        allowed = {".png", ".jpg", ".jpeg"}
        return sorted(path for path in result_dir.rglob("*") if path.suffix.lower() in allowed)

    @staticmethod
    def _read_final_report(result_dir: Path) -> dict[str, str]:
        candidates = [result_dir / "final_report.txt", result_dir / "plots" / "final_report.txt"]
        for path in candidates:
            if not path.exists():
                continue
            metrics: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    metrics[key] = value
            return metrics
        return {}
