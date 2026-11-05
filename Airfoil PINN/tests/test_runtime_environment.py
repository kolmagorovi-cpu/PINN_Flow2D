"""Environment selection and bootstrap regression tests."""
import os
import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from airfoil_pinn.app import runtime


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_explicit_override(self):
        with patch.dict(os.environ, {"AIRFOIL_PINN_PYTHON_ENV": "custom"}):
            self.assertEqual(runtime._select_python_environment(), Path("custom"))

    def test_shared_has_priority(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "is_file", return_value=True):
            self.assertEqual(runtime._select_python_environment(), runtime.DEFAULT_PYTHON_ENV_DIR)

    def test_local_selected_when_shared_missing(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "is_file", return_value=False):
            self.assertEqual(runtime._select_python_environment(), runtime.PROJECT_ROOT / "venv")

    def test_create_local_and_restart(self):
        with (
            patch.object(runtime, "PYTHON_ENV_DIR", runtime.PROJECT_ROOT / "venv"),
            patch.object(Path, "is_file", side_effect=[False, True]),
            patch.object(Path, "exists", return_value=True),
            patch.object(sys, "executable", "system-python"),
            patch.object(runtime.subprocess, "run") as run,
            patch.object(runtime, "_restart_inside_venv", side_effect=SystemExit(0)) as restart,
            self.assertRaises(SystemExit) as caught,
        ):
            runtime.ensure_virtual_environment()
        self.assertEqual(caught.exception.code, 0)
        run.assert_called_once_with(["system-python", "-m", "venv", str(runtime.PROJECT_ROOT / "venv")], check=True)
        restart.assert_called_once()

    def test_install_missing_local_dependencies(self):
        with (
            patch.object(runtime, "PYTHON_ENV_DIR", runtime.PROJECT_ROOT / "venv"),
            patch.object(runtime, "_venv_python", return_value=Path(sys.executable)),
            patch.object(runtime.importlib.util, "find_spec", return_value=None),
            patch.object(runtime.subprocess, "run") as run,
            patch.object(runtime, "_restart_inside_venv") as restart,
        ):
            runtime.ensure_virtual_environment()
        run.assert_called_once_with([sys.executable, "-m", "pip", "install", *[p for _, p in runtime.REQUIRED_PACKAGES]], check=True)
        restart.assert_called_once()

    def test_shared_dependencies_are_not_installed(self):
        with (
            patch.object(runtime, "PYTHON_ENV_DIR", Path("external-env")),
            patch.object(runtime, "_venv_python", return_value=Path(sys.executable)),
            patch.object(runtime.importlib.util, "find_spec", return_value=None),
            patch.object(runtime.subprocess, "run") as run,
            self.assertRaises(SystemExit),
        ):
            runtime.ensure_virtual_environment()
        run.assert_not_called()

    def test_creation_failure_stops_startup(self):
        with (
            patch.object(runtime, "PYTHON_ENV_DIR", runtime.PROJECT_ROOT / "venv"),
            patch.object(Path, "is_file", return_value=False),
            patch.object(runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "venv")),
            patch.object(runtime, "_restart_inside_venv") as restart,
            self.assertRaises(SystemExit) as caught,
        ):
            runtime.ensure_virtual_environment()
        self.assertEqual(caught.exception.code, 1)
        restart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
