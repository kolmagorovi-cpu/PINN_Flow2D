"""The application must reuse D:/Python without package installation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from airfoil_pinn.app import runtime


class SharedRuntimeEnvironmentTests(unittest.TestCase):
    """Guard the no-bootstrap-install environment contract."""

    def test_required_packages_resolve_from_shared_environment(self) -> None:
        environment = runtime.PYTHON_ENV_DIR.resolve()
        self.assertEqual(
            runtime._venv_python().resolve(),
            Path("D:/Python/BaseEnv/Scripts/python.exe").resolve(),
        )
        for import_name, _ in runtime.REQUIRED_PACKAGES:
            with self.subTest(package=import_name):
                spec = importlib.util.find_spec(import_name)
                self.assertIsNotNone(spec)
                self.assertIsNotNone(spec.origin)
                self.assertTrue(
                    Path(spec.origin).resolve().is_relative_to(environment),
                    f"{import_name} loaded outside {environment}: {spec.origin}",
                )

    def test_missing_dependency_never_invokes_subprocess_install(self) -> None:
        with (
            patch.object(runtime.importlib.util, "find_spec", return_value=None),
            patch.object(runtime.subprocess, "run") as run,
            self.assertRaises(SystemExit),
        ):
            runtime.ensure_virtual_environment()

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
