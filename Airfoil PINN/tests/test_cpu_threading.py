"""Tests for configurable PyTorch CPU thread pools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import ConfigurationError, validate_static_config
from airfoil_pinn.training.cpu_runtime import configure_cpu_threading
from airfoil_pinn.training.menus.process import process_settings


class CPUThreadingTests(unittest.TestCase):
    def test_auto_uses_all_available_logical_cpus(self) -> None:
        with (
            patch("airfoil_pinn.training.cpu_runtime.os.cpu_count", return_value=12),
            patch("airfoil_pinn.training.cpu_runtime.torch.set_num_threads") as set_intra,
            patch(
                "airfoil_pinn.training.cpu_runtime.torch.get_num_threads",
                return_value=12,
            ),
            patch(
                "airfoil_pinn.training.cpu_runtime.torch.get_num_interop_threads",
                side_effect=[4, 1, 1],
            ),
            patch(
                "airfoil_pinn.training.cpu_runtime.torch.set_num_interop_threads"
            ) as set_interop,
        ):
            settings = configure_cpu_threading(
                {"cpu_threads": "auto", "cpu_interop_threads": 1},
                cpu_active=True,
            )

        set_intra.assert_called_once_with(12)
        set_interop.assert_called_once_with(1)
        self.assertEqual(settings["intra_op"], 12)
        self.assertEqual(settings["inter_op"], 1)

    def test_cuda_does_not_reconfigure_cpu_pools(self) -> None:
        with (
            patch("airfoil_pinn.training.cpu_runtime.torch.set_num_threads") as set_intra,
            patch(
                "airfoil_pinn.training.cpu_runtime.torch.set_num_interop_threads"
            ) as set_interop,
        ):
            settings = configure_cpu_threading(
                {"cpu_threads": 3, "cpu_interop_threads": 2},
                cpu_active=False,
            )

        set_intra.assert_not_called()
        set_interop.assert_not_called()
        self.assertFalse(settings["active"])

    def test_invalid_thread_count_is_rejected(self) -> None:
        config = ConfigManager().config
        config["process"]["cpu_threads"] = 0
        with self.assertRaisesRegex(ConfigurationError, "cpu_threads"):
            validate_static_config(config)

    def test_cpu_device_allows_zero_gpus(self) -> None:
        config = ConfigManager().config
        config["process"].update({"device": "cpu", "num_gpus": 0})
        validate_static_config(config)

    def test_device_menu_requests_cpu_thread_count(self) -> None:
        config = ConfigManager()
        with (
            patch("builtins.input", side_effect=["2", "cpu", "6", "0"]),
            patch("airfoil_pinn.training.menus.process.clear_screen"),
            patch("airfoil_pinn.training.menus.process.print_submenu_header"),
            patch("airfoil_pinn.training.menus.process.print_submenu_footer"),
        ):
            process_settings(config)

        self.assertEqual(config.config["process"]["device"], "cpu")
        self.assertEqual(config.config["process"]["num_gpus"], 0)
        self.assertEqual(config.config["process"]["cpu_threads"], 6)


if __name__ == "__main__":
    unittest.main()
