#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запуск всех тестов пакета ``pipe_bend_pinn_data``.

Использование (из корня «Pipe-Bend PINN»)::

    python pipe_bend_pinn_data/tests/run_all.py
"""

from __future__ import annotations

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    suite = unittest.TestLoader().discover(
        start_dir=TESTS_DIR,
        pattern="test_*.py",
        top_level_dir=PROJECT_ROOT,
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
