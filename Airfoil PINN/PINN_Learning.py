#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимая консольная точка входа обучения PINN.

Основная реализация перенесена в ``airfoil_pinn.training.learning_cli``.
Этот файл сохранён, чтобы команды ``python PINN_Learning.py`` и
``python -m PINN_Learning`` продолжали работать во время поэтапного
рефакторинга.
"""

from airfoil_pinn.training.learning_cli import *  # noqa: F401,F403
from airfoil_pinn.training.learning_cli import main


if __name__ == "__main__":
    main()
