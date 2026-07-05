#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый импорт модуля дискретизации.

Основная реализация перенесена в ``airfoil_pinn.sampling``. Старое имя
``Collocation`` остается рабочим для существующих сценариев и импортов.
"""

from airfoil_pinn.sampling import *  # noqa: F401,F403
from airfoil_pinn.sampling import __all__  # noqa: F401


if __name__ == "__main__":
    S2DR()
