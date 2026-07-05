#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Совместимый фасад пакета утилит.

Реальная реализация перенесена в ``airfoil_pinn.utils``. Этот пакет оставлен,
чтобы старые сценарии с импортами вида ``from utils import safe_grad`` работали
без изменений во время поэтапного рефакторинга.
"""

from airfoil_pinn.utils import *  # noqa: F401,F403
from airfoil_pinn.utils import __all__  # noqa: F401
