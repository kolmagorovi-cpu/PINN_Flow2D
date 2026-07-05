#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консольный вход модуля дискретизации S2DR.

Главное меню импортирует эту тонкую обертку, а расчетная и интерактивная логика
пока находится в ``airfoil_pinn.sampling.core``.
"""

from airfoil_pinn.sampling.core import S2DR

__all__ = ["S2DR"]
