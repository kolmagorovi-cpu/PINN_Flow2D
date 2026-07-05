#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакет управления конфигурацией Airfoil PINN.

Здесь лежит ``ConfigManager`` и небольшие консольные помощники, которые нужны
обучающему CLI. Корневой модуль ``config_manager`` оставлен как совместимый
фасад для старых сценариев и тестов.
"""

from airfoil_pinn.config.manager import ConfigManager, warn_activation, yes_no

__all__ = ["ConfigManager", "warn_activation", "yes_no"]
