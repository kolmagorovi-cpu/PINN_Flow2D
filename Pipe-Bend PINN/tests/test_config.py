#!/usr/bin/env python3
"""Тест загрузки конфигурации из config.yaml + подконфигов."""
import sys
import os

# Корень проекта — родительский каталог tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipe_bend_pinn.config.manager import ConfigManager


cfg = ConfigManager()
print(f"Проект: {cfg.config.get('project_name')}")
print(f"Файл точек: {cfg.config['data'].get('points_file')}")
print(f"Архитектура: {cfg.config['architecture']['hidden_layers']}")
print(f"Активация: {cfg.config['architecture'].get('activation')}")
print(f"Эпохи: {cfg.config['training']['epochs']}")
print(f"Batch size: {cfg.config['training']['batch_size']}")
print(f"LR: {cfg.config['training'].get('lr', 1e-3)}")
print(f"Two-stage: {cfg.config['training']['two_stage']}")
print(f"BC weights: {cfg.config['loss_weights']['bc_weights']}")
print(f"Device: {cfg.config['process']['device']}")
print(f"Главный конфиг: {cfg.main_config_path}")
print("✅ Конфигурация загружена успешно")