#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Общие константы проекта Pipe-Bend PINN.

Содержит все магические числа и конфигурационные константы,
используемые в различных частях проекта.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import sys
from typing import Final

# =============================================================================
# ГЕОМЕТРИЯ И ДИСКРЕТИЗАЦИЯ
# =============================================================================

CLASSIFICATION_TOLERANCE_FACTOR: Final[float] = 0.1
MAX_ATTEMPTS_MULTIPLIER: Final[int] = 20
DEFAULT_BATCH_SIZE: Final[int] = 10_000
DEFAULT_STEP_SIZE: Final[float] = 0.01
DEFAULT_PLOT_DPI: Final[int] = 300

# =============================================================================
# ОБУЧЕНИЕ PINN
# =============================================================================

EXPONENTIAL_SMOOTHING_ALPHA: Final[float] = 0.9
ADAPTIVE_WEIGHTS_WINDOW: Final[int] = 10
RAR_DEFAULT_POINTS: Final[int] = 200
EPSILON: Final[float] = 1e-12
FLOAT_EPSILON: Final[float] = 1e-8
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_TRAINING_BATCH_SIZE: Final[int] = 10
DEFAULT_EPOCHS: Final[int] = 1000

# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

DEFAULT_FIGURE_SIZE: Final[tuple[int, int]] = (10, 8)
DEFAULT_COLORMAP: Final[str] = 'jet'
SCATTER_ALPHA: Final[float] = 0.3
SCATTER_POINT_SIZE: Final[int] = 1

# =============================================================================
# UI И МЕНЮ
# =============================================================================

MENU_LINE_WIDTH: Final[int] = 60
MENU_SEPARATOR_CHAR: Final[str] = '─'

STATUS_READY_ICON: Final[str] = '✅'
STATUS_DEVELOPMENT_ICON: Final[str] = '🔄'
STATUS_PLANNED_ICON: Final[str] = '📅'

# =============================================================================
# ФАЙЛОВЫЕ ОПЕРАЦИИ
# =============================================================================

DEFAULT_ENCODING: Final[str] = 'utf-8'
JSON_INDENT: Final[int] = 4

FILE_EXT_JSON: Final[str] = '.json'
FILE_EXT_YAML: Final[str] = '.yaml'
FILE_EXT_NPY: Final[str] = '.npy'
FILE_EXT_CSV: Final[str] = '.csv'
FILE_EXT_PT: Final[str] = '.pt'

# =============================================================================
# МНОГОПРОЦЕССОРНОСТЬ
# =============================================================================

DEFAULT_NUM_PROCESSES: Final[int] = 4
QUEUE_TIMEOUT: Final[float] = 1.0

# =============================================================================
# ГРАНИЧНЫЕ УСЛОВИЯ
# =============================================================================

DEFAULT_TOP_BOUNDARY_TYPE: Final[str] = 'wall'
DEFAULT_SYMMETRY_TYPE: Final[str] = 'symmetry'

# =============================================================================
# АРХИТЕКТУРА СЕТИ
# =============================================================================

DEFAULT_HIDDEN_LAYERS: Final[list[int]] = [50, 50, 50, 50]
DEFAULT_ACTIVATION: Final[str] = 'tanh'
DEFAULT_OUTPUT_DIM: Final[int] = 3
DEFAULT_INPUT_DIM: Final[int] = 5

DEFAULT_OUTPUT_BIAS: Final[tuple[float, float, float]] = (0.1, 0.0, 0.0)

# =============================================================================
# ОТВОД (PIPE BEND)
# =============================================================================

PIPE_BEND_ARC_N_POINTS: Final[int] = 50
PIPE_BEND_DEFAULT_ANGLE: Final[float] = 90.0

# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

LOG_LEVEL_DEBUG: Final[str] = 'DEBUG'
LOG_LEVEL_INFO: Final[str] = 'INFO'
LOG_LEVEL_WARNING: Final[str] = 'WARNING'
LOG_LEVEL_ERROR: Final[str] = 'ERROR'
LOG_FORMAT: Final[str] = '%(asctime)s - %(levelname)s - %(message)s'

# =============================================================================
# СИСТЕМНЫЕ
# =============================================================================

IS_WINDOWS: Final[bool] = sys.platform == 'win32'

# TQDM availability (non-Final due to try/except reassignment)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Torch availability (non-Final due to try/except reassignment)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    'CLASSIFICATION_TOLERANCE_FACTOR',
    'MAX_ATTEMPTS_MULTIPLIER',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_STEP_SIZE',
    'DEFAULT_PLOT_DPI',
    'EXPONENTIAL_SMOOTHING_ALPHA',
    'ADAPTIVE_WEIGHTS_WINDOW',
    'RAR_DEFAULT_POINTS',
    'EPSILON',
    'FLOAT_EPSILON',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_TRAINING_BATCH_SIZE',
    'DEFAULT_EPOCHS',
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_COLORMAP',
    'SCATTER_ALPHA',
    'SCATTER_POINT_SIZE',
    'MENU_LINE_WIDTH',
    'MENU_SEPARATOR_CHAR',
    'STATUS_READY_ICON',
    'STATUS_DEVELOPMENT_ICON',
    'STATUS_PLANNED_ICON',
    'DEFAULT_ENCODING',
    'JSON_INDENT',
    'FILE_EXT_JSON',
    'FILE_EXT_YAML',
    'FILE_EXT_NPY',
    'FILE_EXT_CSV',
    'FILE_EXT_PT',
    'DEFAULT_NUM_PROCESSES',
    'QUEUE_TIMEOUT',
    'DEFAULT_TOP_BOUNDARY_TYPE',
    'DEFAULT_SYMMETRY_TYPE',
    'DEFAULT_HIDDEN_LAYERS',
    'DEFAULT_ACTIVATION',
    'DEFAULT_OUTPUT_DIM',
    'DEFAULT_INPUT_DIM',
    'DEFAULT_OUTPUT_BIAS',
    'PIPE_BEND_ARC_N_POINTS',
    'PIPE_BEND_DEFAULT_ANGLE',
    'LOG_LEVEL_DEBUG',
    'LOG_LEVEL_INFO',
    'LOG_LEVEL_WARNING',
    'LOG_LEVEL_ERROR',
    'LOG_FORMAT',
    'IS_WINDOWS',
    'TQDM_AVAILABLE',
    'TORCH_AVAILABLE',
]
