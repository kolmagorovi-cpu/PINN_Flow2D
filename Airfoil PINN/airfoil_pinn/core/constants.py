#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Общие константы проекта Airfoil PINN.

Этот модуль содержит все магические числа и конфигурационные константы,
используемые в различных частях проекта для улучшения читаемости и
поддерживаемости кода.

Example:
    >>> from airfoil_pinn.core.constants import CLASSIFICATION_TOLERANCE_FACTOR, DEFAULT_BATCH_SIZE
    >>> tol = step * CLASSIFICATION_TOLERANCE_FACTOR
    >>> batch = DEFAULT_BATCH_SIZE
"""

import sys
from typing import Final

# =============================================================================
# ГЕОМЕТРИЯ И ДИСКРЕТИЗАЦИЯ
# =============================================================================

#: Точность классификации границ (доля от шага)
CLASSIFICATION_TOLERANCE_FACTOR: Final[float] = 0.1

#: Множитель для максимального количества попыток генерации точек
MAX_ATTEMPTS_MULTIPLIER: Final[int] = 20

#: Размер батча по умолчанию для генерации точек
DEFAULT_BATCH_SIZE: Final[int] = 10_000

#: Шаг дискретизации по умолчанию [м]
DEFAULT_STEP_SIZE: Final[float] = 0.01

#: DPI для сохранения графиков
DEFAULT_PLOT_DPI: Final[int] = 300

# =============================================================================
# ОБУЧЕНИЕ PINN
# =============================================================================

#: Коэффициент сглаживания для экспоненциального скользящего среднего
EXPONENTIAL_SMOOTHING_ALPHA: Final[float] = 0.9

#: Размер окна для адаптивных весов потерь
ADAPTIVE_WEIGHTS_WINDOW: Final[int] = 10

#: Количество точек по умолчанию для RAR (Residual Adaptive Refinement)
RAR_DEFAULT_POINTS: Final[int] = 200

#: Эпсилон для защиты от деления на ноль
EPSILON: Final[float] = 1e-12

#: Эпсилон для сравнения чисел с плавающей точкой
FLOAT_EPSILON: Final[float] = 1e-8

#: learning_rate по умолчанию
DEFAULT_LEARNING_RATE: Final[float] = 1e-3

#: Размер батча для обучения по умолчанию
DEFAULT_TRAINING_BATCH_SIZE: Final[int] = 10

#: Количество эпох по умолчанию
DEFAULT_EPOCHS: Final[int] = 1000

# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

#: Размеры фигуры по умолчанию (width, height)
DEFAULT_FIGURE_SIZE: Final[tuple[int, int]] = (10, 8)

#: Цветовая карта по умолчанию
DEFAULT_COLORMAP: Final[str] = 'jet'

#: Прозрачность для scatter plots
SCATTER_ALPHA: Final[float] = 0.3

#: Размер точки для scatter plots
SCATTER_POINT_SIZE: Final[int] = 1

# =============================================================================
# UI И МЕНЮ
# =============================================================================

#: Ширина заголовков меню
MENU_LINE_WIDTH: Final[int] = 60

#: Символ разделителя меню
MENU_SEPARATOR_CHAR: Final[str] = '─'

#: Символы для статусов модулей
STATUS_READY_ICON: Final[str] = '✅'
STATUS_DEVELOPMENT_ICON: Final[str] = '🔄'
STATUS_PLANNED_ICON: Final[str] = '📅'

#: Символы для вывода
CHECKMARK_ICON: Final[str] = '✓'
CROSS_ICON: Final[str] = '❌'
WARNING_ICON: Final[str] = '⚠️'
INFO_ICON: Final[str] = '💡'
ARROW_ICON: Final[str] = '↳'

# =============================================================================
# ФАЙЛОВЫЕ ОПЕРАЦИИ
# =============================================================================

#: Кодировка файлов по умолчанию
DEFAULT_ENCODING: Final[str] = 'utf-8'

#: Отступ для JSON файлов
JSON_INDENT: Final[int] = 4

#: Расширения файлов
FILE_EXT_JSON: Final[str] = '.json'
FILE_EXT_YAML: Final[str] = '.yaml'
FILE_EXT_NPY: Final[str] = '.npy'
FILE_EXT_CSV: Final[str] = '.csv'
FILE_EXT_PT: Final[str] = '.pt'  # PyTorch checkpoint

# =============================================================================
# МНОГОПРОЦЕССОРНОСТЬ
# =============================================================================

#: Количество процессов по умолчанию
DEFAULT_NUM_PROCESSES: Final[int] = 4

#: Таймаут для операций с очередями (секунды)
QUEUE_TIMEOUT: Final[float] = 1.0

# =============================================================================
# ГРАНИЧНЫЕ УСЛОВИЯ
# =============================================================================

#: Типы границ по умолчанию
DEFAULT_TOP_BOUNDARY_TYPE: Final[str] = 'wall'
DEFAULT_SYMMETRY_TYPE: Final[str] = 'symmetry'

# =============================================================================
# АРХИТЕКТУРА СЕТИ
# =============================================================================

#: Архитектура по умолчанию
DEFAULT_HIDDEN_LAYERS: Final[list[int]] = [50, 50, 50, 50]
DEFAULT_ACTIVATION: Final[str] = 'tanh'
DEFAULT_OUTPUT_DIM: Final[int] = 3  # (u, v, p)
DEFAULT_INPUT_DIM: Final[int] = 5   # legacy fallback; actual input_dim is built from input_params

#: Смещение выхода по умолчанию (u, v, p)
DEFAULT_OUTPUT_BIAS: Final[tuple[float, float, float]] = (0.1, 0.0, 0.0)

# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

#: Уровни логирования
LOG_LEVEL_DEBUG: Final[str] = 'DEBUG'
LOG_LEVEL_INFO: Final[str] = 'INFO'
LOG_LEVEL_WARNING: Final[str] = 'WARNING'
LOG_LEVEL_ERROR: Final[str] = 'ERROR'

#: Формат логов по умолчанию
LOG_FORMAT: Final[str] = '%(asctime)s - %(levelname)s - %(message)s'

# =============================================================================
# СИСТЕМНЫЕ
# =============================================================================

#: Поддержка Windows
IS_WINDOWS: Final[bool] = sys.platform == 'win32'

#: Поддержка tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE: Final[bool] = True
except ImportError:
    TQDM_AVAILABLE: Final[bool] = False

#: Поддержка torch
try:
    import torch
    TORCH_AVAILABLE: Final[bool] = True
except ImportError:
    TORCH_AVAILABLE: Final[bool] = False

#: Список всех экспортируемых констант
__all__: Final[list[str]] = [
    # Геометрия и дискретизация
    'CLASSIFICATION_TOLERANCE_FACTOR',
    'MAX_ATTEMPTS_MULTIPLIER',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_STEP_SIZE',
    'DEFAULT_PLOT_DPI',
    # Обучение PINN
    'EXPONENTIAL_SMOOTHING_ALPHA',
    'ADAPTIVE_WEIGHTS_WINDOW',
    'RAR_DEFAULT_POINTS',
    'EPSILON',
    'FLOAT_EPSILON',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_TRAINING_BATCH_SIZE',
    'DEFAULT_EPOCHS',
    # Визуализация
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_COLORMAP',
    'SCATTER_ALPHA',
    'SCATTER_POINT_SIZE',
    # UI и меню
    'MENU_LINE_WIDTH',
    'MENU_SEPARATOR_CHAR',
    'STATUS_READY_ICON',
    'STATUS_DEVELOPMENT_ICON',
    'STATUS_PLANNED_ICON',
    'CHECKMARK_ICON',
    'CROSS_ICON',
    'WARNING_ICON',
    'INFO_ICON',
    'ARROW_ICON',
    # Файловые операции
    'DEFAULT_ENCODING',
    'JSON_INDENT',
    'FILE_EXT_JSON',
    'FILE_EXT_YAML',
    'FILE_EXT_NPY',
    'FILE_EXT_CSV',
    'FILE_EXT_PT',
    # Многопроцессорность
    'DEFAULT_NUM_PROCESSES',
    'QUEUE_TIMEOUT',
    # Граничные условия
    'DEFAULT_TOP_BOUNDARY_TYPE',
    'DEFAULT_SYMMETRY_TYPE',
    # Архитектура сети
    'DEFAULT_HIDDEN_LAYERS',
    'DEFAULT_ACTIVATION',
    'DEFAULT_OUTPUT_DIM',
    'DEFAULT_INPUT_DIM',
    'DEFAULT_OUTPUT_BIAS',
    # Логирование
    'LOG_LEVEL_DEBUG',
    'LOG_LEVEL_INFO',
    'LOG_LEVEL_WARNING',
    'LOG_LEVEL_ERROR',
    'LOG_FORMAT',
    # Системные
    'IS_WINDOWS',
    'TQDM_AVAILABLE',
    'TORCH_AVAILABLE',
]
