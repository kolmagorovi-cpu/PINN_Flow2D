#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакет утилит для проекта Airfoil PINN.
"""

from .ui_utils import (
    get_menu_choice,
    safe_input_number,
    confirm_action,
    input_with_default,
    clear_screen,
    print_header,
)
from .io_utils import (
    save_json,
    load_json,
    sanitize_filename,
    ensure_directory,
)

# Проверка доступности scipy
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    cKDTree = None
    SCIPY_AVAILABLE = False

# Проверка доступности psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import torch
import numpy as np
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seed установлен: {seed}")


def check_memory_usage(device: torch.device,
                       required_gb: Optional[float] = None,
                       return_free: bool = False) -> Optional[float]:
    """
    Проверка доступной памяти на устройстве.
    """
    free_gb = None
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / 1e9
        total_gb = total / 1e9
        logger.info(f"GPU память: свободно {free_gb:.2f} ГБ, всего {total_gb:.2f} ГБ")
        if required_gb and free_gb < required_gb:
            logger.warning(f"⚠️ Требуется ~{required_gb:.2f} ГБ, свободно только {free_gb:.2f} ГБ")
    else:
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            free_gb = mem.available / 1e9
            total_gb = mem.total / 1e9
            logger.info(f"RAM: свободно {free_gb:.2f} ГБ, всего {total_gb:.2f} ГБ")
            if required_gb and free_gb < required_gb:
                logger.warning(f"⚠️ Требуется ~{required_gb:.2f} ГБ, свободно только {free_gb:.2f} ГБ")
        else:
            logger.info("Проверка RAM недоступна (psutil не установлен)")

    if return_free:
        return free_gb
    return None


def safe_grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True, retain_graph: bool = False, silent: bool = False) -> torch.Tensor:
    """
    Безопасное вычисление градиента dy/dx.
    
    Args:
        y: зависимая переменная
        x: независимая переменная
        create_graph: если True, создаётся граф для вычисления производных высших порядков
        retain_graph: если True, граф сохраняется после вычисления (нужно для многократных вызовов)
        silent: если True, не выводить предупреждения
    
    Returns:
        dy/dx тензор той же формы, что и x
    """
    try:
        if not x.requires_grad:
            if not silent:
                logger.warning(f"safe_grad: x не требует градиента (requires_grad=False). Возвращаю ноль.")
            return torch.zeros_like(x)
        g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=create_graph, retain_graph=retain_graph, allow_unused=True)[0]
        if g is None:
            if not silent:
                logger.warning(f"safe_grad: градиент равен None (allow_unused=True). Возвращаю ноль.")
            return torch.zeros_like(x)
        return g
    except RuntimeError as e:
        if not silent:
            logger.warning(f"Ошибка при вычислении градиента: {e}")
        return torch.zeros_like(x)


__all__ = [
    # UI
    'get_menu_choice',
    'safe_input_number',
    'confirm_action',
    'input_with_default',
    'clear_screen',
    'print_header',
    # I/O
    'save_json',
    'load_json',
    'sanitize_filename',
    'ensure_directory',
    # Общие утилиты
    'check_memory_usage',
    'set_seed',
    'safe_grad',
    'SCIPY_AVAILABLE',
    'PSUTIL_AVAILABLE',
]
