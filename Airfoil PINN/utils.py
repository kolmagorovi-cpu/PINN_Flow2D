#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для проекта PINN Learning.
Содержит вспомогательные функции: проверка памяти, безопасное вычисление градиента,
установка seed для воспроизводимости.

Автор: Колмагоров И.Д.
Версия: 3.260226 (финальная)
"""

import torch
import numpy as np
import random
import logging
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел (torch, numpy, random)
    для обеспечения воспроизводимости.

    Args:
        seed: целое число – значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Для детерминированных алгоритмов (может снизить производительность)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seed установлен: {seed}")


def check_memory_usage(device: torch.device,
                       required_gb: Optional[float] = None,
                       return_free: bool = False) -> Optional[float]:
    """
    Проверка доступной памяти на устройстве.
    Выводит предупреждение, если памяти недостаточно.
    Может возвращать объём свободной памяти.

    Args:
        device: устройство (cuda или cpu)
        required_gb: требуемый объём памяти в ГБ (опционально)
        return_free: если True, возвращает объём свободной памяти в ГБ

    Returns:
        float: свободная память в ГБ, если return_free=True, иначе None
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


def safe_grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True, silent: bool = False) -> torch.Tensor:
    """
    Безопасное вычисление градиента dy/dx.
    Если x не требует градиента или произошла ошибка, возвращает ноль.
    В режиме silent предупреждения не выводятся (полезно для ожидаемых ошибок на границах).

    Args:
        y: выходной тензор (скаляр или форма, совместимая с x)
        x: входной тензор, по которому берётся производная
        create_graph: создавать ли граф для дальнейшего дифференцирования
        silent: если True, подавляет предупреждения об ошибках

    Returns:
        Тензор градиента той же формы, что и x, или нулевой тензор при ошибке.
    """
    try:
        if not x.requires_grad:
            if not silent:
                logger.warning(f"safe_grad: x не требует градиента (requires_grad=False). Возвращаю ноль.")
            return torch.zeros_like(x)
        g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=create_graph, allow_unused=True)[0]
        if g is None:
            if not silent:
                logger.warning(f"safe_grad: градиент равен None (allow_unused=True). Возвращаю ноль.")
            return torch.zeros_like(x)
        return g
    except RuntimeError as e:
        if not silent:
            logger.warning(f"Ошибка при вычислении градиента: {e}")
        return torch.zeros_like(x)