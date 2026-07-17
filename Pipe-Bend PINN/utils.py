"""
Утилиты для Pipe-Bend PINN.

Содержит вспомогательные функции: проверка памяти, безопасное вычисление градиента,
установка seed для воспроизводимости.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
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
    """Устанавливает seed для всех генераторов случайных чисел."""
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
    """Проверка доступной памяти на устройстве."""
    free_gb = None
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / 1e9
        total_gb = total / 1e9
        logger.info(f"GPU память: свободно {free_gb:.2f} ГБ, всего {total_gb:.2f} ГБ")
        if required_gb and free_gb < required_gb:
            logger.warning(f"Требуется ~{required_gb:.2f} ГБ, свободно только {free_gb:.2f} ГБ")
    else:
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            free_gb = mem.available / 1e9
            total_gb = mem.total / 1e9
            logger.info(f"RAM: свободно {free_gb:.2f} ГБ, всего {total_gb:.2f} ГБ")
            if required_gb and free_gb < required_gb:
                logger.warning(f"Требуется ~{required_gb:.2f} ГБ, свободно только {free_gb:.2f} ГБ")
        else:
            logger.info("Проверка RAM недоступна")

    if return_free:
        return free_gb
    return None


def safe_grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True, silent: bool = False) -> torch.Tensor:
    """Безопасное вычисление градиента dy/dx."""
    try:
        if not x.requires_grad:
            if not silent:
                logger.warning(f"safe_grad: x не требует градиента")
            return torch.zeros_like(x)
        g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=create_graph, allow_unused=True)[0]
        if g is None:
            if not silent:
                logger.warning(f"safe_grad: градиент равен None")
            return torch.zeros_like(x)
        return g
    except RuntimeError as e:
        if not silent:
            logger.warning(f"Ошибка при вычислении градиента: {e}")
        return torch.zeros_like(x)


def clear_screen() -> None:
    """Очищает экран консоли."""
    import os, sys
    os.system('cls' if sys.platform == 'win32' else 'clear')
    print('\033c')


def safe_input_number(prompt: str, default: Optional[float] = None, allow_empty: bool = False) -> float:
    """Безопасный ввод числа."""
    while True:
        try:
            val = input(prompt).strip()
            if allow_empty and val == '':
                return default if default is not None else 0.0
            return float(val) if val else (default if default is not None else 0.0)
        except ValueError:
            print("Введите корректное число.")


def input_with_default(prompt: str, default: str) -> str:
    """Ввод строки со значением по умолчанию."""
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def confirm_action(prompt: str, default: bool = False) -> bool:
    """Запрос подтверждения действия."""
    suffix = " (Y/n): " if default else " (y/N): "
    ans = input(prompt + suffix).strip().lower()
    if default:
        return ans not in ['n', 'no']
    return ans in ['y', 'yes']


def print_header(title: str) -> None:
    """Выводит заголовок."""
    print(f"\n{'='*60}\n{title.center(60)}\n{'='*60}")


def get_menu_choice(options: dict) -> str:
    """Выводит меню и возвращает выбор пользователя."""
    for key, desc in options.items():
        print(f"{key}. {desc}")
    return input("Ваш выбор: ").strip()