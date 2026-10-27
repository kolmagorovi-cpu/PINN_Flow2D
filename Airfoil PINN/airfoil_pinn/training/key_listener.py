#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль клавиатурного слушателя для проекта PINN Learning.
Обеспечивает интерактивное управление процессом обучения через горячие клавиши.
Исправленная версия: добавлен флаг для прямого ввода LR, улучшено логирование.
"""

import threading
import time
import logging
from typing import Optional

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["KeyListener"]


class KeyListener:
    """
    Слушает нажатия клавиш в отдельном потоке и устанавливает соответствующие флаги.
    Основной поток обучения должен периодически проверять эти флаги и выполнять действия.

    Поддерживаемые комбинации:
        'ctrl+p'     – сохранить текущие графики и поля
        'ctrl+w'     – редактировать веса потерь
        'ctrl+r'     – сбросить оптимизатор с шумом (с возможностью изменения шума)
        'ctrl+s'     – сохранить чекпоинт
        'ctrl+l'     – увеличить learning rate
        'ctrl+k'     – уменьшить learning rate
        'ctrl+u'     – установить learning rate вручную (запрос значения)
    """
    def __init__(self):
        self.save_requested = False
        self.edit_requested = False
        self.reset_requested = False
        self.checkpoint_requested = False
        self.lr_increase_requested = False
        self.lr_decrease_requested = False
        self.lr_set_requested = False   # новый флаг для прямого ввода LR
        self.running = True
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Запускает поток прослушивания, если библиотека keyboard доступна."""
        if KEYBOARD_AVAILABLE:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(
                "Клавиатурный слушатель запущен.\n"
                "  Ctrl+P      – сохранить графики и поля\n"
                "  Ctrl+W      – редактировать веса\n"
                "  Ctrl+R      – сброс оптимизатора с шумом (с запросом шума)\n"
                "  Ctrl+S      – сохранить чекпоинт\n"
                "  Ctrl+L      – увеличить learning rate\n"
                "  Ctrl+K      – уменьшить learning rate\n"
                "  Ctrl+U      – установить learning rate вручную"
            )
        else:
            logger.warning(
                "Модуль 'keyboard' не установлен. Интерактивные функции недоступны.\n"
                "Добавьте пакет keyboard в общее окружение D:\\Python\\BaseEnv."
            )

    def _run(self):
        """Тело потока: добавляет горячие клавиши и ожидает."""
        try:
            keyboard.add_hotkey('ctrl+p', self._on_save_key)
            keyboard.add_hotkey('ctrl+w', self._on_edit_key)
            keyboard.add_hotkey('ctrl+r', self._on_reset_key)
            keyboard.add_hotkey('ctrl+s', self._on_checkpoint_key)
            keyboard.add_hotkey('ctrl+l', self._on_lr_increase_key)
            keyboard.add_hotkey('ctrl+k', self._on_lr_decrease_key)
            keyboard.add_hotkey('ctrl+u', self._on_lr_set_key)  # новая комбинация
            while self.running:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Ошибка в клавиатурном слушателе: {e}")

    def _on_save_key(self):
        self.save_requested = True
        logger.debug("Нажата комбинация Ctrl+P – запрос на сохранение результатов")

    def _on_edit_key(self):
        self.edit_requested = True
        logger.debug("Нажата комбинация Ctrl+W – запрос на редактирование весов")

    def _on_reset_key(self):
        self.reset_requested = True
        logger.debug("Нажата комбинация Ctrl+R – запрос на сброс оптимизатора")

    def _on_checkpoint_key(self):
        self.checkpoint_requested = True
        logger.debug("Нажата комбинация Ctrl+S – запрос на сохранение чекпоинта")

    def _on_lr_increase_key(self):
        self.lr_increase_requested = True
        logger.debug("Нажата комбинация Ctrl+L – запрос на увеличение LR")

    def _on_lr_decrease_key(self):
        self.lr_decrease_requested = True
        logger.debug("Нажата комбинация Ctrl+K – запрос на уменьшение LR")

    def _on_lr_set_key(self):
        self.lr_set_requested = True
        logger.debug("Нажата комбинация Ctrl+U – запрос на установку LR вручную")

    def stop(self):
        """Останавливает поток и удаляет все горячие клавиши."""
        self.running = False
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except Exception as e:
                logger.warning(f"Ошибка при отключении горячих клавиш: {e}")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
