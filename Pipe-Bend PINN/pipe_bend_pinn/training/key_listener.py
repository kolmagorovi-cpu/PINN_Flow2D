from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль клавиатурного слушателя для Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
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


class KeyListener:
    """Слушает нажатия клавиш в отдельном потоке."""

    def __init__(self):
        self.save_requested = False
        self.edit_requested = False
        self.reset_requested = False
        self.checkpoint_requested = False
        self.lr_increase_requested = False
        self.lr_decrease_requested = False
        self.lr_set_requested = False
        self.running = True
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if KEYBOARD_AVAILABLE:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info(
                "Клавиатурный слушатель запущен.\n"
                "  Ctrl+P – сохранить графики\n"
                "  Ctrl+S – сохранить чекпоинт\n"
                "  Ctrl+R – сброс оптимизатора\n"
                "  Ctrl+L – увеличить LR\n"
                "  Ctrl+K – уменьшить LR"
            )
        else:
            logger.warning("Модуль 'keyboard' не установлен. pip install keyboard")

    def _run(self):
        try:
            keyboard.add_hotkey('ctrl+p', self._on_save_key)
            keyboard.add_hotkey('ctrl+s', self._on_checkpoint_key)
            keyboard.add_hotkey('ctrl+r', self._on_reset_key)
            keyboard.add_hotkey('ctrl+l', self._on_lr_increase_key)
            keyboard.add_hotkey('ctrl+k', self._on_lr_decrease_key)
            while self.running:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Ошибка в клавиатурном слушателе: {e}")

    def _on_save_key(self):
        self.save_requested = True

    def _on_checkpoint_key(self):
        self.checkpoint_requested = True

    def _on_reset_key(self):
        self.reset_requested = True

    def _on_lr_increase_key(self):
        self.lr_increase_requested = True

    def _on_lr_decrease_key(self):
        self.lr_decrease_requested = True

    def stop(self):
        self.running = False
        if KEYBOARD_AVAILABLE:
            try:
                keyboard.unhook_all()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)