from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Обработчик интерактивных действий пользователя.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InteractiveHandler:
    """Обработка горячих клавиш и интерактивных запросов."""

    def __init__(self, trainer, metrics_collector, visualizer, checkpoint_manager,
                 key_listener, loss_computer, data_handler, model, device, dtype,
                 config, sa_weights=None, plots_dir="Pictures"):
        self.trainer = trainer
        self.metrics_collector = metrics_collector
        self.visualizer = visualizer
        self.checkpoint_manager = checkpoint_manager
        self.key_listener = key_listener
        self.loss_computer = loss_computer
        self.data_handler = data_handler
        self.model = model
        self.device = device
        self.dtype = dtype
        self.config = config
        self.sa_weights = sa_weights
        self.plots_dir = plots_dir
        self.optimizer_reset_requested = False

    def handle_keys(self, epoch: int) -> None:
        """Проверяет и обрабатывает нажатые горячие клавиши."""
        if self.key_listener is None:
            return

        if self.key_listener.save_requested:
            self._save_current_results(epoch)
            self.key_listener.save_requested = False

        if self.key_listener.checkpoint_requested:
            self._save_checkpoint(epoch)
            self.key_listener.checkpoint_requested = False

        if self.key_listener.reset_requested:
            self._handle_reset()
            self.key_listener.reset_requested = False

        if self.key_listener.lr_increase_requested:
            self._adjust_lr(1.5)
            self.key_listener.lr_increase_requested = False

        if self.key_listener.lr_decrease_requested:
            self._adjust_lr(0.5)
            self.key_listener.lr_decrease_requested = False

    def _save_current_results(self, epoch: int) -> None:
        """Сохраняет текущие результаты."""
        try:
            from pipe_bend_pinn.training.visualizer import Visualizer
            self.visualizer.save_plots(self.plots_dir, epoch)
            logger.info(f"Результаты сохранены на эпохе {epoch}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")

    def _save_checkpoint(self, epoch: int) -> None:
        """Сохраняет чекпоинт."""
        try:
            self.checkpoint_manager.save(
                filename=f"pinn_epoch_{epoch}.pt",
                model=self.model,
                optimizer=self.trainer.optimizer,
                scheduler=None,
                loss_weights=self.loss_computer.loss_weights,
                history=self.metrics_collector.history,
                sa_weights=self.sa_weights,
                epoch=epoch,
                config=self.config
            )
            logger.info(f"Чекпоинт сохранён на эпохе {epoch}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении чекпоинта: {e}")

    def _handle_reset(self) -> None:
        """Обрабатывает запрос на сброс оптимизатора."""
        try:
            noise_str = input("Амплитуда шума при сбросе [0.01]: ").strip()
            noise = float(noise_str) if noise_str else 0.01
            self.optimizer_reset_requested = True
            logger.info(f"Запрошен сброс оптимизатора (noise={noise})")
        except (ValueError, EOFError):
            logger.warning("Некорректный ввод для шума")

    def _adjust_lr(self, factor: float) -> None:
        """Изменяет learning rate."""
        for param_group in self.trainer.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = old_lr * factor
            logger.info(f"LR изменён: {old_lr:.6f} → {param_group['lr']:.6f} (x{factor})")