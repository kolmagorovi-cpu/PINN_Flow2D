#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль interactive для проекта PINN Learning.
Содержит класс InteractiveHandler, обрабатывающий интерактивные действия пользователя
(горячие клавиши) во время обучения.
"""

import logging
from typing import Optional, Dict, Any

import torch
import numpy as np

from airfoil_pinn.training.contract import build_training_contract

logger = logging.getLogger(__name__)

__all__ = ["InteractiveHandler"]


class InteractiveHandler:
    """
    Обрабатывает пользовательский ввод через горячие клавиши во время обучения.
    Проверяет флаги KeyListener и выполняет соответствующие действия:
    сохранение графиков, редактирование весов, сброс оптимизатора,
    сохранение чекпоинта, изменение learning rate.
    """

    def __init__(
        self,
        trainer,                # экземпляр Trainer
        metrics_collector,      # экземпляр MetricsCollector
        visualizer,             # экземпляр Visualizer
        checkpoint_manager,     # экземпляр CheckpointManager
        key_listener,           # экземпляр KeyListener
        loss_computer,          # экземпляр LossComputer
        data_handler,           # экземпляр DataHandler
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
        sa_weights: Optional[torch.nn.Module] = None,
        plots_dir: Optional[str] = None          # директория для сохранения графиков
    ):
        """
        Args:
            trainer: объект Trainer (для доступа к модели, оптимизатору, loss_weights и т.д.)
            metrics_collector: объект MetricsCollector (для истории)
            visualizer: объект Visualizer (для сохранения графиков)
            checkpoint_manager: объект CheckpointManager (для сохранения чекпоинтов)
            key_listener: объект KeyListener (флаги нажатий)
            loss_computer: объект LossComputer (для обновления весов потерь)
            data_handler: объект DataHandler (для информации о границах)
            model: нейросетевая модель
            device: устройство
            dtype: тип данных
            config: полная конфигурация (для доступа к настройкам сохранения и т.д.)
            sa_weights: SA-веса (опционально)
            plots_dir: директория для сохранения графиков (если None, используется директория из visualizer)
        """
        self.trainer = trainer
        self.metrics = metrics_collector
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

        # Флаг для запроса пересоздания оптимизатора после сброса с шумом
        self.optimizer_reset_requested = False

    def handle_keys(self, current_epoch: int) -> None:
        """
        Проверяет флаги KeyListener и вызывает соответствующие обработчики.
        Должен вызываться в цикле обучения после каждой эпохи.

        Args:
            current_epoch: текущий номер эпохи (для сохранения и логов)
        """
        if self.key_listener.save_requested:
            self._save_current_results(current_epoch)
            self.key_listener.save_requested = False

        if self.key_listener.edit_requested:
            self._edit_weights_interactive()
            self.key_listener.edit_requested = False

        if self.key_listener.reset_requested:
            self._reset_optimizer_interactive()
            self.key_listener.reset_requested = False

        if self.key_listener.checkpoint_requested:
            self._save_checkpoint_manual(current_epoch)
            self.key_listener.checkpoint_requested = False

        if self.key_listener.lr_increase_requested:
            self._adjust_lr(1.2)
            self.key_listener.lr_increase_requested = False

        if self.key_listener.lr_decrease_requested:
            self._adjust_lr(0.8)
            self.key_listener.lr_decrease_requested = False

        if self.key_listener.lr_set_requested:
            self._set_lr_interactive()
            self.key_listener.lr_set_requested = False

    # ------------------------------------------------------------------
    # Обработчики конкретных действий
    # ------------------------------------------------------------------
    def _save_current_results(self, epoch: int) -> None:
        """
        Сохраняет текущие графики и поля (вызывается по Ctrl+P).
        """
        logger.info("🖫 Сохранение текущих результатов (Ctrl+P)...")
        # Сохраняем графики
        if self.plots_dir:
            self.visualizer.save_plots(self.plots_dir, epoch)
        else:
            # Если директория не задана, используем внутреннюю логику visualizer (может отсутствовать)
            logger.warning("Директория для сохранения графиков не указана, пропускаем сохранение графиков.")

        # Получаем предсказания на всём датасете
        dataset = self.data_handler.dataset
        x_norm = dataset.x.to(self.device, dtype=self.dtype)
        y_norm = dataset.y.to(self.device, dtype=self.dtype)
        sdf_norm = dataset.sdf.to(self.device, dtype=self.dtype)
        bt = dataset.boundary_types.to(self.device, dtype=self.dtype) if dataset.boundary_types is not None else None

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.loss_computer.build_input_tensor(x_norm, y_norm, sdf_norm, bt)
            out = self.model(input_tensor)
            u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]

            # Применяем hard constraints, если нужно (для отображения правильных значений на границах)
            if self.loss_computer.bc_type == 'hard':
                # Маски и значения нужно получить из датасета
                u, v, p = self.loss_computer.apply_hard_constraints(
                    u_raw, v_raw, p_raw,
                    dataset.mask_u_fixed.to(self.device).view(-1, 1),
                    dataset.mask_v_fixed.to(self.device).view(-1, 1),
                    dataset.mask_p_fixed.to(self.device).view(-1, 1),
                    dataset.u_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                    dataset.v_bc.to(self.device, dtype=self.dtype).view(-1, 1),
                    dataset.p_bc.to(self.device, dtype=self.dtype).view(-1, 1)
                )
            else:
                u, v, p = u_raw, v_raw, p_raw
        if was_training:
            self.model.train()

        # Сохраняем поля (нормализованные и физические)
        if self.plots_dir:
            self.visualizer.plot_final_fields_from_tensors(
                u, v, p, x_norm, y_norm, self.data_handler,
                self.plots_dir, epoch,
                boundary_types=bt,
                boundary_type_names=self.data_handler.boundary_type_names,
                plot_physical=False
            )
        logger.info("Результаты сохранены.")

    def _edit_weights_interactive(self) -> None:
        """
        Интерактивное редактирование весов функции потерь (Ctrl+W).
        """
        print("\n" + "=" * 60)
        print("РЕДАКТИРОВАНИЕ ВЕСОВ ПОТЕРЬ".center(60))
        print("=" * 60)
        print("Текущие веса (PDE и BC):")
        for key, value in self.loss_computer.loss_weights.items():
            print(f"  {key}: {value:.2e}")
        if self.loss_computer.integral_condition:
            print(f"  integral_weight: {self.loss_computer.integral_weight:.2e}")
        if self.loss_computer.fvm is not None:
            print(f"  fvm_weight: {self.loss_computer.fvm_weight:.2e}")
        print("\nВведите новые значения (Enter оставляет текущее). Для выхода без изменений введите 'q'.")

        new_weights = {}
        for key in list(self.loss_computer.loss_weights.keys()):
            val = input(f"{key} [{self.loss_computer.loss_weights[key]:.2e}]: ").strip()
            if val.lower() == 'q':
                print("Редактирование отменено.")
                return
            if val:
                try:
                    new_weights[key] = float(val)
                except ValueError:
                    print("Неверный формат, вес оставлен без изменений.")
            else:
                new_weights[key] = self.loss_computer.loss_weights[key]

        # Обновляем веса в loss_computer
        self.loss_computer.loss_weights.update(new_weights)

        # Интегральное условие
        if self.loss_computer.integral_condition:
            val = input(f"integral_weight [{self.loss_computer.integral_weight:.2e}]: ").strip()
            if val.lower() == 'q':
                print("Редактирование отменено.")
                return
            if val:
                try:
                    self.loss_computer.integral_weight = float(val)
                except ValueError:
                    print("Неверный формат, вес оставлен без изменений.")

        # FVM-потеря
        if self.loss_computer.fvm is not None:
            val = input(f"fvm_weight [{self.loss_computer.fvm_weight:.2e}]: ").strip()
            if val.lower() == 'q':
                print("Редактирование отменено.")
                return
            if val:
                try:
                    self.loss_computer.fvm_weight = float(val)
                except ValueError:
                    print("Неверный формат, вес оставлен без изменений.")

        print("Веса обновлены.")
        print("=" * 60)

    def _reset_optimizer_interactive(self) -> None:
        """
        Сброс оптимизатора с добавлением шума (Ctrl+R).
        Запрашивает новую амплитуду шума.
        """
        if self.trainer.using_lbfgs:
            logger.info("Сброс оптимизатора недоступен в режиме L-BFGS.")
            return

        new_noise = input(f"Введите новую амплитуду шума (текущая {self.trainer.reset_noise}): ").strip()
        if new_noise:
            try:
                self.trainer.reset_noise = float(new_noise)
            except ValueError:
                logger.warning("Неверное значение, используется текущее.")

        logger.info("🔄 Ручной сброс оптимизатора с шумом...")
        self.trainer.reset_optimizer(reset_noise=self.trainer.reset_noise)
        # Устанавливаем флаг, чтобы solver пересоздал оптимизатор при следующей возможности
        self.optimizer_reset_requested = True

    def _save_checkpoint_manual(self, epoch: int) -> None:
        """
        Сохраняет чекпоинт по запросу пользователя (Ctrl+S).
        """
        logger.info("💾 Сохранение чекпоинта по запросу...")
        training_contract = build_training_contract(
            self.config,
            self.data_handler,
            self.loss_computer,
        )
        self.checkpoint_manager.save(
            filename=f"pinn_manual_epoch_{epoch}.pt",
            model=self.model,
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler if hasattr(self.trainer, 'scheduler') else None,
            loss_weights=self.loss_computer.loss_weights,
            history=self.metrics.history,
            sa_weights=self.sa_weights,
            epoch=epoch,
            config=self.config,
            extra_state={
                "training_contract": training_contract.to_metadata(),
                "data_handler_state": self.data_handler.get_state(),
                "trainer_state": self.trainer.get_state(),
                "using_lbfgs": self.trainer.using_lbfgs,
            },
        )

    def _adjust_lr(self, factor: float) -> None:
        """
        Изменяет learning rate, умножая текущее значение на factor.

        Args:
            factor: множитель (>1 для увеличения, <1 для уменьшения)
        """
        if self.trainer.using_lbfgs:
            logger.info("Изменение LR недоступно в режиме L-BFGS.")
            return
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] *= factor
        logger.info(f"Learning rate изменён на {self.trainer.optimizer.param_groups[0]['lr']:.2e}")

    def _set_lr_interactive(self) -> None:
        """
        Устанавливает learning rate вручную (Ctrl+U).
        """
        if self.trainer.using_lbfgs:
            logger.info("Изменение LR недоступно в режиме L-BFGS.")
            return
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        new_lr = input(f"Введите новое значение learning rate (текущее {current_lr:.2e}): ").strip()
        if new_lr:
            try:
                new_lr_float = float(new_lr)
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = new_lr_float
                logger.info(f"Learning rate установлен на {new_lr_float:.2e}")
            except ValueError:
                logger.warning("Неверное значение, LR не изменён.")
