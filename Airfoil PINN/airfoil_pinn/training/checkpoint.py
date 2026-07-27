#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Управление чекпоинтами для проекта PINN Learning.
Содержит класс CheckpointManager для сохранения и загрузки состояния модели,
оптимизатора, истории и конфигурации.

Автор: Колмагоров И.Д.
Версия: 3.260226 (финальная)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Optional, Any, Union

from airfoil_pinn.training.models import SelfAdaptiveWeights

logger = logging.getLogger(__name__)

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """
    Сохраняет и загружает чекпоинты, включая состояние модели, оптимизатора,
    планировщика, веса потерь, историю обучения и конфигурацию.
    Модель сохраняется без обёртки DataParallel (только base_model), что обеспечивает
    совместимость при загрузке на другом количестве GPU.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: директория для сохранения чекпоинтов.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Returns the real model behind DataParallel or torch.compile wrappers."""
        if isinstance(model, nn.DataParallel):
            model = model.module
        return getattr(model, '_orig_mod', model)

    def save(self,
             filename: str,
             model: nn.Module,
             optimizer: optim.Optimizer,
             scheduler: Any,
             loss_weights: Dict,
             history: Dict,
             sa_weights: Optional[SelfAdaptiveWeights] = None,
             epoch: Optional[int] = None,
             config: Optional[Dict] = None,
             extra_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Сохраняет чекпоинт.

        Args:
            filename: имя файла (например, 'pinn_epoch_100.pt')
            model: модель PyTorch (может быть обёрнута в DataParallel)
            optimizer: оптимизатор
            scheduler: планировщик learning rate
            loss_weights: словарь весов потерь
            history: словарь истории обучения
            sa_weights: SA-веса (если используются)
            epoch: номер эпохи
            config: полная конфигурация проекта
        """
        # Извлекаем базовую модель, если используются DataParallel/torch.compile
        base_model = self._unwrap_model(model)
        state_dict = base_model.state_dict()
        checkpoint = {
            'checkpoint_schema': {
                'name': 'airfoil-pinn.checkpoint',
                'version': 2,
            },
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_weights': loss_weights,
            'history': history,
            'epoch': epoch,
            'config': config,
        }
        if extra_state:
            checkpoint.update(extra_state)
        if sa_weights is not None:
            checkpoint['sa_weights_state_dict'] = sa_weights.state_dict()

        path = os.path.join(self.checkpoint_dir, filename)
        temporary_path = path + ".tmp"
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise
        logger.info(f"Чекпоинт сохранён: {path} (размер state_dict: {sum(p.numel() for p in state_dict.values())} параметров)")

    def load(self, path: str, map_location: torch.device) -> Dict:
        """
        Загружает чекпоинт.

        Args:
            path: путь к файлу чекпоинта
            map_location: устройство для загрузки (например, torch.device('cuda'))

        Returns:
            словарь с содержимым чекпоинта

        Raises:
            FileNotFoundError: если файл не найден.
            RuntimeError: если файл повреждён или не может быть загружен.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Чекпоинт {path} не найден")
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"Чекпоинт загружен из {path}")
            return checkpoint
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки чекпоинта {path}: {e}")
