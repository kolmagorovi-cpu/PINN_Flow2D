from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Управление чекпоинтами для Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Optional, Any

from pipe_bend_pinn.training.models import SelfAdaptiveWeights

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Сохраняет и загружает чекпоинты."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, filename: str, model: nn.Module, optimizer: optim.Optimizer,
             scheduler: Any, loss_weights: Dict, history: Dict,
             sa_weights: Optional[SelfAdaptiveWeights] = None,
             epoch: Optional[int] = None, config: Optional[Dict] = None) -> None:
        if isinstance(model, nn.DataParallel):
            base_model = model.module
        else:
            base_model = model

        state_dict = base_model.state_dict()
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_weights': loss_weights,
            'history': history,
            'epoch': epoch,
            'config': config,
        }
        if sa_weights is not None:
            checkpoint['sa_weights_state_dict'] = sa_weights.state_dict()

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Чекпоинт сохранён: {path}")

    def load(self, path: str, map_location: torch.device) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Чекпоинт {path} не найден")
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"Чекпоинт загружен из {path}")
            return checkpoint
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки чекпоинта {path}: {e}")