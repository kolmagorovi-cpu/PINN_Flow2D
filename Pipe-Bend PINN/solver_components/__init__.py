"""
Компоненты решателя для Pipe-Bend PINN.
"""

from .loss_computer import LossComputer
from .trainer import Trainer
from .metrics import MetricsCollector
from .optimizers import LocalMinimaDetector, CyclicLR
from .interactive import InteractiveHandler