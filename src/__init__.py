# LPO-Emotion-VLM: Listwise Preference Optimization for Fine-Grained Affective Intelligence
# in Vision-Language Models

from .lpo_loss import plackett_luce_loss, compute_pl_rewards, compute_lpo_metrics
from .lpo_dataset import LPODataset, LPODataCollator, make_lpo_data_module
from .lpo_trainer import LPOTrainer

__version__ = "1.0.0"

__all__ = [
    "plackett_luce_loss",
    "compute_pl_rewards",
    "compute_lpo_metrics",
    "LPODataset",
    "LPODataCollator",
    "make_lpo_data_module",
    "LPOTrainer",
]
