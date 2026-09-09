"""
SIH26012 AI Module
Deep learning pipelines for cadastral boundary extraction.
"""

from .dataset import CadastreDataset, create_dataloaders
from .transforms import get_train_transforms, get_val_transforms
from .models.unet_baseline import CadastreUNetBaseline, build_baseline_model
from .losses import CombinedFocalDiceLoss, FocalLoss, DiceLoss
from .metrics import SegmentationMetricsTracker, compute_all_metrics
from .trainer import CadastreTrainer
from .visualizer import visualize_predictions, plot_training_history

__all__ = [
    "CadastreDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    "CadastreUNetBaseline",
    "build_baseline_model",
    "CombinedFocalDiceLoss",
    "FocalLoss",
    "DiceLoss",
    "SegmentationMetricsTracker",
    "compute_all_metrics",
    "CadastreTrainer",
    "visualize_predictions",
    "plot_training_history",
]
