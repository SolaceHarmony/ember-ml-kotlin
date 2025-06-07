"""
Training module for ember_ml.

This module provides backend-agnostic implementations of training components
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.training.optimizer import Optimizer, SGD, Adam
from ember_ml.training.loss import Loss, MSELoss, CrossEntropyLoss
from ember_ml.training.metrics import (
    classification_metrics, binary_classification_metrics, confusion_matrix,
    roc_auc, precision_recall_curve, average_precision_score,
    regression_metrics
)

__all__ = [
    # Module names
    'optimizer',
    'loss',
    'metrics',
    
    # Optimizer classes
    'Optimizer',
    'SGD',
    'Adam',
    
    # Loss classes
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
    
    # Metrics functions
    'classification_metrics',
    'binary_classification_metrics',
    'confusion_matrix',
    'roc_auc',
    'precision_recall_curve',
    'average_precision_score',
    'regression_metrics',
]