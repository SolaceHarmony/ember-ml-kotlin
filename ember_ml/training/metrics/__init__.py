"""
Metrics module for the ember_ml library.

This module provides metrics utilities for training and evaluating models.
"""

from ember_ml.training.metrics.classification import (
    classification_metrics,
    binary_classification_metrics,
    confusion_matrix,
    roc_auc,
    precision_recall_curve,
    average_precision_score
)

from ember_ml.training.metrics.regression import (
    regression_metrics
)

__all__ = [
    # Classification metrics
    'classification_metrics',
    'binary_classification_metrics',
    'confusion_matrix',
    'roc_auc',
    'precision_recall_curve',
    'average_precision_score',
    
    # Regression metrics
    'regression_metrics',
]