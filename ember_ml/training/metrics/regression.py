"""
Regression metrics for the ember_ml library.

This module provides metrics utilities for regression tasks.
"""

from typing import Dict
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ember_ml import ops

# Import sklearn metrics that we haven't implemented yet
from sklearn.metrics import r2_score

def regression_metrics(y_true: TensorLike, y_pred: TensorLike) -> Dict[str, TensorLike]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mse': ops.mse(y_true, y_pred),
        'rmse': ops.sqrt(ops.mse(y_true, y_pred)),
        'mae': ops.mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)  # Still using sklearn for r2_score
    }