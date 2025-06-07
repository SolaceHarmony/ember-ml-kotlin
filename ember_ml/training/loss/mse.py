"""
Mean Squared Error (MSE) loss for ember_ml.

This module provides a backend-agnostic implementation of the MSE loss
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Dict, List, Optional, Union, Any, Callable

from ember_ml import ops
from ember_ml.training.loss.base import Loss

class MSELoss(Loss):
    """
    Mean Squared Error (MSE) loss.
    
    This loss computes the mean squared error between the predicted and true values.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the MSE loss.
        
        Args:
            reduction: Type of reduction to apply to the loss ('mean', 'sum', or 'none')
        """
        super().__init__(reduction=reduction)
    
    def forward(self, y_pred, y_true):
        """
        Compute the MSE loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            MSE loss
        """
        # Compute squared error
        squared_error = ops.square(ops.subtract(y_pred, y_true))
        
        # Apply reduction
        return self._reduce(squared_error)