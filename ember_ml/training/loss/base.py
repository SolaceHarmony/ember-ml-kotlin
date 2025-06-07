"""
Base loss class for ember_ml.

This module provides a backend-agnostic implementation of the base loss class
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Dict, List, Optional, Union, Any, Callable

from ember_ml import ops

class Loss:
    """
    Base class for all loss functions.
    
    This class provides the basic interface for loss functions.
    All loss functions should inherit from this class.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the loss function.
        
        Args:
            reduction: Type of reduction to apply to the loss ('mean', 'sum', or 'none')
        """
        self.reduction = reduction
    
    def __call__(self, y_pred, y_true):
        """
        Compute the loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred, y_true):
        """
        Forward pass to compute the loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _reduce(self, loss):
        """
        Apply reduction to the loss.
        
        Args:
            loss: Unreduced loss
            
        Returns:
            Reduced loss
        """
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return ops.stats.mean(loss)
        elif self.reduction == 'sum':
            return stats.sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")