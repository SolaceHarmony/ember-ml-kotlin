"""
Cross Entropy loss for ember_ml.

This module provides a backend-agnostic implementation of the Cross Entropy loss
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Dict, List, Optional, Union, Any, Callable

from ember_ml import ops
from ember_ml.training.loss.base import Loss
from ember_ml.nn import tensor

class CrossEntropyLoss(Loss):
    """
    Cross Entropy loss.
    
    This loss computes the cross entropy between the predicted and true values.
    It is commonly used for classification tasks.
    """
    
    def __init__(self, reduction='mean', ignore_index=-100, label_smoothing=0.0):
        """
        Initialize the Cross Entropy loss.
        
        Args:
            reduction: Type of reduction to apply to the loss ('mean', 'sum', or 'none')
            ignore_index: Index to ignore in the target
            label_smoothing: Float in [0.0, 1.0] for label smoothing
        """
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, y_pred, y_true):
        """
        Compute the Cross Entropy loss.
        
        Args:
            y_pred: Predicted values (logits)
            y_true: True values (class indices or one-hot encoded)
            
        Returns:
            Cross Entropy loss
        """
        # Check if y_true is one-hot encoded
        if len(tensor.shape(y_true)) == len(tensor.shape(y_pred)):
            return self._forward_one_hot(y_pred, y_true)
        else:
            return self._forward_sparse(y_pred, y_true)
    
    def _forward_one_hot(self, y_pred, y_true):
        """
        Compute the Cross Entropy loss for one-hot encoded targets.
        
        Args:
            y_pred: Predicted values (logits)
            y_true: True values (one-hot encoded)
            
        Returns:
            Cross Entropy loss
        """
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            num_classes = tensor.shape(y_true)[-1]
            y_true = ops.add(
                ops.multiply(y_true, 1.0 - self.label_smoothing),
                ops.multiply(tensor.ones_like(y_true), self.label_smoothing / num_classes)
            )
        
        # Compute log softmax
        log_softmax = ops.log_softmax(y_pred, axis=-1)
        
        # Compute cross entropy
        loss = ops.multiply(ops.negative(y_true), log_softmax)
        loss = stats.sum(loss, axis=-1)
        
        # Apply reduction
        return self._reduce(loss)
    
    def _forward_sparse(self, y_pred, y_true):
        """
        Compute the Cross Entropy loss for sparse targets.
        
        Args:
            y_pred: Predicted values (logits)
            y_true: True values (class indices)
            
        Returns:
            Cross Entropy loss
        """
        # Create mask for ignored indices
        mask = ops.not_equal(y_true, self.ignore_index)
        y_true = ops.where(mask, y_true, tensor.zeros_like(y_true))
        
        # Compute log softmax
        log_softmax = ops.log_softmax(y_pred, axis=-1)
        
        # Gather log probabilities for the true classes
        batch_size = tensor.shape(y_pred)[0]
        num_classes = tensor.shape(y_pred)[-1]
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = ops.one_hot(y_true, num_classes)
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            y_true_one_hot = ops.add(
                ops.multiply(y_true_one_hot, 1.0 - self.label_smoothing),
                ops.multiply(tensor.ones_like(y_true_one_hot), self.label_smoothing / num_classes)
            )
        
        # Compute cross entropy
        loss = ops.multiply(ops.negative(y_true_one_hot), log_softmax)
        loss = stats.sum(loss, axis=-1)
        
        # Apply mask for ignored indices
        loss = ops.where(mask, loss, tensor.zeros_like(loss))
        
        # Apply reduction
        if self.reduction == 'mean':
            # Compute mean over non-ignored elements
            return ops.divide(stats.sum(loss), stats.sum(tensor.cast(mask, tensor.float32)))
        else:
            return self._reduce(loss)