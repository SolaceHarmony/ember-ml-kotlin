"""
Backend-agnostic implementation of normalization.

This module provides a normalization implementation using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any, Tuple
from ember_ml.ops import stats
from ember_ml import ops
from ember_ml.nn import tensor

class Normalize:
    """Scale input vectors individually to unit norm.
    
    This implementation is backend-agnostic and works with all backends (NumPy, PyTorch, MLX).
    It implements the NormalizeInterface from ember_ml.features.interfaces.
    """
    
    def __init__(self):
        """Initialize Normalize."""
        self.norm_ = "l2"
        self.axis_ = 1
        self.norms_ = None
    
    def fit(
        self,
        X: Any,
        *,
        norm: str = "l2",
        axis: int = 1,
    ) -> "Normalize":
        """
        Compute the norm to be used for normalization.
        
        Args:
            X: Input data
            norm: The norm to use:
                - 'l1': Sum of absolute values
                - 'l2': Square root of sum of squares
                - 'max': Maximum absolute value
            axis: Axis along which to normalize
            
        Returns:
            Self
        """
        X_tensor = tensor.convert_to_tensor(X)
        self.norm_ = norm
        self.axis_ = axis
        
        # Compute norms
        if norm == "l1":
            self.norms_ = stats.sum(ops.abs(X_tensor), axis=axis, keepdims=True)
        elif norm == "l2":
            self.norms_ = ops.sqrt(stats.sum(ops.square(X_tensor), axis=axis, keepdims=True))
        elif norm == "max":
            self.norms_ = stats.max(ops.abs(X_tensor), axis=axis, keepdims=True)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        
        return self
    
    def transform(self, X: Any) -> Any:
        """
        Normalize data.
        
        Args:
            X: Input data
            
        Returns:
            X_normalized: Normalized data
        """
        if self.norms_ is None:
            raise ValueError("Normalize not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        
        # Avoid division by zero
        eps = ops.finfo(X_tensor.dtype).eps
        norms_clipped = tensor.maximum(self.norms_, eps)
        
        # Normalize
        X_normalized = X_tensor / norms_clipped
        
        return X_normalized
    
    def fit_transform(
        self,
        X: Any,
        *,
        norm: str = "l2",
        axis: int = 1,
    ) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input data
            norm: The norm to use:
                - 'l1': Sum of absolute values
                - 'l2': Square root of sum of squares
                - 'max': Maximum absolute value
            axis: Axis along which to normalize
            
        Returns:
            X_normalized: Normalized data
        """
        self.fit(
            X,
            norm=norm,
            axis=axis,
        )
        return self.transform(X)