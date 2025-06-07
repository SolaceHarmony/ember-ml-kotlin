"""
Backend-agnostic implementation of standardization.

This module provides a standardization implementation using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Any

from ember_ml import ops
from ember_ml.nn import tensor

class Standardize:
    """Standardize features by removing the mean and scaling to unit variance.
    
    This implementation is backend-agnostic and works with all backends (NumPy, PyTorch, MLX).
    It implements the StandardizeInterface from ember_ml.features.interfaces.
    """
    
    def __init__(self):
        """Initialize Standardize."""
        self.mean_ = None
        self.scale_ = None
        self.with_mean_ = True
        self.with_std_ = True
        self.axis_ = 0
    
    def fit(
        self,
        X: Any,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = 0,
    ) -> "Standardize":
        """
        Compute the mean and std to be used for standardization.
        
        Args:
            X: Input data
            with_mean: Whether to center the data
            with_std: Whether to scale the data
            axis: Axis along which to standardize
            
        Returns:
            Self
        """
        X_tensor = tensor.convert_to_tensor(X)
        self.with_mean_ = with_mean
        self.with_std_ = with_std
        self.axis_ = axis
        
        # Compute mean
        if with_mean:
            self.mean_ = ops.stats.mean(X_tensor, axis=axis, keepdims=True)
        else:
            self.mean_ = None
        
        # Compute standard deviation
        if with_std:
            # Calculate variance directly using ops.stats.var(ddof=0)
            # This might be more numerically stable than manual calculation.
            var = ops.stats.var(X_tensor, axis=axis, keepdims=True, ddof=0)

            # Avoid division by zero
            eps = 1e-6  # Keep increased epsilon
            # Use positional arguments min_val=eps, max_val=None for ops.clip
            self.scale_ = ops.sqrt(ops.clip(var, eps, None))
        else:
            self.scale_ = None
        
        return self
    
    def transform(self, X: Any) -> Any:
        """
        Standardize data.
        
        Args:
            X: Input data
            
        Returns:
            X_scaled: Standardized data
        """
        if self.mean_ is None and self.scale_ is None:
            raise ValueError("Standardize not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        
        # Center data
        if self.with_mean_ and self.mean_ is not None:
            X_centered = ops.subtract(X_tensor, self.mean_)
        else:
            X_centered = X_tensor
        
        # Scale data
        if self.with_std_ and self.scale_ is not None:
            X_scaled = ops.divide(X_centered, self.scale_)
        else:
            X_scaled = X_centered
        
        return X_scaled
    
    def fit_transform(
        self,
        X: Any,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = 0,
    ) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input data
            with_mean: Whether to center the data
            with_std: Whether to scale the data
            axis: Axis along which to standardize
            
        Returns:
            X_scaled: Standardized data
        """
        self.fit(
            X,
            with_mean=with_mean,
            with_std=with_std,
            axis=axis,
        )
        return self.transform(X)
    
    def inverse_transform(self, X: Any) -> Any:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Input data
            
        Returns:
            X_original: Original data
        """
        if self.mean_ is None and self.scale_ is None:
            raise ValueError("Standardize not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        
        # Unscale data
        if self.with_std_ and self.scale_ is not None:
            # Use ops.multiply
            X_unscaled = ops.multiply(X_tensor, self.scale_)
        else:
            X_unscaled = X_tensor

        # Uncenter data
        if self.with_mean_ and self.mean_ is not None:
            # Use ops.add
            X_original = ops.add(X_unscaled, self.mean_)
        else:
            X_original = X_unscaled
        
        return X_original