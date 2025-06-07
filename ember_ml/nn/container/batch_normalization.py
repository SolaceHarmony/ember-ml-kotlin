"""
Batch Normalization module for ember_ml.

This module provides a backend-agnostic implementation of a batch normalization layer
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Any, Tuple

from ember_ml import ops
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn import tensor
class BatchNormalization(Module):
    """
    Batch Normalization layer.
    
    Normalizes the activations of the previous layer at each batch.
    
    Attributes:
        epsilon: Small float added to variance to avoid dividing by zero
        momentum: Momentum for the moving average
        axis: Integer, the axis that should be normalized
        center: If True, add offset of beta to normalized tensor
        scale: If True, multiply by gamma
        beta_initializer: Initializer for the beta weight
        gamma_initializer: Initializer for the gamma weight
        moving_mean_initializer: Initializer for the moving mean
        moving_variance_initializer: Initializer for the moving variance
        beta: Offset tensor
        gamma: Scale tensor
        moving_mean: Moving mean tensor
        moving_variance: Moving variance tensor
        initialized: Whether the layer has been initialized
    """
    
    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
    ):
        """
        Initialize a batch normalization layer.
        
        Args:
            axis: Integer, the axis that should be normalized
            momentum: Momentum for the moving average
            epsilon: Small float added to variance to avoid dividing by zero
            center: If True, add offset of beta to normalized tensor
            scale: If True, multiply by gamma
        """
        super().__init__()
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        
        self.beta = None
        self.gamma = None
        self.moving_mean = None
        self.moving_variance = None
        self.initialized = False
    
    def forward(self, x: Any, training: bool = False) -> Any:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            training: Whether to use the moving statistics (False) or compute new statistics (True)
            
        Returns:
            Normalized output tensor
        """
        # Get input shape
        input_shape = tensor.shape(x)
        ndim = len(input_shape)
        
        # Convert axis to positive index
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        
        # Get the dimension to normalize
        dim = input_shape[axis]
        
        # Initialize parameters if not already done
        if not self.initialized:
            param_shape = [1] * ndim
            param_shape[axis] = dim
            
            if self.scale:
                self.gamma = Parameter(tensor.ones(param_shape))
            
            if self.center:
                self.beta = Parameter(tensor.zeros(param_shape))
            
            # Initialize moving statistics
            self.moving_mean = tensor.zeros(param_shape)
            self.moving_variance = tensor.ones(param_shape)
            
            self.initialized = True
        
        # Compute the axes to reduce over (all except the axis to normalize)
        reduction_axes = list(range(ndim))
        reduction_axes.pop(axis)
        
        # Compute mean and variance
        if training:
            # Compute statistics from mini-batch
            mean = ops.stats.mean(x, axis=reduction_axes, keepdims=True)
            variance = ops.stats.var(x, axis=reduction_axes, keepdims=True) # Use ops.stats.var

            # Update moving statistics
            self.moving_mean = ops.add(
                ops.multiply(self.moving_mean, self.momentum),
                ops.multiply(mean, 1.0 - self.momentum)
            )
            self.moving_variance = ops.add(
                ops.multiply(self.moving_variance, self.momentum),
                ops.multiply(variance, 1.0 - self.momentum)
            )
        else:
            # Use moving statistics
            mean = self.moving_mean
            variance = self.moving_variance
        
        # Normalize
        x_centered = ops.subtract(x, mean)
        x_normalized = ops.divide(x_centered, ops.sqrt(ops.add(variance, self.epsilon)))
        
        # Scale and shift
        if self.scale:
            x_normalized = ops.multiply(x_normalized, self.gamma)
        
        if self.center:
            x_normalized = ops.add(x_normalized, self.beta)
        
        return x_normalized
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"BatchNormalization(axis={self.axis}, momentum={self.momentum}, epsilon={self.epsilon})"