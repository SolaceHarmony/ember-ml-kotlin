"""
Glorot initializers for neural network weights.

This module provides Glorot (Xavier) initializers for neural network weights,
which help maintain the variance of activations and gradients across layers.
"""

from typing import Tuple, Optional, Union, Any

from ember_ml import ops
from ember_ml.nn import tensor

def glorot_uniform(shape: Tuple[int, ...], dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Glorot uniform initializer, also called Xavier uniform initializer.
    
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number
    of input units in the weight tensor and `fan_out` is the number of output units.
    
    Args:
        shape: Shape of the tensor to initialize
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    
    limit = ops.sqrt(ops.divide(6.0, ops.add(fan_in, fan_out)))
    
    return tensor.random_uniform(shape, -limit, limit, dtype=dtype, device=device)
def glorot_normal(shape: Tuple[int, ...], dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Glorot normal initializer, also called Xavier normal initializer.
    
    It draws samples from a normal distribution with mean 0 and
    standard deviation `sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
    of input units in the weight tensor and `fan_out` is the number of output units.
    
    Args:
        shape: Shape of the tensor to initialize
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    
    stddev = ops.sqrt(ops.divide(2.0, ops.add(fan_in, fan_out)))
    
    return tensor.random_normal(shape, 0.0, stddev, dtype=dtype, device=device)

def orthogonal(shape: Tuple[int, ...], gain: float = 1.0, dtype: Optional[Any] = None, device: Optional[str] = None):
    """
    Orthogonal initializer.
    
    It generates a random orthogonal matrix using QR decomposition.
    
    This function uses the backend-specific implementation from ops.linearalg.orthogonal,
    which provides optimized implementations for different backends:
    - MLX: Uses HPC implementation with double-single precision for numerical stability
    - PyTorch: Uses native torch.linalg.qr
    - NumPy: Uses native numpy.linalg.qr
    
    Args:
        shape: Shape of the tensor to initialize
        gain: Multiplicative factor to apply to the orthogonal matrix
        dtype: Data type of the tensor
        device: Device to place the tensor on
        
    Returns:
        Initialized tensor
    """
    from ember_ml.ops import linearalg
    
    # Use the backend-specific implementation from ops.linearalg
    return linearalg.orthogonal(shape, gain, dtype, device)