"""
Backend Utilities for ember_ml

This module provides utility functions for working with ember_ml's backend system,
making it easier to convert between NumPy arrays and backend tensors, and to perform
common operations in a backend-agnostic way.
"""

import logging
from typing import Any, List, Optional, Tuple, Union, Dict
from ember_ml.nn import tensor
# Set up logging
logger = logging.getLogger('ember_ml.utils.backend')

# Backend module paths
_BACKENDS = {
    'numpy': 'ember_ml.backend.numpy',
    'torch': 'ember_ml.backend.torch',
    'torch_optimized': 'ember_ml.backend.torch_backend_optimized',
    'mlx': 'ember_ml.backend.mlx'
}

# Import ember_ml backend
# Use relative imports to ensure availability within the package
try:
    # Try absolute import first (for installed package)
    from ember_ml.backend import get_backend, set_backend
    from ember_ml import ops
except ImportError:
    # Fall back to relative import (for development)
    try:
        from ..backend import get_backend, set_backend
        from .. import ops
    except ImportError:
        # Critical failure - ember_ml backend is required
        logger.error("CRITICAL ERROR: ember_ml backend not available. This is a required dependency.")
        raise ImportError("ember_ml backend is required but not available. Please ensure the package is properly installed.")

def get_current_backend() -> str:
    """
    Get the current backend name.
    
    Returns:
        str: Name of the current backend ('mlx', 'torch', 'numpy')
    """
    return get_backend()

def set_preferred_backend(backend_name: Optional[str] = None) -> str:
    """
    Set the preferred backend if available.
    
    Args:
        backend_name: Name of the preferred backend ('mlx', 'torch', 'numpy')
        
    Returns:
        str: Name of the actually set backend
    """
    if backend_name is None:
        # Let ember_ml choose the best available backend
        return get_backend()
    
    try:
        set_backend(backend_name)
        logger.info(f"Set backend to {backend_name}")
        return get_backend()
    except ValueError:
        logger.warning(f"Backend {backend_name} not available. Using default.")
        return get_backend()

def initialize_random_seed(seed: int = 42) -> None:
    """
    Initialize random seed for reproducibility across all backends.
    
    Args:
        seed: Random seed value
    """
    # Import the set_seed function from the current backend
    backend_module = __import__(_BACKENDS[get_current_backend()], fromlist=['set_seed'])
    if hasattr(backend_module, 'set_seed'):
        backend_module.set_seed(seed)
        logger.info(f"Set random seed to {seed} for {get_backend()} backend")
    else:
        logger.warning(f"Backend {get_backend()} does not support setting random seed")


def sin_cos_transform(values: Any, period: float = 1.0) -> Tuple[Any, Any]:
    """
    Apply sine and cosine transformations for cyclical features.
    
    Args:
        values: Input values to transform
        period: Period for the transformation
        
    Returns:
        Tuple of (sin_values, cos_values) in the current backend format
    """
    values_tensor = values 
    
    sin_values = ops.sin(2 * ops.pi * values_tensor / period)
    cos_values = ops.cos(2 * ops.pi * values_tensor / period)
    
    return sin_values, cos_values

def vstack_safe(arrays: List[Any]) -> Any:
    """
    Safely stack arrays vertically using the current backend.
    
    Args:
        arrays: List of arrays to stack
        
    Returns:
        Stacked array in the current backend format
    """
    if not arrays:
        return None
    
    # Convert all arrays to the same format
    converted_arrays = [arr for arr in arrays]
    
    # Check if all arrays have the same shape except for the first dimension
    shapes = [tensor.shape(arr)[1:] for arr in converted_arrays]
    same_shape = all(shape == shapes[0] for shape in shapes)
    
    if same_shape:
        # Use tensor.concatenate for vertical stacking (equivalent to vstack)
        return tensor.concatenate(converted_arrays, axis=0)
    else:
        # If shapes are different, log a warning and return the first array
        logger.warning(f"Arrays have different shapes: {shapes}. Cannot stack.")
        return converted_arrays[0]

def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the current backend.
    
    Returns:
        Dict with backend information
    """
    info = {
        'name': get_current_backend()
    }
    
    # Add more backend-specific information
    try:
        info['device'] = ops.get_device()
    except:
        info['device'] = 'unknown'
    
    return info

def print_backend_info() -> None:
    """Print information about the current backend."""
    info = get_backend_info()
    
    print(f"Current backend: {info['name']}")
    print(f"Device: {info.get('device', 'unknown')}")
    
    # Test a simple operation
    a = tensor.ones((2, 2))
    b = tensor.ones((2, 2))
    c = ops.matmul(a, b)
    
    print(f"Test operation: {a} @ {b} = {c}")