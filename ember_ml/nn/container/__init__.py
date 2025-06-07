"""
Container modules for ember_ml.

This module provides backend-agnostic implementations of container modules
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.nn.container.linear import Linear
from ember_ml.nn.container.dropout import Dropout
from ember_ml.nn.container.sequential import Sequential
from ember_ml.nn.container.batch_normalization import BatchNormalization

# Export all functions and classes
__all__ = [
    
    # Operations
    'Linear',
    'Dropout',
    'Sequential',
    'BatchNormalization',
    # Removed 'Dense' export
]