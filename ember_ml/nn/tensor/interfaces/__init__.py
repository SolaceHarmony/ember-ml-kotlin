"""
Tensor interfaces module.

This module defines the abstract interfaces for tensor operations and data types.
"""

from ember_ml.nn.tensor.interfaces.tensor import TensorInterface
from ember_ml.nn.tensor.interfaces.dtype import DTypeInterface

__all__ = [
    'TensorInterface',
    'DTypeInterface',
]