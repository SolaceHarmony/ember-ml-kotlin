"""
NumPy basic bitwise operations for ember_ml.

This module provides NumPy implementations of basic bitwise operations
(AND, OR, XOR, NOT).
"""

import numpy as np
from typing import Any

# Import NumpyTensor dynamically within functions to avoid circular dependencies
# from ember_ml.backend.numpy.tensor import NumpyTensor
from ember_ml.backend.numpy.types import TensorLike

def bitwise_and(x: TensorLike, y: TensorLike) -> np.ndarray:
    """
    Compute the bitwise AND of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        NumPy array with the element-wise bitwise AND.
    """
    # Dynamically import NumpyTensor to avoid circular imports at module level
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return np.bitwise_and(x_arr, y_arr)

def bitwise_or(x: TensorLike, y: TensorLike) -> np.ndarray:
    """
    Compute the bitwise OR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        NumPy array with the element-wise bitwise OR.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return np.bitwise_or(x_arr, y_arr)

def bitwise_xor(x: TensorLike, y: TensorLike) -> np.ndarray:
    """
    Compute the bitwise XOR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        NumPy array with the element-wise bitwise XOR.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return np.bitwise_xor(x_arr, y_arr)

def bitwise_not(x: TensorLike) -> np.ndarray:
    """
    Compute the bitwise NOT (inversion) of x element-wise.

    Args:
        x: Input tensor or compatible type.

    Returns:
        NumPy array with the element-wise bitwise NOT.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    # NumPy uses invert for NOT operation
    return np.invert(x_arr)