"""
MLX basic bitwise operations for ember_ml.

This module provides MLX implementations of basic bitwise operations
(AND, OR, XOR, NOT).
"""

import mlx.core as mx
from typing import Any

# Import MLXTensor dynamically within functions to avoid circular dependencies
# from ember_ml.backend.mlx.tensor import MLXTensor
from ember_ml.backend.mlx.types import TensorLike

def bitwise_and(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise AND of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        MLX array with the element-wise bitwise AND.
    """
    # Dynamically import MLXTensor to avoid circular imports at module level
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return mx.bitwise_and(x_arr, y_arr)

def bitwise_or(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise OR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        MLX array with the element-wise bitwise OR.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return mx.bitwise_or(x_arr, y_arr)

def bitwise_xor(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the bitwise XOR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        MLX array with the element-wise bitwise XOR.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    y_arr = tensor_ops.convert_to_tensor(y)
    return mx.bitwise_xor(x_arr, y_arr)

def bitwise_not(x: TensorLike) -> mx.array:
    """
    Compute the bitwise NOT (inversion) of x element-wise.

    Args:
        x: Input tensor or compatible type.

    Returns:
        MLX array with the element-wise bitwise NOT.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    # Correction: MLX uses bitwise_invert for NOT operation
    return mx.bitwise_invert(x_arr)