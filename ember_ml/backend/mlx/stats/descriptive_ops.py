"""
MLX descriptive statistical operations for ember_ml.
"""

import mlx.core as mx
from typing import Optional, List, Tuple

# Assuming TensorLike and ShapeLike are defined appropriately elsewhere,
# potentially in a types module or directly using Union types.
# For now, using Any for simplicity until types module is confirmed.
from typing import Any
TensorLike = Any
ShapeLike = Any

# It's crucial that the tensor conversion logic is handled correctly.
# Assuming MLXTensor provides this. Need to ensure imports are valid.
# Using a local import might be safer if circular dependencies are an issue.

def sum(x: TensorLike, axis: Optional[ShapeLike] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the sum of an MLX array along specified axes.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Sum of the array
    """
    # Local import to avoid potential circular dependencies
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_array = tensor_ops.convert(x)

    if axis is None:
        return mx.sum(x_array, keepdims=keepdims)
    if isinstance(axis, (list, tuple)):
        # MLX doesn't support multiple axes directly, so we need to do it sequentially
        result = x_array
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = mx.sum(result, axis=ax, keepdims=keepdims)
        return result
    else:
        return mx.sum(x_array, axis=axis, keepdims=keepdims)

# --- Add other MLX descriptive stats operations here (mean, var, etc.) ---