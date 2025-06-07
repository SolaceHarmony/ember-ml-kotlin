"""MLX tensor casting operations."""

import mlx.core
import mlx.core as mx

from ember_ml.backend.mlx.types import DType, TensorLike


def cast(tensor: TensorLike, dtype: DType) -> mlx.core.array:
    """
    Cast a tensor to a new data type using MLX backend.

    Args:
        tensor: Input tensor-like object.
        dtype: Target data type (EmberDType, mlx.core.Dtype, str, or None).

    Returns:
        MLX array with the new data type.

    Raises:
        ValueError: If the target dtype is invalid.
    """
    # Import MLX specifics lazily
    from ember_ml.backend.mlx.tensor import MLXTensor

    # 1. Validate and get the MLX dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)

    # 2. Convert the input tensor to an MLX array
    tensor_obj = MLXTensor()
    tensor_array = tensor_obj.convert(tensor)

    # 3. If the validated dtype is None (meaning no cast needed), return original array
    if mlx_dtype is None:
        return tensor_array

    # 4. Special handling for bool type
    if hasattr(mlx_dtype, 'name') and mlx_dtype.name == 'bool_':
        # For bool type, we need to compare with zero
        if hasattr(mx, 'bool_'):
            # If MLX has bool_ type, use it
            return mx.array(tensor_array != 0, dtype=mx.bool_)
        else:
            # Otherwise use uint8 as fallback
            return mx.array(tensor_array != 0, dtype=mx.uint8)

    # 5. Perform the cast using the validated mlx_dtype
    try:
        return tensor_array.astype(mlx_dtype)
    except ValueError as e:
        # Handle float64 conversion if not supported
        if "float64 is not supported" in str(e).lower() and str(mlx_dtype) == 'float64':
            # Fall back to float32
            return tensor_array.astype(mx.float32)
        else:
            raise e
        