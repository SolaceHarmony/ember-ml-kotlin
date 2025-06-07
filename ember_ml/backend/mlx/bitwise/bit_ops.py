"""
MLX bit manipulation operations for ember_ml.

This module provides MLX implementations of bit manipulation operations
(count_ones, count_zeros, get_bit, set_bit, toggle_bit).
"""

from typing import Union
import mlx.core as mx
from ember_ml.backend.mlx.types import TensorLike

def count_ones(x: TensorLike) -> mx.array:
    """
    Count the number of set bits (1s) in each element of x (population count).

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        MLX array with the count of set bits for each element.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert(x)

    if not mx.issubdtype(x_arr.dtype, mx.integer):
        raise TypeError(f"count_ones requires an integer type, got {x_arr.dtype}")

    # Implementation using SWAR algorithm (SIMD Within A Register) for popcount
    # Adapted for MLX. Assumes up to 64-bit integers.
    dtype = x_arr.dtype
    if dtype == mx.uint8 or dtype == mx.int8:
        c1 = mx.array(0x55, dtype=dtype)
        c2 = mx.array(0x33, dtype=dtype)
        c3 = mx.array(0x0F, dtype=dtype)
        x_arr = mx.subtract(x_arr, mx.bitwise_and(mx.right_shift(x_arr, 1), c1))
        x_arr = mx.add(mx.bitwise_and(x_arr, c2), mx.bitwise_and(mx.right_shift(x_arr, 2), c2))
        x_arr = mx.bitwise_and(mx.add(x_arr, mx.right_shift(x_arr, 4)), c3)
        return x_arr.astype(mx.int32) # Return count as int32
    elif dtype == mx.uint16 or dtype == mx.int16:
        c1 = mx.array(0x5555, dtype=dtype)
        c2 = mx.array(0x3333, dtype=dtype)
        c3 = mx.array(0x0F0F, dtype=dtype)
        x_arr = mx.subtract(x_arr, mx.bitwise_and(mx.right_shift(x_arr, 1), c1))
        x_arr = mx.add(mx.bitwise_and(x_arr, c2), mx.bitwise_and(mx.right_shift(x_arr, 2), c2))
        x_arr = mx.bitwise_and(mx.add(x_arr, mx.right_shift(x_arr, 4)), c3)
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 8))
        return mx.bitwise_and(x_arr, 0x001F).astype(mx.int32)
    elif dtype == mx.uint32 or dtype == mx.int32:
        c1 = mx.array(0x55555555, dtype=dtype)
        c2 = mx.array(0x33333333, dtype=dtype)
        c3 = mx.array(0x0F0F0F0F, dtype=dtype)
        x_arr = mx.subtract(x_arr, mx.bitwise_and(mx.right_shift(x_arr, 1), c1))
        x_arr = mx.add(mx.bitwise_and(x_arr, c2), mx.bitwise_and(mx.right_shift(x_arr, 2), c2))
        x_arr = mx.bitwise_and(mx.add(x_arr, mx.right_shift(x_arr, 4)), c3)
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 8))
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 16))
        return mx.bitwise_and(x_arr, 0x0000003F).astype(mx.int32)
    elif dtype == mx.uint64 or dtype == mx.int64:
        c1 = mx.array(0x5555555555555555, dtype=dtype)
        c2 = mx.array(0x3333333333333333, dtype=dtype)
        c3 = mx.array(0x0F0F0F0F0F0F0F0F, dtype=dtype)
        c4 = mx.array(0x0101010101010101, dtype=dtype) # For final multiplication step
        x_arr = mx.subtract(x_arr, mx.bitwise_and(mx.right_shift(x_arr, 1), c1))
        x_arr = mx.add(mx.bitwise_and(x_arr, c2), mx.bitwise_and(mx.right_shift(x_arr, 2), c2))
        x_arr = mx.bitwise_and(mx.add(x_arr, mx.right_shift(x_arr, 4)), c3)
        # The standard SWAR uses multiplication here, which might be slow or complex
        # Alternative: continue with shifts and adds
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 8))
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 16))
        x_arr = mx.add(x_arr, mx.right_shift(x_arr, 32))
        return mx.bitwise_and(x_arr, 0x000000000000007F).astype(mx.int32)
    else:
        raise TypeError(f"Unsupported integer type for count_ones: {dtype}")


def count_zeros(x: TensorLike) -> mx.array:
    """
    Count the number of unset bits (0s) in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        MLX array with the count of unset bits for each element.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert(x)

    if not mx.issubdtype(x_arr.dtype, mx.integer):
        raise TypeError(f"count_zeros requires an integer type, got {x_arr.dtype}")

    # Determine bit width based on dtype
    if x_arr.dtype == mx.uint8 or x_arr.dtype == mx.int8:
        bit_width = 8
    elif x_arr.dtype == mx.uint16 or x_arr.dtype == mx.int16:
        bit_width = 16
    elif x_arr.dtype == mx.uint32 or x_arr.dtype == mx.int32:
        bit_width = 32
    elif x_arr.dtype == mx.uint64 or x_arr.dtype == mx.int64:
        bit_width = 64
    else:
        raise TypeError(f"Unsupported integer type for count_zeros: {x_arr.dtype}")

    ones = count_ones(x_arr)
    # Ensure bit_width is broadcastable
    bit_width_arr = mx.array(bit_width, dtype=ones.dtype)
    return mx.subtract(bit_width_arr, ones)


def get_bit(x: TensorLike, position: TensorLike) -> mx.array:
    """
    Get the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        MLX array with the bit value (0 or 1) at the specified position(s).
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert(x)
    pos_arr = tensor_ops.convert(position)

    if not mx.issubdtype(x_arr.dtype, mx.integer):
        raise TypeError(f"get_bit requires an integer type for x, got {x_arr.dtype}")
    if not mx.issubdtype(pos_arr.dtype, mx.integer):
         pos_arr = pos_arr.astype(mx.int32) # Cast position to int

    # Create mask: 1 << position
    one = mx.array(1).astype(x_arr.dtype) # Mask should have same type as x
    mask = mx.left_shift(one, pos_arr)

    # Extract bit: (x & mask) >> position
    extracted_bit = mx.right_shift(mx.bitwise_and(x_arr, mask), pos_arr)
    # Return as int32 for consistency of bit value representation
    return extracted_bit.astype(mx.int32)


def set_bit(x: TensorLike, position: TensorLike, value: TensorLike) -> mx.array:
    """
    Set the bit at the specified position in each element of x to value (0 or 1).

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.
        value: Bit value(s) (0 or 1). Integer or tensor.

    Returns:
        MLX array with the bit at the specified position(s) set.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert(x)
    pos_arr = tensor_ops.convert(position)
    val_arr = tensor_ops.convert(value)

    if not mx.issubdtype(x_arr.dtype, mx.integer):
        raise TypeError(f"set_bit requires an integer type for x, got {x_arr.dtype}")
    if not mx.issubdtype(pos_arr.dtype, mx.integer):
         pos_arr = pos_arr.astype(mx.int32)
    if not mx.issubdtype(val_arr.dtype, mx.integer):
         val_arr = val_arr.astype(mx.int32) # Cast value to int

    # Create mask: 1 << position
    one = mx.array(1).astype(x_arr.dtype)
    mask = mx.left_shift(one, pos_arr)

    # Clear the bit: x & (~mask)
    cleared_x = mx.bitwise_and(x_arr, mx.bitwise_invert(mask)) # Use bitwise_invert

    # Prepare value to be set: (value & 1) << position
    # Ensure value is 0 or 1, then shift
    value_bit = mx.bitwise_and(val_arr.astype(x_arr.dtype), one)
    value_shifted = mx.left_shift(value_bit, pos_arr)

    # Set the bit using OR: cleared_x | value_shifted
    return mx.bitwise_or(cleared_x, value_shifted)


def toggle_bit(x: TensorLike, position: TensorLike) -> mx.array:
    """
    Toggle the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        MLX array with the bit at the specified position(s) toggled.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()
    x_arr = tensor_ops.convert(x)
    pos_arr = tensor_ops.convert(position)

    if not mx.issubdtype(x_arr.dtype, mx.integer):
        raise TypeError(f"toggle_bit requires an integer type for x, got {x_arr.dtype}")
    if not mx.issubdtype(pos_arr.dtype, mx.integer):
         pos_arr = pos_arr.astype(mx.int32)

    # Create mask: 1 << position
    one = mx.array(1).astype(x_arr.dtype)
    mask = mx.left_shift(one, pos_arr)

    # Toggle the bit using XOR: x ^ mask
    return mx.bitwise_xor(x_arr, mask)