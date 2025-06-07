"""
NumPy bit manipulation operations for ember_ml.

This module provides NumPy implementations of bit manipulation operations
(count_ones, count_zeros, get_bit, set_bit, toggle_bit).
"""

import numpy as np
from typing import Union

# Import NumpyTensor dynamically within functions
# from ember_ml.backend.numpy.tensor import NumpyTensor
from ember_ml.backend.numpy.types import TensorLike

def count_ones(x: TensorLike) -> np.ndarray:
    """
    Count the number of set bits (1s) in each element of x (population count).

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        NumPy array with the count of set bits for each element.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)

    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError(f"count_ones requires an integer type, got {x_arr.dtype}")

    # Implementation using SWAR algorithm (SIMD Within A Register) for popcount
    # Adapted for NumPy. Assumes up to 64-bit integers.
    dtype = x_arr.dtype
    if dtype == np.uint8 or dtype == np.int8:
        c1 = np.array(0x55, dtype=dtype)
        c2 = np.array(0x33, dtype=dtype)
        c3 = np.array(0x0F, dtype=dtype)
        x_arr = np.subtract(x_arr, np.bitwise_and(np.right_shift(x_arr, 1), c1))
        x_arr = np.add(np.bitwise_and(x_arr, c2), np.bitwise_and(np.right_shift(x_arr, 2), c2))
        x_arr = np.bitwise_and(np.add(x_arr, np.right_shift(x_arr, 4)), c3)
        return x_arr.astype(np.int32) # Return count as int32
    elif dtype == np.uint16 or dtype == np.int16:
        c1 = np.array(0x5555, dtype=dtype)
        c2 = np.array(0x3333, dtype=dtype)
        c3 = np.array(0x0F0F, dtype=dtype)
        x_arr = np.subtract(x_arr, np.bitwise_and(np.right_shift(x_arr, 1), c1))
        x_arr = np.add(np.bitwise_and(x_arr, c2), np.bitwise_and(np.right_shift(x_arr, 2), c2))
        x_arr = np.bitwise_and(np.add(x_arr, np.right_shift(x_arr, 4)), c3)
        x_arr = np.add(x_arr, np.right_shift(x_arr, 8))
        return np.bitwise_and(x_arr, 0x001F).astype(np.int32)
    elif dtype == np.uint32 or dtype == np.int32:
        c1 = np.array(0x55555555, dtype=dtype)
        c2 = np.array(0x33333333, dtype=dtype)
        c3 = np.array(0x0F0F0F0F, dtype=dtype)
        x_arr = np.subtract(x_arr, np.bitwise_and(np.right_shift(x_arr, 1), c1))
        x_arr = np.add(np.bitwise_and(x_arr, c2), np.bitwise_and(np.right_shift(x_arr, 2), c2))
        x_arr = np.bitwise_and(np.add(x_arr, np.right_shift(x_arr, 4)), c3)
        x_arr = np.add(x_arr, np.right_shift(x_arr, 8))
        x_arr = np.add(x_arr, np.right_shift(x_arr, 16))
        return np.bitwise_and(x_arr, 0x0000003F).astype(np.int32)
    elif dtype == np.uint64 or dtype == np.int64:
        c1 = np.array(0x5555555555555555, dtype=dtype)
        c2 = np.array(0x3333333333333333, dtype=dtype)
        c3 = np.array(0x0F0F0F0F0F0F0F0F, dtype=dtype)
        c4 = np.array(0x0101010101010101, dtype=dtype) # For final multiplication step
        x_arr = np.subtract(x_arr, np.bitwise_and(np.right_shift(x_arr, 1), c1))
        x_arr = np.add(np.bitwise_and(x_arr, c2), np.bitwise_and(np.right_shift(x_arr, 2), c2))
        x_arr = np.bitwise_and(np.add(x_arr, np.right_shift(x_arr, 4)), c3)
        # The standard SWAR uses multiplication here, which might be slow or complex
        # Alternative: continue with shifts and adds
        x_arr = np.add(x_arr, np.right_shift(x_arr, 8))
        x_arr = np.add(x_arr, np.right_shift(x_arr, 16))
        x_arr = np.add(x_arr, np.right_shift(x_arr, 32))
        return np.bitwise_and(x_arr, 0x000000000000007F).astype(np.int32)
    else:
        raise TypeError(f"Unsupported integer type for count_ones: {dtype}")


def count_zeros(x: TensorLike) -> np.ndarray:
    """
    Count the number of unset bits (0s) in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        NumPy array with the count of unset bits for each element.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)

    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError(f"count_zeros requires an integer type, got {x_arr.dtype}")

    # Determine bit width based on dtype
    if x_arr.dtype == np.uint8 or x_arr.dtype == np.int8:
        bit_width = 8
    elif x_arr.dtype == np.uint16 or x_arr.dtype == np.int16:
        bit_width = 16
    elif x_arr.dtype == np.uint32 or x_arr.dtype == np.int32:
        bit_width = 32
    elif x_arr.dtype == np.uint64 or x_arr.dtype == np.int64:
        bit_width = 64
    else:
        raise TypeError(f"Unsupported integer type for count_zeros: {x_arr.dtype}")

    ones = count_ones(x_arr)
    # Ensure bit_width is broadcastable
    bit_width_arr = np.array(bit_width, dtype=ones.dtype)
    return np.subtract(bit_width_arr, ones)


def get_bit(x: TensorLike, position: TensorLike) -> np.ndarray:
    """
    Get the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        NumPy array with the bit value (0 or 1) at the specified position(s).
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    pos_arr = tensor_ops.convert_to_tensor(position)

    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError(f"get_bit requires an integer type for x, got {x_arr.dtype}")
    if not np.issubdtype(pos_arr.dtype, np.integer):
         pos_arr = pos_arr.astype(np.int32) # Cast position to int

    # Create mask: 1 << position
    one = np.array(1).astype(x_arr.dtype) # Mask should have same type as x
    mask = np.left_shift(one, pos_arr)

    # Extract bit: (x & mask) >> position
    extracted_bit = np.right_shift(np.bitwise_and(x_arr, mask), pos_arr)
    # Return as int32 for consistency of bit value representation
    return extracted_bit.astype(np.int32)


def set_bit(x: TensorLike, position: TensorLike, value: TensorLike) -> np.ndarray:
    """
    Set the bit at the specified position in each element of x to value (0 or 1).

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.
        value: Bit value(s) (0 or 1). Integer or tensor.

    Returns:
        NumPy array with the bit at the specified position(s) set.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    pos_arr = tensor_ops.convert_to_tensor(position)
    val_arr = tensor_ops.convert_to_tensor(value)

    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError(f"set_bit requires an integer type for x, got {x_arr.dtype}")
    if not np.issubdtype(pos_arr.dtype, np.integer):
         pos_arr = pos_arr.astype(np.int32)
    if not np.issubdtype(val_arr.dtype, np.integer):
         val_arr = val_arr.astype(np.int32) # Cast value to int

    # Create mask: 1 << position
    one = np.array(1).astype(x_arr.dtype)
    mask = np.left_shift(one, pos_arr)

    # Clear the bit: x & (~mask)
    cleared_x = np.bitwise_and(x_arr, np.invert(mask)) # Use invert for NOT operation

    # Prepare value to be set: (value & 1) << position
    # Ensure value is 0 or 1, then shift
    value_bit = np.bitwise_and(val_arr.astype(x_arr.dtype), one)
    value_shifted = np.left_shift(value_bit, pos_arr)

    # Set the bit using OR: cleared_x | value_shifted
    return np.bitwise_or(cleared_x, value_shifted)


def toggle_bit(x: TensorLike, position: TensorLike) -> np.ndarray:
    """
    Toggle the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        NumPy array with the bit at the specified position(s) toggled.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    pos_arr = tensor_ops.convert_to_tensor(position)

    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError(f"toggle_bit requires an integer type for x, got {x_arr.dtype}")
    if not np.issubdtype(pos_arr.dtype, np.integer):
         pos_arr = pos_arr.astype(np.int32)

    # Create mask: 1 << position
    one = np.array(1).astype(x_arr.dtype)
    mask = np.left_shift(one, pos_arr)

    # Toggle the bit using XOR: x ^ mask
    return np.bitwise_xor(x_arr, mask)