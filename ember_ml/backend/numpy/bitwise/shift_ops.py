"""
NumPy shift bitwise operations for ember_ml.

This module provides NumPy implementations of bitwise shift operations
(left_shift, right_shift, rotate_left, rotate_right).
"""

import numpy as np
from typing import Union

# Import NumpyTensor dynamically within functions
# from ember_ml.backend.numpy.tensor import NumpyTensor
from ember_ml.backend.numpy.types import TensorLike

def left_shift(x: TensorLike, shifts: TensorLike) -> np.ndarray:
    """
    Shift the bits of x to the left by shifts positions.

    Args:
        x: Input tensor or compatible type.
        shifts: Number of bits to shift (integer or tensor).

    Returns:
        NumPy array with x shifted left by shifts bits.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    shifts_arr = tensor_ops.convert_to_tensor(shifts)
    # Ensure shifts is integer type
    if not np.issubdtype(shifts_arr.dtype, np.integer):
         shifts_arr = shifts_arr.astype(np.int32)
    return np.left_shift(x_arr, shifts_arr)

def right_shift(x: TensorLike, shifts: TensorLike) -> np.ndarray:
    """
    Shift the bits of x to the right by shifts positions.

    Args:
        x: Input tensor or compatible type.
        shifts: Number of bits to shift (integer or tensor).

    Returns:
        NumPy array with x shifted right by shifts bits.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    shifts_arr = tensor_ops.convert_to_tensor(shifts)
    # Ensure shifts is integer type
    if not np.issubdtype(shifts_arr.dtype, np.integer):
         shifts_arr = shifts_arr.astype(np.int32)
    return np.right_shift(x_arr, shifts_arr)

def rotate_left(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> np.ndarray:
    """
    Rotate the bits of x to the left by shifts positions.

    Args:
        x: Input tensor or compatible type (must be unsigned integer type).
        shifts: Number of bits to rotate (integer or tensor).
        bit_width: The bit width of the integer type (e.g., 8, 16, 32, 64).

    Returns:
        NumPy array with x rotated left by shifts bits.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    shifts_arr = tensor_ops.convert_to_tensor(shifts)

    # Ensure unsigned integer type for rotation logic
    if not np.issubdtype(x_arr.dtype, np.unsignedinteger):
        # Attempt to cast, but warn or raise if inappropriate?
        # For now, assume user provides appropriate unsigned type or cast happens before.
        # Example cast (might need adjustment based on desired behavior):
        if np.issubdtype(x_arr.dtype, np.signedinteger):
             print(f"Warning: rotate_left received signed integer type {x_arr.dtype}. Casting to unsigned.")
             # Choose appropriate unsigned type based on bit_width or original type
             if x_arr.dtype == np.int8: x_arr = x_arr.astype(np.uint8)
             elif x_arr.dtype == np.int16: x_arr = x_arr.astype(np.uint16)
             elif x_arr.dtype == np.int32: x_arr = x_arr.astype(np.uint32)
             elif x_arr.dtype == np.int64: x_arr = x_arr.astype(np.uint64)
             else: raise TypeError(f"Unsupported signed integer type for rotate_left: {x_arr.dtype}")
        else:
             raise TypeError(f"rotate_left requires an unsigned integer type, got {x_arr.dtype}")


    # Ensure shifts is integer type
    if not np.issubdtype(shifts_arr.dtype, np.integer):
         shifts_arr = shifts_arr.astype(np.int32)

    # Normalize shifts to be within [0, bit_width)
    # Ensure bit_width array has compatible dtype with shifts_arr for remainder
    bit_width_np = np.array(bit_width, dtype=shifts_arr.dtype) # Use different name
    shifts_arr = np.remainder(shifts_arr, bit_width_np)

    # Perform rotation using shifts and bitwise OR
    # Ensure bit_width array has compatible dtype for subtraction
    bit_width_np = np.array(bit_width, dtype=shifts_arr.dtype)
    # Cast shift amounts explicitly to the target unsigned type for safety
    # Ensure right_shift_amount is calculated using integer arithmetic compatible with shifts_arr
    right_shift_amount = np.subtract(bit_width_np, shifts_arr)
    # Cast the final shift amounts to the same type as x_arr before shifting
    right_shift_amount_casted = right_shift_amount.astype(x_arr.dtype)
    shifts_arr_casted = shifts_arr.astype(x_arr.dtype)

    left_part = np.left_shift(x_arr, shifts_arr_casted)
    right_part = np.right_shift(x_arr, right_shift_amount_casted)

    # Ensure final result is the same dtype as input
    return np.bitwise_or(left_part, right_part).astype(x_arr.dtype)


def rotate_right(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> np.ndarray:
    """
    Rotate the bits of x to the right by shifts positions.

    Args:
        x: Input tensor or compatible type (must be unsigned integer type).
        shifts: Number of bits to rotate (integer or tensor).
        bit_width: The bit width of the integer type (e.g., 8, 16, 32, 64).

    Returns:
        NumPy array with x rotated right by shifts bits.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()
    x_arr = tensor_ops.convert_to_tensor(x)
    shifts_arr = tensor_ops.convert_to_tensor(shifts)

    # Ensure unsigned integer type for rotation logic
    if not np.issubdtype(x_arr.dtype, np.unsignedinteger):
        if np.issubdtype(x_arr.dtype, np.signedinteger):
             print(f"Warning: rotate_right received signed integer type {x_arr.dtype}. Casting to unsigned.")
             if x_arr.dtype == np.int8: x_arr = x_arr.astype(np.uint8) # Correct astype
             elif x_arr.dtype == np.int16: x_arr = x_arr.astype(np.uint16) # Correct astype
             elif x_arr.dtype == np.int32: x_arr = x_arr.astype(np.uint32) # Correct astype
             elif x_arr.dtype == np.int64: x_arr = x_arr.astype(np.uint64) # Correct astype
             else: raise TypeError(f"Unsupported signed integer type for rotate_right: {x_arr.dtype}")
        else:
             raise TypeError(f"rotate_right requires an unsigned integer type, got {x_arr.dtype}")

    # Ensure shifts is integer type
    if not np.issubdtype(shifts_arr.dtype, np.integer):
         shifts_arr = shifts_arr.astype(np.int32)

    # Normalize shifts to be within [0, bit_width)
    # Ensure bit_width array has compatible dtype with shifts_arr for remainder
    bit_width_np = np.array(bit_width, dtype=shifts_arr.dtype) # Use different name
    shifts_arr = np.remainder(shifts_arr, bit_width_np)

    # Perform rotation using shifts and bitwise OR
    # Ensure bit_width array has compatible dtype for subtraction
    bit_width_np = np.array(bit_width, dtype=shifts_arr.dtype)
    # Cast shift amounts explicitly to the target unsigned type for safety
    left_shift_amount = np.subtract(bit_width_np, shifts_arr).astype(x_arr.dtype)
    shifts_arr_casted = shifts_arr.astype(x_arr.dtype) # Cast shifts amount too

    right_part = np.right_shift(x_arr, shifts_arr_casted)
    left_part = np.left_shift(x_arr, left_shift_amount)

    # Ensure final result is the same dtype as input
    return np.bitwise_or(left_part, right_part).astype(x_arr.dtype)