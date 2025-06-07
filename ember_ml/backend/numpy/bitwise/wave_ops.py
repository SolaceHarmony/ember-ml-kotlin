"""
NumPy binary wave operations for ember_ml.

This module provides NumPy implementations of operations specific to
binary wave processing, such as interference and propagation.
"""

import numpy as np
from typing import List, Union

# Import NumpyTensor dynamically within functions
# from ember_ml.backend.numpy.tensor import NumpyTensor
from ember_ml.backend.numpy.types import TensorLike
from .basic_ops import bitwise_and, bitwise_or, bitwise_xor # Import from sibling module

def binary_wave_interference(waves: List[TensorLike], mode: str = 'xor') -> np.ndarray:
    """
    Apply wave interference between multiple binary patterns element-wise.

    Args:
        waves: List of input tensors or compatible types (must be integer type).
               All tensors must be broadcastable to a common shape.
        mode: Interference type ('xor', 'and', or 'or'). Defaults to 'xor'.

    Returns:
        NumPy array representing the interference pattern.
    """
    if not waves:
        raise ValueError("Input list 'waves' cannot be empty.")

    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor_ops = NumpyTensor()

    # Convert first wave and check type
    result_arr = tensor_ops.convert_to_tensor(waves[0])
    if not np.issubdtype(result_arr.dtype, np.integer):
        raise TypeError(f"binary_wave_interference requires integer types, got {result_arr.dtype} for first wave.")

    # Apply interference iteratively
    for wave_like in waves[1:]:
        wave_arr = tensor_ops.convert_to_tensor(wave_like)
        if not np.issubdtype(wave_arr.dtype, np.integer):
             raise TypeError(f"binary_wave_interference requires integer types, got {wave_arr.dtype}.")
        # Ensure dtypes match or are compatible for bitwise ops if necessary
        # NumPy typically handles broadcasting and type promotion for bitwise ops
        if mode == 'xor':
            result_arr = bitwise_xor(result_arr, wave_arr)
        elif mode == 'and':
            result_arr = bitwise_and(result_arr, wave_arr)
        elif mode == 'or':
            result_arr = bitwise_or(result_arr, wave_arr)
        else:
            raise ValueError(f"Unsupported interference mode: '{mode}'. Choose 'xor', 'and', or 'or'.")

    return result_arr

def binary_wave_propagate(wave: TensorLike, shift: TensorLike) -> np.ndarray:
    """
    Propagate a binary wave by shifting its bits.

    Positive shift corresponds to left shift, negative to right shift.

    Args:
        wave: Input tensor or compatible type (must be integer type).
        shift: Number of positions to shift (integer or tensor).

    Returns:
        NumPy array representing the propagated wave pattern.
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    from .shift_ops import left_shift, right_shift # Import from sibling
    tensor_ops = NumpyTensor()
    wave_arr = tensor_ops.convert_to_tensor(wave)
    shift_arr = tensor_ops.convert_to_tensor(shift)

    if not np.issubdtype(wave_arr.dtype, np.integer):
        raise TypeError(f"binary_wave_propagate requires an integer type for wave, got {wave_arr.dtype}")
    if not np.issubdtype(shift_arr.dtype, np.integer):
         shift_arr = shift_arr.astype(np.int32) # Cast shift to int

    # Handle scalar shift for clarity
    if shift_arr.ndim == 0:
        shift_val = shift_arr.item()
        if shift_val >= 0:
            return left_shift(wave_arr, shift_arr)
        else:
            return right_shift(wave_arr, np.negative(shift_arr))
    else:
        # Handle tensor shift using where (more complex but handles element-wise shifts)
        positive_mask = np.greater_equal(shift_arr, 0)
        negative_mask = np.less(shift_arr, 0)

        # Calculate results for both positive and negative shifts
        left_shifted = left_shift(wave_arr, shift_arr) # left_shift handles 0 shift correctly
        right_shifted = right_shift(wave_arr, np.negative(shift_arr)) # right_shift handles 0 shift correctly

        # Combine results based on the mask
        # Initialize with left_shifted for positive shifts (or zero shifts)
        result = np.where(positive_mask, left_shifted, wave_arr)
        # Update with right_shifted for negative shifts
        result = np.where(negative_mask, right_shifted, result)
        return result


def create_duty_cycle(length: int, duty_cycle: float) -> np.ndarray:
    """
    Create a binary pattern array with a specified duty cycle.

    Args:
        length: The length of the binary pattern (number of bits).
        duty_cycle: The fraction of bits that should be 1 (between 0.0 and 1.0).

    Returns:
        NumPy array (int32) representing the binary pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(duty_cycle, (float, int)) or not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("Duty cycle must be a float or int between 0.0 and 1.0.")

    num_ones = int(round(length * duty_cycle)) # Round to nearest integer

    # Create pattern with 1s at the beginning
    if num_ones <= 0:
        pattern = np.zeros((length,), dtype=np.int32)
    elif num_ones >= length:
        pattern = np.ones((length,), dtype=np.int32)
    else:
        pattern = np.concatenate([
            np.ones((num_ones,), dtype=np.int32),
            np.zeros((length - num_ones,), dtype=np.int32)
        ])

    return pattern

def generate_blocky_sin(length: int, half_period: int) -> np.ndarray:
    """
    Generate a blocky sine wave pattern (square wave).

    Args:
        length: The length of the binary pattern (number of bits).
        half_period: Half the period of the wave in bits.

    Returns:
        NumPy array (int32) representing the blocky sine wave pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(half_period, int) or half_period <= 0:
        raise ValueError("Half period must be a positive integer.")

    full_period = 2 * half_period
    indices = np.arange(length)

    # Calculate cycle position: indices % full_period
    cycle_position = np.remainder(indices, np.array(full_period, dtype=indices.dtype))

    # Create pattern: 1 for first half of period, 0 for second half
    pattern = np.where(np.less(cycle_position, np.array(half_period, dtype=cycle_position.dtype)),
                       np.array(1, dtype=np.int32),
                       np.array(0, dtype=np.int32))

    return pattern