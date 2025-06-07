"""
MLX binary wave operations for ember_ml.

This module provides MLX implementations of operations specific to
binary wave processing, such as interference and propagation.
"""

import mlx.core as mx
from typing import List, Union

# Import MLXTensor dynamically within functions
# from ember_ml.backend.mlx.tensor import MLXTensor
from ember_ml.backend.mlx.types import TensorLike
from typing import List, Union
import mlx.core as mx
from ember_ml.backend.mlx.types import TensorLike

def binary_wave_interference(waves: List[TensorLike], mode: str = 'xor') -> mx.array:
    """
    Apply wave interference between multiple binary patterns element-wise.

    Args:
        waves: List of input tensors or compatible types (must be integer type).
               All tensors must be broadcastable to a common shape.
        mode: Interference type ('xor', 'and', or 'or'). Defaults to 'xor'.

    Returns:
        MLX array representing the interference pattern.
    """
    if not waves:
        raise ValueError("Input list 'waves' cannot be empty.")

    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor_ops = MLXTensor()

    # Convert first wave and check type
    result_arr = tensor_ops.convert(waves[0])
    if not mx.issubdtype(result_arr.dtype, mx.integer):
        raise TypeError(f"binary_wave_interference requires integer types, got {result_arr.dtype} for first wave.")

    # Apply interference iteratively
    for wave_like in waves[1:]:
        wave_arr = tensor_ops.convert(wave_like)
        if not mx.issubdtype(wave_arr.dtype, mx.integer):
             raise TypeError(f"binary_wave_interference requires integer types, got {wave_arr.dtype}.")
        # Ensure dtypes match or are compatible for bitwise ops if necessary
        # MLX typically handles broadcasting and type promotion for bitwise ops
        if mode == 'xor':
            result_arr = bitwise_xor(result_arr, wave_arr)
        elif mode == 'and':
            result_arr = bitwise_and(result_arr, wave_arr)
        elif mode == 'or':
            result_arr = bitwise_or(result_arr, wave_arr)
        else:
            raise ValueError(f"Unsupported interference mode: '{mode}'. Choose 'xor', 'and', or 'or'.")

    return result_arr

def binary_wave_propagate(wave: TensorLike, shift: TensorLike) -> mx.array:
    """
    Propagate a binary wave by shifting its bits.

    Positive shift corresponds to left shift, negative to right shift.

    Args:
        wave: Input tensor or compatible type (must be integer type).
        shift: Number of positions to shift (integer or tensor).

    Returns:
        MLX array representing the propagated wave pattern.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    from ember_ml.backend.mlx.tensor import MLXTensor
    from .shift_ops import left_shift, right_shift # Import from sibling
    tensor_ops = MLXTensor()
    wave_arr = tensor_ops.convert(wave)
    shift_arr = tensor_ops.convert(shift)

    if not mx.issubdtype(wave_arr.dtype, mx.integer):
        raise TypeError(f"binary_wave_propagate requires an integer type for wave, got {wave_arr.dtype}")
    if not mx.issubdtype(shift_arr.dtype, mx.integer):
         shift_arr = mx.astype(shift_arr, mx.int32) # Cast shift to int

    # Handle scalar shift for clarity
    if shift_arr.ndim == 0:
        shift_val = shift_arr.item()
        if shift_val >= 0:
            return left_shift(wave_arr, shift_arr)
        else:
            return right_shift(wave_arr, mx.negative(shift_arr))
    else:
        # Handle tensor shift using where (more complex but handles element-wise shifts)
        positive_mask = mx.greater_equal(shift_arr, 0)
        negative_mask = mx.less(shift_arr, 0)

        # Calculate results for both positive and negative shifts
        left_shifted = left_shift(wave_arr, shift_arr) # left_shift handles 0 shift correctly
        right_shifted = right_shift(wave_arr, mx.negative(shift_arr)) # right_shift handles 0 shift correctly

        # Combine results based on the mask
        # Initialize with left_shifted for positive shifts (or zero shifts)
        result = mx.where(positive_mask, left_shifted, wave_arr)
        # Update with right_shifted for negative shifts
        result = mx.where(negative_mask, right_shifted, result)
        return result


def create_duty_cycle(length: int, duty_cycle: float) -> mx.array:
    """
    Create a binary pattern array with a specified duty cycle.

    Args:
        length: The length of the binary pattern (number of bits).
        duty_cycle: The fraction of bits that should be 1 (between 0.0 and 1.0).

    Returns:
        MLX array (int32) representing the binary pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(duty_cycle, (float, int)) or not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("Duty cycle must be a float or int between 0.0 and 1.0.")

    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    num_ones = mx.round(mx.multiply(tensor.convert(length, dtype=mx.float32), tensor.convert(duty_cycle, dtype=mx.float32)))

    # Create pattern with 1s at the beginning
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    length_arr = tensor.convert(length, dtype=mx.int32)
    num_ones_arr = mx.astype(num_ones, mx.int32)

    pattern = mx.where(
        mx.less_equal(num_ones_arr, tensor.convert(0, dtype=mx.int32)),
        mx.zeros((length,), dtype=mx.int32),
        mx.where(
            mx.greater_equal(num_ones_arr, length_arr),
            mx.ones((length,), dtype=mx.int32),
            mx.concatenate([
                mx.ones((num_ones_arr.item(),), dtype=mx.int32),
                mx.zeros((length_arr.item() - num_ones_arr.item(),), dtype=mx.int32)
            ])
        )
    )

    return pattern

def generate_blocky_sin(length: int, half_period: int) -> mx.array:
    """
    Generate a blocky sine wave pattern (square wave).

    Args:
        length: The length of the binary pattern (number of bits).
        half_period: Half the period of the wave in bits.

    Returns:
        MLX array (int32) representing the blocky sine wave pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(half_period, int) or half_period <= 0:
        raise ValueError("Half period must be a positive integer.")

    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    full_period = mx.multiply(tensor.convert(2, dtype=mx.int32), tensor.convert(half_period, dtype=mx.int32))
    indices = mx.arange(tensor.convert(length, dtype=mx.int32))

    # Calculate cycle position: indices % full_period
    cycle_position = mx.remainder(indices, mx.array(full_period, dtype=indices.dtype))

    # Create pattern: 1 for first half of period, 0 for second half
    pattern = mx.where(mx.less(cycle_position, mx.array(half_period, dtype=cycle_position.dtype)),
                       mx.array(1, dtype=mx.int32),
                       mx.array(0, dtype=mx.int32))

    return pattern