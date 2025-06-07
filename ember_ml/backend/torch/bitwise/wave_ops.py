"""
PyTorch binary wave operations for ember_ml.

This module provides PyTorch implementations of operations specific to
binary wave processing, such as interference and propagation.
"""

import torch
from typing import List

# Import TorchTensor dynamically within functions
# from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.types import TensorLike
from ember_ml.backend.torch.bitwise.basic_ops import bitwise_and, bitwise_or, bitwise_xor # Import from sibling module
from ember_ml.backend.torch.bitwise.shift_ops import left_shift, right_shift # Import from sibling module

def binary_wave_interference(waves: List[TensorLike], mode: str = 'xor') -> torch.Tensor:
    """
    Apply wave interference between multiple binary patterns element-wise.

    Args:
        waves: List of input tensors or compatible types (must be integer or bool type).
               All tensors must be broadcastable to a common shape.
        mode: Interference type ('xor', 'and', or 'or'). Defaults to 'xor'.

    Returns:
        PyTorch tensor representing the interference pattern.
    """
    if not waves:
        raise ValueError("Input list 'waves' cannot be empty.")

    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()

    # Convert first wave and check type
    result_tensor = tensor_ops.convert_to_tensor(waves[0])
    if not (result_tensor.dtype == torch.bool or torch.is_floating_point(result_tensor) == False):
         raise TypeError(f"binary_wave_interference requires boolean or integer tensors, got {result_tensor.dtype} for first wave.")

    # Apply interference iteratively
    for wave_like in waves[1:]:
        wave_tensor = tensor_ops.convert_to_tensor(wave_like)
        if not (wave_tensor.dtype == torch.bool or torch.is_floating_point(wave_tensor) == False):
             raise TypeError(f"binary_wave_interference requires boolean or integer tensors, got {wave_tensor.dtype}.")
        # PyTorch handles broadcasting and type promotion for bitwise ops
        if mode == 'xor':
            result_tensor = bitwise_xor(result_tensor, wave_tensor)
        elif mode == 'and':
            result_tensor = bitwise_and(result_tensor, wave_tensor)
        elif mode == 'or':
            result_tensor = bitwise_or(result_tensor, wave_tensor)
        else:
            raise ValueError(f"Unsupported interference mode: '{mode}'. Choose 'xor', 'and', or 'or'.")

    return result_tensor

def binary_wave_propagate(wave: TensorLike, shift: TensorLike) -> torch.Tensor:
    """
    Propagate a binary wave by shifting its bits.

    Positive shift corresponds to left shift, negative to right shift.

    Args:
        wave: Input tensor or compatible type (must be integer type).
        shift: Number of positions to shift (integer or tensor).

    Returns:
        PyTorch tensor representing the propagated wave pattern.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    wave_tensor = tensor_ops.convert_to_tensor(wave)
    shift_tensor = tensor_ops.convert_to_tensor(shift)

    if torch.is_floating_point(wave_tensor):
        raise TypeError(f"binary_wave_propagate requires an integer type for wave, got {wave_tensor.dtype}")
    if torch.is_floating_point(shift_tensor):
         shift_tensor = shift_tensor.to(torch.int) # Cast shift to int

    # Handle scalar shift for clarity and potential efficiency
    if shift_tensor.numel() == 1:
        shift_val = shift_tensor.item()
        if shift_val >= 0:
            return left_shift(wave_tensor, shift_tensor)
        else:
            # Use positive shift value for right_shift
            return right_shift(wave_tensor, torch.neg(shift_tensor))
    else:
        # Handle tensor shift using masks (more complex but handles element-wise shifts)
        zero = torch.tensor(0, dtype=shift_tensor.dtype, device=shift_tensor.device)
        positive_mask = torch.ge(shift_tensor, zero)
        negative_mask = torch.lt(shift_tensor, zero)

        # Calculate results for both positive and negative shifts
        left_shifted = left_shift(wave_tensor, shift_tensor)
        # Use positive shift value for right_shift
        right_shifted = right_shift(wave_tensor, torch.neg(shift_tensor))

        # Combine results based on the mask
        # Initialize with wave_tensor where mask is false, then update
        result = torch.where(positive_mask, left_shifted, wave_tensor)
        result = torch.where(negative_mask, right_shifted, result)
        return result


def create_duty_cycle(length: int, duty_cycle: float) -> torch.Tensor:
    """
    Create a binary pattern tensor with a specified duty cycle.

    Args:
        length: The length of the binary pattern (number of bits).
        duty_cycle: The fraction of bits that should be 1 (between 0.0 and 1.0).

    Returns:
        PyTorch tensor (int32) representing the binary pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(duty_cycle, (float, int)) or not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("Duty cycle must be a float or int between 0.0 and 1.0.")

    num_ones = int(round(length * duty_cycle)) # Round to nearest integer

    # Create pattern with 1s at the beginning
    if num_ones <= 0:
        pattern = torch.zeros(length, dtype=torch.int32)
    elif num_ones >= length:
        pattern = torch.ones(length, dtype=torch.int32)
    else:
        pattern = torch.cat([
            torch.ones(num_ones, dtype=torch.int32),
            torch.zeros(length - num_ones, dtype=torch.int32)
        ])

    return pattern

def generate_blocky_sin(length: int, half_period: int) -> torch.Tensor:
    """
    Generate a blocky sine wave pattern (square wave).

    Args:
        length: The length of the binary pattern (number of bits).
        half_period: Half the period of the wave in bits.

    Returns:
        PyTorch tensor (int32) representing the blocky sine wave pattern.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    if not isinstance(half_period, int) or half_period <= 0:
        raise ValueError("Half period must be a positive integer.")

    full_period = 2 * half_period
    indices = torch.arange(length)

    # Calculate cycle position: indices % full_period
    cycle_position = torch.remainder(indices, full_period)

    # Create pattern: 1 for first half of period, 0 for second half
    pattern = torch.where(cycle_position < half_period,
                          torch.tensor(1, dtype=torch.int32),
                          torch.tensor(0, dtype=torch.int32))

    return pattern