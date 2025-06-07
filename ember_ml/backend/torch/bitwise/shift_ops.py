"""
PyTorch shift bitwise operations for ember_ml.

This module provides PyTorch implementations of bitwise shift operations
(left_shift, right_shift, rotate_left, rotate_right).
"""

import torch
from typing import Union

# Import TorchTensor dynamically within functions
# from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.types import TensorLike

def left_shift(x: TensorLike, shifts: TensorLike) -> torch.Tensor:
    """
    Shift the bits of x to the left by shifts positions.

    Args:
        x: Input tensor or compatible type.
        shifts: Number of bits to shift (integer or tensor).

    Returns:
        PyTorch tensor with x shifted left by shifts bits.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    shifts_tensor = tensor_ops.convert_to_tensor(shifts)
    # Ensure integer types
    if not torch.is_floating_point(x_tensor):
        if not torch.is_floating_point(shifts_tensor):
             # PyTorch requires shifts to be tensor or scalar, handles broadcasting
             return torch.bitwise_left_shift(x_tensor, shifts_tensor)
        else:
             raise TypeError(f"Shifts must be an integer type for left_shift, got {shifts_tensor.dtype}")
    else:
         raise TypeError(f"Input must be an integer type for left_shift, got {x_tensor.dtype}")


def right_shift(x: TensorLike, shifts: TensorLike) -> torch.Tensor:
    """
    Shift the bits of x to the right by shifts positions.

    Args:
        x: Input tensor or compatible type.
        shifts: Number of bits to shift (integer or tensor).

    Returns:
        PyTorch tensor with x shifted right by shifts bits.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    shifts_tensor = tensor_ops.convert_to_tensor(shifts)
    # Ensure integer types
    if not torch.is_floating_point(x_tensor):
        if not torch.is_floating_point(shifts_tensor):
             return torch.bitwise_right_shift(x_tensor, shifts_tensor)
        else:
             raise TypeError(f"Shifts must be an integer type for right_shift, got {shifts_tensor.dtype}")
    else:
         raise TypeError(f"Input must be an integer type for right_shift, got {x_tensor.dtype}")


def rotate_left(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> torch.Tensor:
    """
    Rotate the bits of x to the left by shifts positions.

    Args:
        x: Input tensor or compatible type (must be integer type).
        shifts: Number of bits to rotate (integer or tensor).
        bit_width: The bit width of the integer type (e.g., 8, 16, 32, 64).

    Returns:
        PyTorch tensor with x rotated left by shifts bits.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    shifts_tensor = tensor_ops.convert_to_tensor(shifts)

    # Ensure integer types
    if torch.is_floating_point(x_tensor):
        raise TypeError(f"rotate_left requires an integer type, got {x_tensor.dtype}")
    if torch.is_floating_point(shifts_tensor):
        shifts_tensor = shifts_tensor.to(torch.int) # Cast shifts to int

    # Determine appropriate integer type based on bit_width if needed,
    # but PyTorch shifts handle different int types. Assume input type is correct.
    # Check bit_width validity
    if bit_width not in [8, 16, 32, 64]:
         print(f"Warning: Unusual bit_width {bit_width} for rotate_left. Ensure input tensor type matches.")


    # Normalize shifts to be within [0, bit_width)
    bit_width_tensor = torch.tensor(bit_width, dtype=shifts_tensor.dtype, device=shifts_tensor.device)
    shifts_tensor = torch.remainder(shifts_tensor, bit_width_tensor)

    # Perform rotation using shifts and bitwise OR
    right_shift_amount = torch.subtract(bit_width_tensor, shifts_tensor)

    left_part = torch.bitwise_left_shift(x_tensor, shifts_tensor)
    right_part = torch.bitwise_right_shift(x_tensor, right_shift_amount)

    return torch.bitwise_or(left_part, right_part)


def rotate_right(x: TensorLike, shifts: TensorLike, bit_width: int = 32) -> torch.Tensor:
    """
    Rotate the bits of x to the right by shifts positions.

    Args:
        x: Input tensor or compatible type (must be integer type).
        shifts: Number of bits to rotate (integer or tensor).
        bit_width: The bit width of the integer type (e.g., 8, 16, 32, 64).

    Returns:
        PyTorch tensor with x rotated right by shifts bits.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    shifts_tensor = tensor_ops.convert_to_tensor(shifts)

    # Ensure integer types
    if torch.is_floating_point(x_tensor):
        raise TypeError(f"rotate_right requires an integer type, got {x_tensor.dtype}")
    if torch.is_floating_point(shifts_tensor):
        shifts_tensor = shifts_tensor.to(torch.int) # Cast shifts to int

    if bit_width not in [8, 16, 32, 64]:
         print(f"Warning: Unusual bit_width {bit_width} for rotate_right. Ensure input tensor type matches.")

    # Normalize shifts to be within [0, bit_width)
    bit_width_tensor = torch.tensor(bit_width, dtype=shifts_tensor.dtype, device=shifts_tensor.device)
    shifts_tensor = torch.remainder(shifts_tensor, bit_width_tensor)

    # Perform rotation using shifts and bitwise OR
    left_shift_amount = torch.subtract(bit_width_tensor, shifts_tensor)

    right_part = torch.bitwise_right_shift(x_tensor, shifts_tensor)
    left_part = torch.bitwise_left_shift(x_tensor, left_shift_amount)

    return torch.bitwise_or(left_part, right_part)