"""
PyTorch bit manipulation operations for ember_ml.

This module provides PyTorch implementations of bit manipulation operations
(count_ones, count_zeros, get_bit, set_bit, toggle_bit).
"""

import torch

# Import TorchTensor dynamically within functions
# from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.types import TensorLike

def count_ones(x: TensorLike) -> torch.Tensor:
    """
    Count the number of set bits (1s) in each element of x (population count).

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        PyTorch tensor with the count of set bits for each element (int32).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)

    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"count_ones requires boolean or integer tensors, got {x_tensor.dtype}")

    # PyTorch >= 1.9 has bitwise_count_ones
    if hasattr(torch, 'bitwise_count_ones'):
        # Return as int32 for consistency across backends
        return torch.bitwise_count_ones(x_tensor).to(torch.int32)
    else:
        # Fallback implementation for older PyTorch versions using bit manipulation
        # This is the "population count" or "Hamming weight" algorithm
        
        # Determine bit width based on dtype
        if x_tensor.dtype == torch.uint8 or x_tensor.dtype == torch.int8:
            bit_width = 8
        elif x_tensor.dtype == torch.int16:  # torch.uint16 doesn't exist
            bit_width = 16
        elif x_tensor.dtype == torch.int32:  # torch.uint32 doesn't exist
            bit_width = 32
        elif x_tensor.dtype == torch.int64:  # torch.uint64 doesn't exist
            bit_width = 64
        elif x_tensor.dtype == torch.bool:
            # For boolean tensors, just convert to int32 (True=1, False=0)
            return x_tensor.to(torch.int32)
        else:
            raise TypeError(f"Unsupported integer type for count_ones: {x_tensor.dtype}")
        
        # Convert to unsigned type for consistent bit manipulation
        # We'll use int32 or int64 for the computation
        if bit_width <= 32:
            x_unsigned = x_tensor.to(torch.int32)
            
            # For int8, mask to only consider the lowest 8 bits
            if x_tensor.dtype == torch.int8:
                mask_8bit = torch.tensor(0xFF, dtype=torch.int32, device=x_tensor.device)
                x_unsigned = torch.bitwise_and(x_unsigned, mask_8bit)
                
            # Constants for 32-bit integers
            m1 = torch.tensor(0x55555555, dtype=torch.int32, device=x_tensor.device)  # 01010101...
            m2 = torch.tensor(0x33333333, dtype=torch.int32, device=x_tensor.device)  # 00110011...
            m4 = torch.tensor(0x0F0F0F0F, dtype=torch.int32, device=x_tensor.device)  # 00001111...
        else:
            x_unsigned = x_tensor.to(torch.int64)
            
            # For int8, mask to only consider the lowest 8 bits
            if x_tensor.dtype == torch.int8:
                mask_8bit = torch.tensor(0xFF, dtype=torch.int64, device=x_tensor.device)
                x_unsigned = torch.bitwise_and(x_unsigned, mask_8bit)
                
            # Constants for 64-bit integers
            m1 = torch.tensor(0x5555555555555555, dtype=torch.int64, device=x_tensor.device)
            m2 = torch.tensor(0x3333333333333333, dtype=torch.int64, device=x_tensor.device)
            m4 = torch.tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64, device=x_tensor.device)
        
        # Count bits in parallel using the population count algorithm
        # Step 1: Count bits in pairs (0-2)
        x_unsigned = torch.add(torch.bitwise_and(x_unsigned, m1),
                              torch.bitwise_and(torch.bitwise_right_shift(x_unsigned, 1), m1))
        # Step 2: Count bits in nibbles (0-4)
        x_unsigned = torch.add(torch.bitwise_and(x_unsigned, m2),
                              torch.bitwise_and(torch.bitwise_right_shift(x_unsigned, 2), m2))
        # Step 3: Count bits in bytes (0-8)
        x_unsigned = torch.add(torch.bitwise_and(x_unsigned, m4),
                              torch.bitwise_and(torch.bitwise_right_shift(x_unsigned, 4), m4))
        
        # For 8-bit integers, we're done
        if bit_width == 8:
            return x_unsigned.to(torch.int32)
        
        # For wider integers, continue summing bytes
        x_unsigned = torch.bitwise_and(
            torch.add(x_unsigned, torch.bitwise_right_shift(x_unsigned, 8)),
            torch.tensor(0x00FF00FF, dtype=x_unsigned.dtype, device=x_tensor.device)
        )
        
        if bit_width <= 16:
            return x_unsigned.to(torch.int32)
        
        # For 32-bit and 64-bit integers, continue summing
        x_unsigned = torch.bitwise_and(
            torch.add(x_unsigned, torch.bitwise_right_shift(x_unsigned, 16)),
            torch.tensor(0x0000FFFF, dtype=x_unsigned.dtype, device=x_tensor.device)
        )
        
        if bit_width <= 32:
            return x_unsigned.to(torch.int32)
        
        # For 64-bit integers, one more step
        x_unsigned = torch.bitwise_and(
            torch.add(x_unsigned, torch.bitwise_right_shift(x_unsigned, 32)),
            torch.tensor(0xFFFFFFFF, dtype=x_unsigned.dtype, device=x_tensor.device)
        )
        
        return x_unsigned.to(torch.int32)  # Return as int32 for consistency


def count_zeros(x: TensorLike) -> torch.Tensor:
    """
    Count the number of unset bits (0s) in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).

    Returns:
        PyTorch tensor with the count of unset bits for each element (int32).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)

    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"count_zeros requires boolean or integer tensors, got {x_tensor.dtype}")

    # Determine bit width based on dtype
    if x_tensor.dtype == torch.uint8 or x_tensor.dtype == torch.int8:
        bit_width = 8
    elif x_tensor.dtype == torch.int16: # torch.uint16 doesn't exist
        bit_width = 16
    elif x_tensor.dtype == torch.int32: # torch.uint32 doesn't exist
        bit_width = 32
    elif x_tensor.dtype == torch.int64: # torch.uint64 doesn't exist
        bit_width = 64
    elif x_tensor.dtype == torch.bool:
         bit_width = 1 # Count zeros in boolean tensor
    else:
        # This case should ideally be caught by the initial check, but added for safety
        raise TypeError(f"Unsupported integer type for count_zeros: {x_tensor.dtype}")

    ones = count_ones(x_tensor) # Returns int32
    # Ensure bit_width is a tensor of the same dtype and device for subtraction
    bit_width_tensor = torch.tensor(bit_width, dtype=ones.dtype, device=ones.device)
    return torch.subtract(bit_width_tensor, ones)


def get_bit(x: TensorLike, position: TensorLike) -> torch.Tensor:
    """
    Get the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        PyTorch tensor with the bit value (0 or 1) at the specified position(s) (int32).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    pos_tensor = tensor_ops.convert_to_tensor(position)

    if torch.is_floating_point(x_tensor):
        raise TypeError(f"get_bit requires an integer type for x, got {x_tensor.dtype}")
    if torch.is_floating_point(pos_tensor):
         pos_tensor = pos_tensor.to(torch.int) # Cast position to int

    # Create mask: 1 << position
    # Ensure '1' is a tensor of the same type and device as x_tensor
    one = torch.tensor(1, dtype=x_tensor.dtype, device=x_tensor.device)
    mask = torch.bitwise_left_shift(one, pos_tensor)

    # Extract bit: (x & mask) >> position
    extracted_bit = torch.bitwise_right_shift(torch.bitwise_and(x_tensor, mask), pos_tensor)
    # Return as int32 for consistency
    return extracted_bit.to(torch.int32)


def set_bit(x: TensorLike, position: TensorLike, value: TensorLike) -> torch.Tensor:
    """
    Set the bit at the specified position in each element of x to value (0 or 1).

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.
        value: Bit value(s) (0 or 1). Integer or tensor.

    Returns:
        PyTorch tensor with the bit at the specified position(s) set.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    pos_tensor = tensor_ops.convert_to_tensor(position)
    val_tensor = tensor_ops.convert_to_tensor(value)

    if torch.is_floating_point(x_tensor):
        raise TypeError(f"set_bit requires an integer type for x, got {x_tensor.dtype}")
    if torch.is_floating_point(pos_tensor):
         pos_tensor = pos_tensor.to(torch.int)
    if torch.is_floating_point(val_tensor):
         val_tensor = val_tensor.to(torch.int) # Cast value to int

    # Create mask: 1 << position
    one = torch.tensor(1, dtype=x_tensor.dtype, device=x_tensor.device)
    mask = torch.bitwise_left_shift(one, pos_tensor)

    # Clear the bit: x & (~mask)
    cleared_x = torch.bitwise_and(x_tensor, torch.bitwise_not(mask))

    # Prepare value to be set: (value & 1) << position
    # Ensure value is 0 or 1, then shift
    value_bit = torch.bitwise_and(val_tensor.to(x_tensor.dtype), one) # Cast value to x's dtype
    value_shifted = torch.bitwise_left_shift(value_bit, pos_tensor)

    # Set the bit using OR: cleared_x | value_shifted
    return torch.bitwise_or(cleared_x, value_shifted)


def toggle_bit(x: TensorLike, position: TensorLike) -> torch.Tensor:
    """
    Toggle the bit at the specified position in each element of x.

    Args:
        x: Input tensor or compatible type (must be integer type).
        position: Bit position(s) (0-based, LSB). Integer or tensor.

    Returns:
        PyTorch tensor with the bit at the specified position(s) toggled.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    pos_tensor = tensor_ops.convert_to_tensor(position)

    if torch.is_floating_point(x_tensor):
        raise TypeError(f"toggle_bit requires an integer type for x, got {x_tensor.dtype}")
    if torch.is_floating_point(pos_tensor):
         pos_tensor = pos_tensor.to(torch.int)

    # Create mask: 1 << position
    one = torch.tensor(1, dtype=x_tensor.dtype, device=x_tensor.device)
    mask = torch.bitwise_left_shift(one, pos_tensor)

    # Toggle the bit using XOR: x ^ mask
    return torch.bitwise_xor(x_tensor, mask)