"""
PyTorch basic bitwise operations for ember_ml.

This module provides PyTorch implementations of basic bitwise operations
(AND, OR, XOR, NOT).
"""

import torch

# Import TorchTensor dynamically within functions
# from ember_ml.backend.torch.tensor import TorchTensor
from ember_ml.backend.torch.types import TensorLike

def bitwise_and(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute the bitwise AND of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        PyTorch tensor with the element-wise bitwise AND.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    # Ensure boolean or integer type for PyTorch bitwise ops
    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"Bitwise AND requires boolean or integer tensors, got {x_tensor.dtype}")
    if not (y_tensor.dtype == torch.bool or torch.is_floating_point(y_tensor) == False):
         raise TypeError(f"Bitwise AND requires boolean or integer tensors, got {y_tensor.dtype}")
    return torch.bitwise_and(x_tensor, y_tensor)

def bitwise_or(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute the bitwise OR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        PyTorch tensor with the element-wise bitwise OR.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"Bitwise OR requires boolean or integer tensors, got {x_tensor.dtype}")
    if not (y_tensor.dtype == torch.bool or torch.is_floating_point(y_tensor) == False):
         raise TypeError(f"Bitwise OR requires boolean or integer tensors, got {y_tensor.dtype}")
    return torch.bitwise_or(x_tensor, y_tensor)

def bitwise_xor(x: TensorLike, y: TensorLike) -> torch.Tensor:
    """
    Compute the bitwise XOR of x and y element-wise.

    Args:
        x: First input tensor or compatible type.
        y: Second input tensor or compatible type.

    Returns:
        PyTorch tensor with the element-wise bitwise XOR.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"Bitwise XOR requires boolean or integer tensors, got {x_tensor.dtype}")
    if not (y_tensor.dtype == torch.bool or torch.is_floating_point(y_tensor) == False):
         raise TypeError(f"Bitwise XOR requires boolean or integer tensors, got {y_tensor.dtype}")
    return torch.bitwise_xor(x_tensor, y_tensor)

def bitwise_not(x: TensorLike) -> torch.Tensor:
    """
    Compute the bitwise NOT (inversion) of x element-wise.

    Args:
        x: Input tensor or compatible type.

    Returns:
        PyTorch tensor with the element-wise bitwise NOT.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    x_tensor = tensor_ops.convert_to_tensor(x)
    if not (x_tensor.dtype == torch.bool or torch.is_floating_point(x_tensor) == False):
         raise TypeError(f"Bitwise NOT requires boolean or integer tensors, got {x_tensor.dtype}")
    # PyTorch uses bitwise_not for inversion
    return torch.bitwise_not(x_tensor)