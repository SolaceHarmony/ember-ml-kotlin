"""
PyTorch comparison operations for ember_ml.

This module provides PyTorch implementations of comparison operations.
"""

import torch
from typing import Any
from ember_ml.backend.torch.types import TensorLike

# We avoid creating global instances to prevent circular imports
# Each function will create its own instances when needed


def equal(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if two tensors are equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x == y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.eq(x_tensor, y_tensor)


def not_equal(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if two tensors are not equal element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x != y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.ne(x_tensor, y_tensor)


def less(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if one tensor is less than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x < y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.lt(x_tensor, y_tensor)


def less_equal(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if one tensor is less than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x <= y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.le(x_tensor, y_tensor)


def greater(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if one tensor is greater than another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x > y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.gt(x_tensor, y_tensor)


def greater_equal(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Check if one tensor is greater than or equal to another element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x >= y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.ge(x_tensor, y_tensor)


def logical_and(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the logical AND of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x AND y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.logical_and(x_tensor, y_tensor)


def logical_or(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the logical OR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x OR y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.logical_or(x_tensor, y_tensor)


def logical_not(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the logical NOT of a tensor element-wise.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean tensor with True where NOT x
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    
    return torch.logical_not(x_tensor)


def logical_xor(x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the logical XOR of two tensors element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Boolean tensor with True where x XOR y
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.logical_xor(x_tensor, y_tensor)


def allclose(x: TensorLike, y: TensorLike, rtol: float = 1e-5, atol: float = 1e-8) -> Any: # Changed bool to Any
    """
    Check if all elements of two tensors are close within a tolerance.
    
    Args:
        x: First tensor
        y: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean indicating if all elements are close
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.allclose(x_tensor, y_tensor, rtol=rtol, atol=atol)

def isclose(x: TensorLike, y: TensorLike, rtol: float = 1e-5, atol: float = 1e-8) -> Any: # Changed torch.Tensor to Any
    """
    Check if elements of two tensors are close within a tolerance element-wise.
    
    Args:
        x: First tensor
        y: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean tensor with True where elements are close
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.isclose(x_tensor, y_tensor, rtol=rtol, atol=atol)


def all(x: TensorLike, axis: Any = None) -> Any: # No change needed, already Any
    """
    Check if all elements in a tensor are True.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.
            
    Returns:
        Boolean tensor with True if all elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    
    if axis is None:
        return torch.all(x_tensor)
    else:
        return torch.all(x_tensor, dim=axis)


def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean tensor
        x: Tensor with values to use where condition is True
        y: Tensor with values to use where condition is False
        
    Returns:
        Tensor with values from x where condition is True, and values from y elsewhere
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    condition_tensor = tensor_ops.convert_to_tensor(condition)
    x_tensor = tensor_ops.convert_to_tensor(x)
    y_tensor = tensor_ops.convert_to_tensor(y)
    
    return torch.where(condition_tensor, x_tensor, y_tensor)


def isnan(x: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Test element-wise for NaN values.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean tensor with True where x is NaN, False otherwise
    """
    # Create instances for each call to avoid circular imports
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    
    x_tensor = tensor_ops.convert_to_tensor(x)
    
    return torch.isnan(x_tensor)


# Removed TorchComparisonOps class as it's redundant with standalone functions