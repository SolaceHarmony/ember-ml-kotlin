"""PyTorch tensor creation operations."""

from typing import Optional, Union

import torch
from torch import Tensor

from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.tensor.ops.utility import _create_new_tensor  # Import the helper
from ember_ml.backend.torch.types import TensorLike, DType, Shape, ShapeLike, ScalarLike


def zeros(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.

    Args:
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Tensor of zeros
    """
    # Pass shape via kwargs
    return _create_new_tensor(torch.zeros, dtype=dtype, device=device, shape=shape)

def ones(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones.

    Args:
        shape: The shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Tensor of ones
    """
    # Pass shape via kwargs
    return _create_new_tensor(torch.ones, dtype=dtype, device=device, shape=shape)

def zeros_like(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        data: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor first
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    return torch.zeros_like(tensor, dtype=torch_dtype, device=device)

def ones_like(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        data: The input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as the input
    """
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Convert to PyTorch tensor first
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor = tensor_ops.convert_to_tensor(data)
    return torch.ones_like(tensor, dtype=torch_dtype, device=device)

def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix.

    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Identity matrix
    """
    # Pass n and m via kwargs
    if m is None:
        m = n
    # torch.eye doesn't take shape, helper needs to handle kwargs correctly
    return _create_new_tensor(torch.eye, dtype=dtype, device=device, n=n, m=m)

def full(shape: ShapeLike, fill_value: ScalarLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value.

    Args:
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Tensor filled with the specified value
    """
    # Pass shape and fill_value via kwargs
    return _create_new_tensor(torch.full, dtype=dtype, device=device, shape=shape, fill_value=fill_value)

def full_like(data: TensorLike, fill_value: Union[int, float, bool], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value with the same shape as the input.
    
    Args:
        data: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor filled with the specified value with the same shape as data
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor_ops = TorchTensor()
    tensor_torch = tensor_ops.convert_to_tensor(data)
    
    # Handle string dtype values
    torch_dtype = None
    if dtype is not None:
        torch_dtype = TorchDType().from_dtype_str(dtype)
    
    # Create a full tensor with the same shape as the input
    return torch.full_like(tensor_torch, fill_value, dtype=torch_dtype, device=device)

def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1,
           dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive, or end if stop is None)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Tensor with evenly spaced values
    """
    # Pass start, end (stop in torch), step via kwargs
    if stop is None:
        # If only one positional arg provided, it's end
        return _create_new_tensor(torch.arange, dtype=dtype, device=device, start=start, step=step)
    else:
        return _create_new_tensor(torch.arange, dtype=dtype, device=device, start=start, end=stop, step=step)

def linspace(start: Union[int, float], stop: Union[int, float], num: int,
             dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device to place the tensor on

    Returns:
        Tensor with evenly spaced values
    """
    # Pass start, end (stop in torch), steps (num in torch) via kwargs
    return _create_new_tensor(torch.linspace, dtype=dtype, device=device, start=start, end=stop, steps=num)

def meshgrid(*arrays: TensorLike, indexing: str = 'xy') -> tuple[Tensor, ...]:
    """
    Generate multidimensional coordinate grids from 1-D coordinate arrays.

    Args:
        *arrays: 1-D arrays representing the coordinates of a grid.
        indexing: Cartesian ('xy', default) or matrix ('ij') indexing.

    Returns:
        List of PyTorch tensors representing the coordinate grids.
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    Tensor = TorchTensor()
    torch_arrays = [Tensor.convert_to_tensor(arr) for arr in arrays]
    # Pass indexing argument directly to torch.meshgrid
    # Note: Default behavior might change in future PyTorch versions.
    return torch.meshgrid(*torch_arrays, indexing=indexing)