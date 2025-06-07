"""
PyTorch descriptive statistics operations.

This module provides implementations of descriptive statistics using PyTorch.
"""

import torch
from typing import Union, Sequence, Optional, Any

from ember_ml.backend.torch.types import TensorLike


def median(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
          keepdims: bool = False) -> torch.Tensor:
    """
    Compute the median along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the median
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Median of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute median over all elements
        result = torch.median(x_tensor.flatten())
        return result.reshape(1) if keepdims else result
    else:
        # Compute median along the specified axis
        result = torch.median(x_tensor, dim=axis)
        return result.values.unsqueeze(axis) if keepdims else result.values

def std(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
       keepdims: bool = False, ddof: int = 0) -> torch.Tensor:
    """
    Compute the standard deviation along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the standard deviation
        keepdims: Whether to keep the reduced dimensions
        ddof: Delta degrees of freedom
        
    Returns:
        Standard deviation of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute std over all elements
        result = torch.std(x_tensor.flatten(), unbiased=(ddof == 1))
        return result.reshape(1) if keepdims else result
    else:
        # Compute std along the specified axis
        result = torch.std(x_tensor, dim=axis, unbiased=(ddof == 1))
        return result.unsqueeze(axis) if keepdims else result

def percentile(x: TensorLike, q: Union[float, torch.Tensor], 
              axis: Optional[Union[int, Sequence[int]]] = None, 
              keepdims: bool = False) -> torch.Tensor:
    """
    Compute the q-th percentile along the specified axis.
    
    Args:
        x: Input tensor
        q: Percentile(s) to compute, in range [0, 100]
        axis: Axis or axes along which to compute the percentile
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        q-th percentile of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    # Convert percentile to quantile (0-1)
    q_normalized = torch.divide(torch.tensor(q, dtype=x_tensor.dtype, device=x_tensor.device), torch.tensor(100.0, dtype=x_tensor.dtype, device=x_tensor.device))
    
    if axis is None:
        # If axis is None, compute percentile over all elements
        x_flat = x_tensor.flatten()
        sorted_x, _ = torch.sort(x_flat)
        
        # Calculate the index for the percentile
        # Convert to tensor first to avoid type errors
        idx = torch.tensor(q_normalized, device=x_flat.device) * (x_flat.size(0) - 1)
        idx_floor = torch.floor(idx).long()
        idx_ceil = torch.ceil(idx).long()
        
        # Handle edge cases
        if idx_floor == idx_ceil:
            result = sorted_x[idx_floor]
        else:
            # Linear interpolation
            weight_ceil = torch.subtract(idx, idx_floor.float())
            weight_floor = torch.subtract(torch.tensor(1.0, dtype=weight_ceil.dtype, device=weight_ceil.device), weight_ceil)
            result = torch.add(torch.multiply(weight_floor, sorted_x[idx_floor]), torch.multiply(weight_ceil, sorted_x[idx_ceil]))
        
        return result.reshape(1) if keepdims else result
    else:
        # Sort along the specified axis
        sorted_x, _ = torch.sort(x_tensor, dim=axis)
        
        # Get the size along the specified axis
        dim_size = x_tensor.size(axis)
        
        # Calculate the index for the percentile
        # Convert to tensor first to avoid type errors
        idx = torch.tensor(q_normalized, device=x_tensor.device) * (dim_size - 1)
        idx_floor = torch.floor(idx).long()
        idx_ceil = torch.ceil(idx).long()
        
        # Create indices for gathering values
        indices = [slice(None)] * x_tensor.dim()
        
        # Handle edge cases
        if idx_floor == idx_ceil:
            indices[axis] = idx_floor
            result = sorted_x[indices]
        else:
            # Linear interpolation
            weight_ceil = torch.subtract(idx, idx_floor.float())
            weight_floor = torch.subtract(torch.tensor(1.0, dtype=weight_ceil.dtype, device=weight_ceil.device), weight_ceil)
            
            indices[axis] = idx_floor
            floor_val = sorted_x[indices]
            
            indices[axis] = idx_ceil
            ceil_val = sorted_x[indices]
            
            result = torch.add(torch.multiply(weight_floor, floor_val), torch.multiply(weight_ceil, ceil_val))
        
        return result if keepdims else result.squeeze(axis)

def mean(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False, dtype: Optional[Any] = None) -> torch.Tensor:
    """
    Compute the mean along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions
        dtype: Optional data type for the output
        
    Returns:
        Mean of the tensor
    """
    # Use lazy imports to avoid circular dependencies
    # Convert input to PyTorch tensor with the validated dtype
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    tensor_instance = TorchTensor()
    x_tensor = tensor_instance.convert(x, dtype=dtype)
    
    # Get the dtype directly from the tensor
    # This is simpler and more reliable than re-validating
    torch_dtype = x_tensor.dtype
    
    if axis is None:
        # If axis is None, compute mean over all elements
        result = torch.mean(x_tensor.flatten())
        result = result.reshape(1) if keepdims else result
    else:
        # Compute mean along the specified axis
        # Calculate mean without keepdim first
        result = torch.mean(x_tensor, dim=axis, keepdim=False)
        if keepdims:
            # Construct target shape with 1s for reduced axes
            target_shape = list(x_tensor.shape)
            if isinstance(axis, Sequence):
                 for dim in axis:
                      target_shape[dim] = 1
            else: # axis is int
                 target_shape[axis] = 1
            result = result.reshape(target_shape)
    
    return result

def var(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False, ddof: int = 0) -> torch.Tensor:
    """
    Compute the variance along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        ddof: Delta degrees of freedom
        
    Returns:
        Variance of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute variance over all elements
        result = torch.var(x_tensor.flatten(), unbiased=(ddof == 1))
        return result.reshape(1) if keepdims else result
    else:
        # Compute variance along the specified axis
        # Calculate variance without keepdim first
        result = torch.var(x_tensor, dim=axis, unbiased=(ddof == 1), keepdim=False)
        if keepdims:
            # Construct target shape with 1s for reduced axes
            target_shape = list(x_tensor.shape)
            if isinstance(axis, Sequence):
                 for dim in axis:
                      target_shape[dim] = 1
            else: # axis is int
                 target_shape[axis] = 1
            return result.reshape(target_shape)
        else:
             return result

def max(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> torch.Tensor:
    """
    Compute the maximum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum value of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute max over all elements
        result = torch.max(x_tensor.flatten())
        return result.reshape(1) if keepdims else result
    else:
        # Compute max along the specified axis
        result = torch.max(x_tensor, dim=axis)
        return result.values.unsqueeze(axis) if keepdims else result.values

def min(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> torch.Tensor:
    """
    Compute the minimum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum value of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute min over all elements
        result = torch.min(x_tensor.flatten())
        return result.reshape(1) if keepdims else result
    else:
        # Compute min along the specified axis
        result = torch.min(x_tensor, dim=axis)
        return result.values.unsqueeze(axis) if keepdims else result.values

def sum(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> torch.Tensor:
    """
    Compute the sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute sum over all elements
        result = torch.sum(x_tensor.flatten())
        return result.reshape(1) if keepdims else result
    else:
        # Compute sum along the specified axis
        result = torch.sum(x_tensor, dim=axis)
        return result.unsqueeze(axis) if keepdims else result

def cumsum(x: TensorLike, axis: Optional[int] = None) -> torch.Tensor:
    """
    Compute the cumulative sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the cumulative sum
        
    Returns:
        Cumulative sum of the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute cumsum over flattened tensor
        return torch.cumsum(x_tensor.flatten(), dim=0)
    else:
        # Compute cumsum along the specified axis
        return torch.cumsum(x_tensor, dim=axis)

def argmax(x: TensorLike, axis: Optional[int] = None,
          keepdims: bool = False) -> torch.Tensor:
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the argmax
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Indices of the maximum values
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    if axis is None:
        # If axis is None, compute argmax over flattened tensor
        result = torch.argmax(x_tensor.flatten())
        return result.reshape(1) if keepdims else result
    else:
        # Compute argmax along the specified axis
        result = torch.argmax(x_tensor, dim=axis)
        return result.unsqueeze(axis) if keepdims else result

def sort(x: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    # Sort along the specified axis
    result, _ = torch.sort(x_tensor, dim=axis, descending=descending)
    return result

def argsort(x: TensorLike, axis: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Returns the indices that would sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    # Convert input to PyTorch tensor
    from ember_ml.backend.torch.tensor import TorchTensor
    tensor = TorchTensor()
    x_tensor = tensor.convert(x)
    
    # Get argsort along the specified axis
    _, indices = torch.sort(x_tensor, dim=axis, descending=descending)
    return indices


def gaussian(input_value: TensorLike, mu: TensorLike = 0.0, sigma: TensorLike = 1.0) -> torch.Tensor:
    """
    Compute the value of the Gaussian (normal distribution) function.

    Formula: (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma)^2)

    Args:
        input_value: The input value(s) (x).
        mu: The mean (center) of the distribution. Defaults to 0.0.
        sigma: The standard deviation (spread) of the distribution. Defaults to 1.0.

    Returns:
        The Gaussian function evaluated at the input value(s).
    """
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.torch.math_ops import pi
    tensor = TorchTensor()
    x_tensor = tensor.convert(input_value)
    mu_tensor = tensor.convert(mu)
    sigma_tensor = tensor.convert(sigma)
    half = tensor.convert(0.5)
    two = tensor.convert(2.0)
    pi_tensor = tensor.convert(pi)

    exponent = torch.multiply(
        torch.negative(half),
        torch.square(torch.divide(torch.subtract(x_tensor, mu_tensor), sigma_tensor))
    )
    denominator = torch.multiply(
        sigma_tensor,
        torch.multiply(torch.sqrt(two), torch.sqrt(pi_tensor))
    )
    return torch.divide(torch.exp(exponent), denominator)
