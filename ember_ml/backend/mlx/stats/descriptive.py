"""
MLX descriptive statistics operations.

This module provides implementations of descriptive statistics using MLX.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Any

from ember_ml.backend.mlx.types import TensorLike

from ember_ml.backend.mlx.math_ops import pi as math_pi

def median(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
          keepdims: bool = False) -> mx.array:
    """
    Compute the median along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the median
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Median of the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Sort values along the specified axis
    sorted_x = mx.sort(x_array, axis=axis)
    
    # Get the middle index
    if axis is None:
        # Flatten and compute median
        shape = x_array.shape
        size = 1
        for dim in shape:
             size *= dim # Calculate product using Python
        flat_sorted = mx.reshape(sorted_x, (size,))
        
        # Calculate middle index
        mid_idx = size // 2
        
        # For even-length arrays, take the average of the two middle values
        if size % 2 == 0:
            median_val = mx.divide(mx.add(flat_sorted[mid_idx-1], flat_sorted[mid_idx]), mx.array(2.0))
        else:
            median_val = flat_sorted[mid_idx]
            
        # Handle keepdims
        if keepdims:
            return mx.expand_dims(median_val)
        else:
            return median_val
    else:
        # Get size along the specified axis
        size = x_array.shape[axis]
        
        # Calculate middle index
        mid_idx = size // 2
        
        # Create slice objects for indexing
        indices = [slice(None)] * len(x_array.shape)
        
        # For even-length arrays, take the average of the two middle values
        if size % 2 == 0:
            # Get the two middle values
            indices[axis] = mid_idx - 1
            mid_val1 = sorted_x[tuple(indices)]
            
            indices[axis] = mid_idx
            mid_val2 = sorted_x[tuple(indices)]
            
            # Average the two middle values
            median_val = mx.divide(mx.add(mid_val1, mid_val2), mx.array(2.0))
        else:
            # Get the middle value
            indices[axis] = mid_idx
            median_val = sorted_x[tuple(indices)]
        
        # Handle keepdims
        if keepdims:
            return median_val if axis is None else mx.expand_dims(median_val, axis=axis)
        else:
            return median_val

def std(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
       keepdims: bool = False, ddof: int = 0) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Compute mean
    mean = mx.mean(x_array, axis=axis, keepdims=True)
    
    # Compute squared deviations
    squared_diff = mx.square(x_array - mean)
    
    # Compute variance
    if axis is None:
        # Flatten and compute
        shape = x_array.shape
        size = 1
        for dim in shape:
             size *= dim # Calculate product using Python
        n = size - ddof
        variance = mx.divide(mx.sum(squared_diff), mx.array(n, dtype=squared_diff.dtype))
    else:
        # Compute along the specified axis
        n = x_array.shape[axis] - ddof
        variance = mx.divide(mx.sum(squared_diff, axis=axis, keepdims=keepdims), mx.array(n, dtype=squared_diff.dtype))
    
    # Return square root of variance
    return mx.sqrt(variance)

def percentile(x: TensorLike, q: Union[float, Any], 
              axis: Optional[Union[int, Sequence[int]]] = None, 
              keepdims: bool = False) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Convert percentile to fraction
    q_frac = mx.divide(mx.array(q, dtype=x_array.dtype), mx.array(100.0, dtype=x_array.dtype))
    
    # Sort the array
    sorted_x = mx.sort(x_array, axis=axis)
    
    if axis is None:
        # Flatten and compute
        shape = x_array.shape
        size = 1
        for dim in shape:
             size *= dim # Calculate product using Python
        flat_sorted = mx.reshape(sorted_x, (size,))
        
        # Calculate index position
        idx = q_frac * (size - 1)
        idx_floor = int(mx.floor(idx))
        idx_ceil = int(mx.ceil(idx))
        
        # Handle edge cases
        if idx_floor == idx_ceil:
            result = flat_sorted[idx_floor]
        else:
            # Linear interpolation
            weight_ceil = idx - idx_floor
            weight_floor = 1.0 - weight_ceil
            result = weight_floor * flat_sorted[idx_floor] + weight_ceil * flat_sorted[idx_ceil]
            
        # Handle keepdims
        if keepdims:
            return mx.expand_dims(result)
        else:
            return result
    else:
        # Get size along the specified axis
        size = x_array.shape[axis]
        
        # Calculate index position
        idx = q_frac * (size - 1)
        idx_floor = int(mx.floor(idx))
        idx_ceil = int(mx.ceil(idx))
        
        # Create slice objects for indexing
        floor_indices = [slice(None)] * len(x_array.shape)
        ceil_indices = [slice(None)] * len(x_array.shape)
        floor_indices[axis] = idx_floor
        ceil_indices[axis] = idx_ceil
        
        # Handle edge cases
        if idx_floor == idx_ceil:
            result = sorted_x[tuple(floor_indices)]
        else:
            # Linear interpolation
            weight_ceil = idx - idx_floor
            weight_floor = 1.0 - weight_ceil
            floor_val = sorted_x[tuple(floor_indices)]
            ceil_val = sorted_x[tuple(ceil_indices)]
            result = weight_floor * floor_val + weight_ceil * ceil_val
        
        # Handle keepdims
        if keepdims:
            if axis is not None:
                return mx.expand_dims(result, axis=axis)
            else:
                return mx.expand_dims(result)
        else:
            return result

def mean(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False, dtype: Optional[Any] = None) -> mx.array:
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
    # Convert input to MLX array with the validated dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x, dtype=dtype)
    
    # Get the dtype directly from the tensor
    # This is simpler and more reliable than re-validating
    mlx_dtype = x_array.dtype
    
    # Compute mean
    result = mx.mean(x_array, axis=axis, keepdims=keepdims)
    
    # Cast to the specified dtype if the result dtype is different
    if result.dtype != mlx_dtype:
        result = result.astype(mlx_dtype)
    
    return result

def var(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False, ddof: int = 0) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Compute mean
    mean = mx.mean(x_array, axis=axis, keepdims=True)
    
    # Compute squared deviations
    squared_diff = mx.square(x_array - mean)
    
    # Compute variance
    if axis is None:
        # Flatten and compute
        shape = x_array.shape
        size = 1
        for dim in shape:
             size *= dim # Calculate product using Python
        n = size - ddof
        variance = mx.divide(mx.sum(squared_diff), mx.array(n, dtype=squared_diff.dtype))
        if keepdims:
            return mx.expand_dims(variance)
        return variance
    else:
        # Compute along the specified axis
        n = x_array.shape[axis] - ddof
        return mx.divide(mx.sum(squared_diff, axis=axis, keepdims=keepdims), mx.array(n, dtype=squared_diff.dtype))

def max(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> mx.array:
    """
    Compute the maximum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum value of the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Compute max
    return mx.max(x_array, axis=axis, keepdims=keepdims)

def min(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> mx.array:
    """
    Compute the minimum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum value of the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Compute min
    return mx.min(x_array, axis=axis, keepdims=keepdims)

def sum(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> mx.array:
    """
    Compute the sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Compute sum
    return mx.sum(x_array, axis=axis, keepdims=keepdims)

def cumsum(x: TensorLike, axis: Optional[int] = None) -> mx.array:
    """
    Compute the cumulative sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the cumulative sum
        
    Returns:
        Cumulative sum of the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Default axis is 0 if None is provided
    if axis is None:
        axis = 0
    
    # Compute cumsum
    return mx.cumsum(x_array, axis=axis)

def argmax(x: TensorLike, axis: Optional[int] = None,
          keepdims: bool = False) -> mx.array:
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the argmax
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Indices of the maximum values
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # If axis is None, flatten the array first
    if axis is None:
        flat_x = mx.reshape(x_array, (-1,))
        result = mx.argmax(flat_x)
        if keepdims:
            return mx.expand_dims(result)
        return result
    
    # Compute argmax along the specified axis
    result = mx.argmax(x_array, axis=axis)
    
    # Handle keepdims
    if keepdims:
        return mx.expand_dims(result, axis=axis)
    return result

def sort(x: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Sort the array
    sorted_x = mx.sort(x_array, axis=axis)
    
    # Reverse if descending
    if descending:
        # Create slice objects for indexing
        indices = [slice(None)] * len(x_array.shape)
        indices[axis] = slice(None, None, -1)
        return sorted_x[tuple(indices)]
    
    return sorted_x

def argsort(x: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Returns the indices that would sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_array = tensor.convert(x)
    
    # Get argsort indices
    indices = mx.argsort(x_array, axis=axis)
    
    # Reverse if descending
    if descending:
        # Create slice objects for indexing
        idx_slice = [slice(None)] * len(x_array.shape)
        idx_slice[axis] = slice(None, None, -1)
        return indices[tuple(idx_slice)]
    
    return indices


def gaussian(input_value: TensorLike, mu: TensorLike = 0.0, sigma: TensorLike = 1.0) -> mx.array:
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
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_tensor = tensor.convert(input_value)
    mu_tensor = tensor.convert(mu)
    sigma_tensor = tensor.convert(sigma)
    half = tensor.convert(0.5)
    two = tensor.convert(2.0)
    pi_tensor = tensor.convert(math_pi)

    exponent = mx.multiply(
        mx.negative(half),
        mx.square(mx.divide(mx.subtract(x_tensor, mu_tensor), sigma_tensor))
    )
    denominator = mx.multiply(
        sigma_tensor,
        mx.multiply(mx.sqrt(two), mx.sqrt(pi_tensor))
    )
    return mx.divide(mx.exp(exponent), denominator)




