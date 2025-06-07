"""
NumPy descriptive statistics operations.

This module provides implementations of descriptive statistics using NumPy.
"""

import numpy as np
from typing import Union, Sequence, Optional,Any

from ember_ml.backend.numpy.types import TensorLike


def median(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
          keepdims: bool = False) -> np.ndarray:
    """
    Compute the median along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the median
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Median of the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute median
    return np.median(x_array, axis=axis, keepdims=keepdims)

def std(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, 
       keepdims: bool = False, ddof: int = 0) -> np.ndarray:
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
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute standard deviation
    return np.std(x_array, axis=axis, keepdims=keepdims, ddof=ddof)

def percentile(x: TensorLike, q: Union[float, np.ndarray], 
              axis: Optional[Union[int, Sequence[int]]] = None, 
              keepdims: bool = False) -> np.ndarray:
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
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute percentile
    return np.percentile(x_array, q, axis=axis, keepdims=keepdims)

def mean(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False, dtype: Optional[Any] = None) -> np.ndarray:
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
    # Convert input to NumPy array with the specified dtype
    from ember_ml.backend.numpy.tensor import NumpyTensor
    
    # Use the new convert method to handle dtype validation and conversion
    tensor = NumpyTensor()
    x_array = tensor.convert(x, dtype=dtype)
    
    # Compute mean
    # The dtype is already applied during conversion, so we don't need to pass it again
    result = np.mean(x_array, axis=axis, keepdims=keepdims)
    
    return result

def var(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False, ddof: int = 0) -> np.ndarray:
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
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute variance
    return np.var(x_array, axis=axis, keepdims=keepdims, ddof=ddof)

def max(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> np.ndarray:
    """
    Compute the maximum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Maximum value of the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute maximum
    return np.max(x_array, axis=axis, keepdims=keepdims)

def min(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> np.ndarray:
    """
    Compute the minimum value along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Minimum value of the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute minimum
    return np.min(x_array, axis=axis, keepdims=keepdims)

def sum(x: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None,
       keepdims: bool = False) -> np.ndarray:
    """
    Compute the sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Sum of the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute sum
    return np.sum(x_array, axis=axis, keepdims=keepdims)

def cumsum(x: TensorLike, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute the cumulative sum along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the cumulative sum
        
    Returns:
        Cumulative sum of the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute cumulative sum
    return np.cumsum(x_array, axis=axis)

def argmax(x: TensorLike, axis: Optional[int] = None,
          keepdims: bool = False) -> np.ndarray:
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to compute the argmax
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Indices of the maximum values
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Compute argmax
    result = np.argmax(x_array, axis=axis)
    
    # Handle keepdims
    if keepdims and axis is not None:
        result = np.expand_dims(result, axis=axis)
    
    return np.array(result) # Ensure return type is always ndarray

def sort(x: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Sort based on direction
    if descending:
        return -np.sort(-x_array, axis=axis)
    else:
        return np.sort(x_array, axis=axis)

def argsort(x: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Returns the indices that would sort a tensor along the specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    x_array = tensor.convert(x)
    
    # Get argsort based on direction
    if descending:
        return np.argsort(-x_array, axis=axis)
    else:
        return np.argsort(x_array, axis=axis)


def gaussian(input_value: TensorLike, mu: TensorLike = 0.0, sigma: TensorLike = 1.0) -> np.ndarray:
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
    # Convert inputs to NumPy arrays
    from ember_ml.backend.numpy.tensor import NumpyTensor
    # Ensure all inputs are tensors
    tensor = NumpyTensor()
    x_tensor = tensor.convert(input_value)
    mu_tensor = tensor.convert(mu)
    sigma_tensor = tensor.convert(sigma)
    # Use np functions for arithmetic
    half = np.array(0.5, dtype=x_tensor.dtype) # Match dtype
    two = np.array(2.0, dtype=x_tensor.dtype)
    diff = np.subtract(x_tensor, mu_tensor)
    scaled_diff = np.divide(diff, sigma_tensor)
    exponent = np.multiply(np.negative(half), np.square(scaled_diff))
    from ember_ml.backend.numpy.math_ops import pi as math_pi
    sqrt_two_pi = np.sqrt(np.multiply(two, math_pi)) # Use np.pi
    denominator = np.multiply(sigma_tensor, sqrt_two_pi)
    return np.divide(np.exp(exponent), denominator)
