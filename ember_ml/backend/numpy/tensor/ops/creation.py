"""NumPy tensor creation operations."""

from typing import Any, List, Literal, Optional, Union

import numpy as np

from ember_ml.backend.numpy.types import DType, TensorLike, Shape, ShapeLike, ScalarLike
# Default dtypes will be handled by _create_new_tensor
from ember_ml.backend.numpy.tensor.ops.utility import _create_new_tensor # Import the helper

def zeros(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy array of zeros using the helper."""
    # Pass shape via kwargs
    return _create_new_tensor(np.zeros, dtype=dtype, device=device, shape=shape)

def ones(shape: Shape, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy array of ones using the helper."""
    # Pass shape via kwargs
    return _create_new_tensor(np.ones, dtype=dtype, device=device, shape=shape)

def zeros_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'np.ndarray':
    """Create a NumPy array of zeros with the same shape as the input."""
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create zeros array with the same shape
    return zeros(shape, dtype, device)

def ones_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'np.ndarray':
    """Create a NumPy array of ones with the same shape as the input."""
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create ones array with the same shape
    return ones(shape, dtype, device)

def full(shape: ShapeLike, fill_value: ScalarLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create a NumPy array filled with a scalar value using the helper."""
    # Pass shape and fill_value via kwargs
    return _create_new_tensor(np.full, dtype=dtype, device=device, shape=shape, fill_value=fill_value)

def full_like(tensor: 'TensorLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'np.ndarray':
    """Create a tensor filled with fill_value with the same shape as input."""
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create array with the same shape filled with fill_value
    return full(shape, fill_value, dtype, device)

def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create an identity matrix using the helper."""
    # Pass N and M via kwargs
    if m is None:
        m = n
    # np.eye defaults to float64 if dtype is None, _create_new_tensor handles default logic
    return _create_new_tensor(np.eye, dtype=dtype, device=device, N=n, M=m)

def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1,
           dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create a sequence of numbers using the helper."""
    # np.arange infers dtype if None, _create_new_tensor handles default logic
    # Pass start, stop, step via kwargs
    if stop is None:
        # If only one positional arg provided to arange wrapper, it's stop
        return _create_new_tensor(np.arange, dtype=dtype, device=device, stop=start, step=step)
    else:
        return _create_new_tensor(np.arange, dtype=dtype, device=device, start=start, stop=stop, step=step)

def linspace(start: Union[int, float], stop: Union[int, float], num: int,
             dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """Create evenly spaced numbers over a specified interval using the helper."""
    # np.linspace infers dtype if None, _create_new_tensor handles default logic
    # Pass start, stop, num via kwargs
    return _create_new_tensor(np.linspace, dtype=dtype, device=device, start=start, stop=stop, num=num)

def meshgrid(*arrays: TensorLike, indexing: Literal['xy', 'ij'] = 'xy') -> List[np.ndarray]:
    """
    Generate multidimensional coordinate grids from 1-D coordinate arrays.

    Args:
        *arrays: 1-D arrays representing the coordinates of a grid.
        indexing: Cartesian ('xy', default) or matrix ('ij') indexing.

    Returns:
        List of NumPy arrays representing the coordinate grids.
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor as Tensor
    np_arrays = [Tensor().convert_to_tensor(arr) for arr in arrays]
    # np.meshgrid defaults to 'ij' if sparse is False, but takes 'xy'/'ij' indexing argument
    return np.meshgrid(*np_arrays, indexing=indexing)