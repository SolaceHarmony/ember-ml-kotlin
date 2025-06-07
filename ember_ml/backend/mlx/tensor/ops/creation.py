"""MLX tensor creation operations."""

from typing import Optional, Union

import mlx.core as mx
import numpy as np

from ember_ml.backend.mlx.types import DType, TensorLike, Shape, ShapeLike, ScalarLike


def zeros(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of zeros."""
    # Validate dtype
    # Ensure shape is a tuple
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
        
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
        
    # Create zeros array with the specified shape and dtype
    x = mx.zeros(shape, dtype=mlx_dtype)
    # Create zeros array with the specified shape and dtype
    return x

def ones(shape: 'Shape', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of ones."""
    # Ensure shape is a tuple
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
        
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
        
    # Create ones array with the specified shape and dtype
    x = mx.ones(shape, dtype=mlx_dtype)
    
    # Create ones array with the specified shape and dtype
    return x

def zeros_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of zeros with the same shape as the input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create zeros array with the same shape
    return zeros(shape, dtype, device)

def ones_like(tensor: 'TensorLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create an MLX array of ones with the same shape as the input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create ones array with the same shape
    return ones(shape, dtype, device)

def full(shape: 'ShapeLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a tensor filled with a scalar value."""
    # Handle scalar shape case
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)

    # Ensure shape is a tuple
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
        
    # Convert fill_value to a tensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    fill_value_tensor = MLXTensor().convert_to_tensor(fill_value)
    
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
        
    # Create full array with the specified shape and fill value
    x = mx.full(shape, fill_value_tensor.item(), dtype=mlx_dtype)
    
    # Create array of the specified shape filled with fill_value
    return x

def full_like(tensor: 'TensorLike', fill_value: 'ScalarLike', dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a tensor filled with fill_value with the same shape as input."""
    # Convert input to MLX array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_array = Tensor().convert_to_tensor(tensor)
    
    # Get shape of input tensor
    shape = tensor_array.shape
    
    # Create array with the same shape filled with fill_value
    return full(shape, fill_value, dtype, device)


def eye(n, m=None, dtype=None, device=None):
    """Create a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: int, number of rows.
        m: int, optional, number of columns. If None, defaults to n.
        dtype: data type of the returned tensor.
        device: device on which to place the created tensor.

    Returns:
        A 2D tensor with ones on the diagonal and zeros elsewhere.
    """
    if m is None:
        m = n
    
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
    
    # Set diagonal elements to 1 using scatter
    min_dim = min(n, m)
    indices = mx.arange(min_dim)
    from ember_ml.backend.mlx.tensor.ops import scatter
    result = scatter((indices, indices), mx.ones(min_dim, dtype=mlx_dtype), (n, m))
    
    return result

def arange(start: ScalarLike, stop: ScalarLike = None, step: int = 1,
          dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create a sequence of numbers.
    
    Args:
        start: Starting value (inclusive)
        stop: Ending value (exclusive)
        step: Step size
        dtype: Data type of the output
        device: Device to place the output on
        
    Returns:
        A tensor with values from start to stop with step size
    """

    
    # Handle single argument case
    if stop is None:
        stop = start
        start = 0
    
    # Convert tensor inputs to Python scalars if needed
    if hasattr(start, 'item'):
        start = float(start.item())
    if hasattr(stop, 'item'):
        stop = float(stop.item())

    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
        
    # Create sequence
    x = mx.arange(start, stop, step, dtype=mlx_dtype)
    # Create sequence
    return x

def linspace(start: Union[int, float], stop: Union[int, float], num: int,
            dtype: 'Optional[DType]' = None, device: Optional[str] = None) -> 'mx.array':
    """Create evenly spaced numbers over a specified interval."""
    # Validate dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    start = MLXTensor().convert_to_tensor(start)
    stop = MLXTensor().convert_to_tensor(stop)
    num = MLXTensor().convert_to_tensor(num)
    # Validate dtype
    from ember_ml.backend.mlx.tensor.ops.utility import _validate_and_get_mlx_dtype
    mlx_dtype = _validate_and_get_mlx_dtype(dtype)
    
    # Handle float64 not supported in MLX
    if str(mlx_dtype) == 'float64':
        mlx_dtype = mx.float32
        
    # Create evenly spaced sequence
    x = mx.linspace(start, stop, num, dtype=mlx_dtype)
    
    # Create evenly spaced sequence
    return x

def meshgrid(*tensors, indexing='xy'):
    """Create coordinate matrices from coordinate vectors.
    
    Args:
        *tensors: One or more tensors with the same dtype
        indexing: Cartesian ('xy', default) or matrix ('ij') indexing of output
        
    Returns:
        List of tensors with shapes determined by the broadcast shape of the
        input tensors and the indexing mode.
    """
    # Convert all inputs to MLX arrays
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor as Tensor
    tensor_arrays = [Tensor().convert_to_tensor(t) for t in tensors]
    
    # Get shapes of input tensors
    shapes = [tensor.shape for tensor in tensor_arrays]
    
    # Check that all inputs are 1D
    for i, shape in enumerate(shapes):
        if len(shape) != 1:
            raise ValueError(f"Expected 1D tensor, got shape {shape} for input {i}")
    
    # Create output tensors
    output_tensors = []
    
    if indexing == 'xy':
        # For 'xy' indexing, the first tensor corresponds to the column indices,
        # and the second tensor corresponds to the row indices
        if len(tensor_arrays) >= 2:
            # Swap the first two tensors for 'xy' indexing
            tensor_arrays[0], tensor_arrays[1] = tensor_arrays[1], tensor_arrays[0]
    
    # Create meshgrid
    for i, tensor in enumerate(tensor_arrays):
        # Create shape for broadcasting
        shape = [1] * len(tensor_arrays)
        shape[i] = len(tensor)
        
        # Reshape tensor for broadcasting
        reshaped = mx.reshape(tensor, shape)
        
        # Create full tensor by broadcasting
        full_shape = [len(t) for t in tensor_arrays]
        output = mx.broadcast_to(reshaped, full_shape)
        
        output_tensors.append(output)
    
    if indexing == 'xy' and len(tensor_arrays) >= 2:
        # Swap back the first two tensors for 'xy' indexing
        output_tensors[0], output_tensors[1] = output_tensors[1], output_tensors[0]
    
    return tuple(output_tensors)