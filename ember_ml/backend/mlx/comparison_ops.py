"""
MLX implementation of comparison operations.

This module provides MLX implementations of comparison operations.
"""

import mlx.core as mx
from typing import Any

# Import from tensor_ops
from ember_ml.backend.mlx.types import TensorLike


def equal(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if two MLX arrays are equal element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements are equal
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.equal(MLXTensor().convert_to_tensor(x), Tensor.convert_to_tensor(y))

def not_equal(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if two MLX arrays are not equal element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements are not equal
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.not_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def less(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if elements of the first MLX array are less than the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are less than y
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.less(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def less_equal(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if elements of the first MLX array are less than or equal to the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are less than or equal to y
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.less_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def greater(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if elements of the first MLX array are greater than the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are greater than y
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.greater(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def greater_equal(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Check if elements of the first MLX array are greater than or equal to the second element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where elements of x are greater than or equal to y
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.greater_equal(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def logical_and(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the logical AND of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where both x and y are True
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.logical_and(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def logical_or(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the logical OR of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where either x or y is True
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.logical_or(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))

def logical_not(x: TensorLike) -> mx.array:
    """
    Compute the logical NOT of an MLX array element-wise.
    
    Args:
        x: Input array
        
    Returns:
        Boolean array with True where x is False
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    return mx.logical_not(Tensor.convert_to_tensor(x))

def logical_xor(x: TensorLike, y: TensorLike) -> mx.array:
    """
    Compute the logical XOR of two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Boolean array with True where exactly one of x or y is True
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    return mx.bitwise_xor(x_tensor, y_tensor)


def allclose(x: TensorLike, y: TensorLike, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    """
    Check if all elements of two MLX arrays are close within a tolerance.
    
    Args:
        x: First array
        y: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean indicating if all elements are close
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    return mx.allclose(x_tensor, y_tensor, rtol=rtol, atol=atol)


def isclose(x: TensorLike, y: TensorLike, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
    """
    Check if elements of two MLX arrays are close within a tolerance element-wise.
    
    Args:
        x: First array
        y: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Boolean array with True where elements are close
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    # Implement isclose using the formula: |x - y| <= atol + rtol * |y|
    abs_diff = mx.abs(mx.subtract(x_tensor, y_tensor))
    tolerance = mx.add(atol, mx.multiply(rtol, mx.abs(y_tensor)))
    return mx.less_equal(abs_diff, tolerance)

def all(x: TensorLike, axis: Any = None, keepdims: bool = False) -> mx.array:
    """
    Check if all elements in a tensor are True.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.
        keepdims: Keep reduced axes as singleton dimensions, defaults to False.
            
    Returns:
        Boolean tensor with True if all elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    return mx.all(x_tensor, axis=axis, keepdims=keepdims)


def any(x: TensorLike, axis: Any = None, keepdims: bool = False) -> mx.array:
    """
    Check if any elements in a tensor are True.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to perform the reduction.
            If None, reduce over all dimensions.
        keepdims: Keep reduced axes as singleton dimensions, defaults to False.
            
    Returns:
        Boolean tensor with True if any elements are True, False otherwise.
        If axis is specified, the result is a tensor with the specified
        axes reduced.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    return mx.any(x_tensor, axis=axis, keepdims=keepdims)



def where(condition: TensorLike, x: TensorLike, y: TensorLike) -> mx.array:
    """
    Return elements chosen from x or y depending on condition.
    
    Args:
        condition: Boolean array
        x: Array with values to use where condition is True
        y: Array with values to use where condition is False
        
    Returns:
        Array with values from x where condition is True, and values from y elsewhere
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    condition_tensor = Tensor.convert_to_tensor(condition)
    x_tensor = Tensor.convert_to_tensor(x)
    y_tensor = Tensor.convert_to_tensor(y)
    return mx.where(condition_tensor, x_tensor, y_tensor)


def isnan(x: TensorLike) -> mx.array:
    """
    Test element-wise for NaN values.
    
    Args:
        x: Input tensor
        
    Returns:
        Boolean tensor with True where x is NaN, False otherwise
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    x_tensor = Tensor.convert_to_tensor(x)
    return mx.isnan(x_tensor)


# Removed MLXComparisonOps class as it's redundant with standalone functions