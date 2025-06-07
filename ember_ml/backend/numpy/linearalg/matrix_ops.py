"""
NumPy matrix linear algebra operations for ember_ml.

This module provides NumPy operations.
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.tensor import NumpyDType

dtype_obj = NumpyDType()

def norm(x: TensorLike,
         ord: Optional[Union[float, Literal["fro", "nuc"]]] = None,
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False) -> TensorLike:
    """
    Compute the matrix or vector norm.
    
    Args:
        x: Input matrix or vector
        ord: Order of the norm
        axis: Axis along which to compute the norm
        keepdims: Whether to keep the reduced dimensions
    
    Returns:
        Norm of the matrix or vector
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Use NumPy's built-in norm function
    return np.linalg.norm(x_array, ord=ord, axis=axis, keepdims=keepdims)

def det(a: TensorLike) -> np.ndarray:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
    
    Returns:
        Determinant of the matrix
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in determinant function
    return np.linalg.det(a_array)

def diag(x: TensorLike, k: int = 0) -> np.ndarray:
    """
    Extract a diagonal or construct a diagonal matrix.
    
    Args:
        x: Input array. If x is 2-D, return the k-th diagonal.
           If x is 1-D, return a 2-D array with x on the k-th diagonal.
        k: Diagonal offset. Use k>0 for diagonals above the main diagonal,
           and k<0 for diagonals below the main diagonal.
           
    Returns:
        The extracted diagonal or constructed diagonal matrix.
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Use NumPy's diag function directly
    return np.diag(x_array, k=k)

def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> np.ndarray:
    """
    Return specified diagonals of an array.
    
    Args:
        x: Input array
        offset: Offset of the diagonal from the main diagonal
        axis1: First axis of the 2-D sub-arrays from which the diagonals should be taken
        axis2: Second axis of the 2-D sub-arrays from which the diagonals should be taken
        
    Returns:
        Array of diagonals
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    x_array = Tensor.convert_to_tensor(x)
    
    # Check if input has at least 2 dimensions
    if x_array.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
        
    # Ensure axis1 and axis2 are different
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different")
    
    # Use NumPy's diagonal function directly
    diag_result = np.diagonal(x_array, offset=offset, axis1=axis1, axis2=axis2)
    return np.moveaxis(diag_result, -1, 0)
