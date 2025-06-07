"""
NumPy solver linear algebra operations for ember_ml.

This module provides NumPy operations.
"""

import numpy as np
from typing import Union, Tuple, Optional

# Import from tensor_ops
from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.tensor import NumpyDType

dtype_obj = NumpyDType()

def solve(a: TensorLike, b: TensorLike) -> np.ndarray:
    """
    Solve a linear system of equations Ax = b for x.
    
    Args:
        a: Coefficient matrix A
        b: Right-hand side vector or matrix b
        
    Returns:
        Solution to the system of equations
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    b_array = Tensor.convert_to_tensor(b)
    
    # Use NumPy's built-in solve function
    return np.linalg.solve(a_array, b_array)

def lstsq(a: TensorLike, b: TensorLike, rcond: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.int32, np.ndarray]:
    """
    Compute the least-squares solution to a linear matrix equation.
    
    Args:
        a: Coefficient matrix
        b: Dependent variable
        rcond: Cutoff for small singular values
        
    Returns:
        Tuple of (solution, residuals, rank, singular values)
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    b_array = Tensor.convert_to_tensor(b)
    
    # Use NumPy's built-in lstsq function
    return np.linalg.lstsq(a_array, b_array, rcond=rcond)