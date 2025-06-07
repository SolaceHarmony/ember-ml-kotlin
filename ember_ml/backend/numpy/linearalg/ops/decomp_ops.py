"""
NumPy decomposition linear algebra operations for ember_ml.

This module provides NumPy operations.
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal

# Import from tensor_ops
from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.tensor import NumpyDType

dtype_obj = NumpyDType()

def cholesky(a: TensorLike) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a positive definite matrix.
    
    Args:
        a: Input positive definite matrix
        
    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in cholesky function
    return np.linalg.cholesky(a_array)

def svd(a: TensorLike, full_matrices: bool = True, compute_uv: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the singular value decomposition of a matrix.
    
    Args:
        a: Input matrix
        full_matrices: If True, return full U and Vh matrices
        compute_uv: If True, compute U and Vh matrices
        
    Returns:
        If compute_uv is True, returns (U, S, Vh), otherwise returns S
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in svd function
    return np.linalg.svd(a_array, full_matrices=full_matrices, compute_uv=compute_uv)

def eig(a: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in eig function
    return np.linalg.eig(a_array)

def eigvals(a: TensorLike) -> np.ndarray:
    """
    Compute the eigenvalues of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Eigenvalues of the matrix
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in eigvals function
    return np.linalg.eigvals(a_array)

def qr(a: TensorLike, mode: str = 'reduced') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the QR decomposition of a matrix.
    
    Args:
        a: Input matrix
        mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
        
    Returns:
        Tuple of (Q, R) matrices
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in qr function
    return np.linalg.qr(a_array, mode=mode)