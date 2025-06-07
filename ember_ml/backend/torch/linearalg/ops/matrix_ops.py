"""
PyTorch matrix operations for ember_ml.

This module provides PyTorch implementations of matrix operations.
"""

import torch
from typing import Optional, Union, List, Tuple, Any

from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
TensorLike = Any


def norm(x: TensorLike, ord=None, axis=None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the matrix or vector norm.
    
    Args:
        x: Input tensor
        ord: Order of the norm (see torch.linalg.norm for details)
        axis: Axis along which to compute the norm
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Norm of the tensor
    """
    tensor = convert_to_tensor(x)
    
    # Handle axis parameter
    if axis is None:
        # Use default behavior
        return torch.linalg.norm(tensor, ord=ord, keepdim=keepdims)
    
    # Convert to tuple if it's a list
    if isinstance(axis, list):
        axis = tuple(axis)
    
    return torch.linalg.norm(tensor, ord=ord, dim=axis, keepdim=keepdims)


def det(a: TensorLike) -> torch.Tensor:
    """
    Compute the determinant of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Determinant of the matrix
    """
    tensor = convert_to_tensor(a)
    return torch.linalg.det(tensor)


def diag(x: TensorLike, k: int = 0) -> torch.Tensor:
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
    tensor = convert_to_tensor(x)
    return torch.diag(tensor, diagonal=k)


def diagonal(x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
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
    tensor = convert_to_tensor(x)
    return torch.diagonal(tensor, offset=offset, dim1=axis1, dim2=axis2)