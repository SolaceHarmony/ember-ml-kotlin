"""
PyTorch solver operations for ember_ml.

This module provides PyTorch implementations of linear system solver operations.
"""

import torch
from typing import Tuple, Optional

# Import from tensor_ops
from ember_ml.backend.torch.types import TensorLike
from ember_ml.backend.torch.tensor import TorchDType


dtype_obj = TorchDType()

def solve(a: TensorLike, b: TensorLike) -> torch.Tensor:
    """
    Solve a linear system of equations Ax = b for x using MLX backend.
    
    Args:
        a: Coefficient matrix A
        b: Right-hand side vector or matrix b
    
    Returns:
        Solution to the system of equations
    
    Notes:
        Uses custom Gauss-Jordan elimination to compute the inverse of A,
        then multiplies by b to get the solution: x = A^(-1) * b.
    """
    # Convert inputs to Torch arrays with float32 dtype
    a_array = torch.Tensor(a, dtype=torch.float32)
    b_array = torch.Tensor(b, dtype=torch.float32)
    from ember_ml.backend.torch.linearalg import inv # Simplified import path
    # Compute the inverse of a using our custom implementation
    a_inv = inv(a_array)
    
    # Multiply the inverse by b to get the solution
    return torch.matmul(a_inv, b_array)


def lstsq(a: TensorLike, b: TensorLike, rcond: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the least-squares solution to a linear matrix equation.
    
    Args:
        a: Coefficient matrix
        b: Dependent variable
        rcond: Cutoff for small singular values
    
    Returns:
        Tuple of (solution, residuals, rank, singular values)
    
    Notes:
        This is a simplified implementation using SVD.
        For large matrices or high precision requirements, consider using
        a more sophisticated algorithm.
    """
    # Convert inputs to MLX arrays with float32 dtype
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    b_array = Tensor.convert_to_tensor(b, dtype=dtype_obj.float32)
                                       
    # Get matrix dimensions
    m, n = a_array.shape
    
    # Ensure b is a matrix
    if b_array.ndim == 1:  # Use ndim instead of len(shape)
        b_array = b_array.reshape(m, 1)
    
    # Compute SVD of A
    from ember_ml.backend.torch.linearalg import svd # Simplified import path
    u, s, vh = svd(a_array)
    
    # Set default rcond if not provided
    rcond_value = 1e-15
    if rcond is None:
        max_dim = torch.max(torch.Tensor(a_array.shape))
        max_s = torch.max(s)
        rcond_tensor = torch.multiply(torch.multiply(max_dim, max_s), torch.Tensor(rcond_value))
    else:
        rcond_tensor = torch.Tensor(rcond)
    
    # Compute rank
    rank = torch.sum(torch.greater(s, rcond_tensor))
    
    # Compute solution
    s_inv = torch.zeros_like(s)
    s_size = s.shape[0]  # Get the size of s
    for i in range(s_size):
        if torch.greater(s[i], rcond_tensor).item():
            # Create a new array with the updated value
            temp = torch.zeros_like(s_inv)
            temp = temp.at[i].add(torch.divide(torch.Tensor(1.0), s[i]))
            s_inv = s_inv + temp
    
    # Compute solution
    solution = torch.zeros((n, b_array.shape[1]), dtype=a_array.dtype)
    for i in range(b_array.shape[1]):
        temp = torch.matmul(torch.transpose(u), b_array[:, i])
        temp = torch.multiply(temp, s_inv)
        solution_col = torch.matmul(torch.transpose(vh), temp)
        solution[:, i] = solution_col
            
    # Compute residuals
    residuals = torch.zeros((b_array.shape[1],), dtype=a_array.dtype)
    for i in range(b_array.shape[1]):
        residual = torch.sum(torch.square(torch.subtract(b_array[:, i], torch.matmul(a_array, solution[:, i]))))
        residuals[i] = residual
    
    return solution, residuals, rank, s

