
"""
MLX matrix linear algebra operations for ember_ml.

This module provides MLX operations.
"""

import mlx.core as mx
from typing import Tuple, Optional

# Import from tensor_ops
from typing import Tuple, Optional

import mlx.core as mx
from ember_ml.backend.mlx.types import TensorLike


def solve(a: TensorLike, b: TensorLike) -> mx.array:
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
    # Convert inputs to MLX arrays with float32 dtype
    a_array = mx.array(a, dtype=mx.float32)
    b_array = mx.array(b, dtype=mx.float32)
    from ember_ml.backend.mlx.linearalg.inverses_ops import inv # Corrected path
    # Compute the inverse of a using our custom implementation
    a_inv = inv(a_array)
    
    # Multiply the inverse by b to get the solution
    return mx.matmul(a_inv, b_array)


def lstsq(a: TensorLike, b: TensorLike, rcond: Optional[float] = None) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
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
    from ember_ml.backend.mlx.tensor import MLXTensor
    from ember_ml.backend.mlx.tensor import MLXDType
    tensor = MLXTensor()
    dtype_obj = MLXDType()
    a_array = tensor.convert(a, dtype=dtype_obj.float32)
    b_array = tensor.convert(b, dtype=dtype_obj.float32)
                                       
    # Get matrix dimensions
    m, n = a_array.shape
    
    # Ensure b is a matrix
    if b_array.ndim == 1:  # Use ndim instead of len(shape)
        b_array = b_array.reshape(m, 1)
    
    # Compute SVD of A
    from ember_ml.backend.mlx.linearalg.svd_ops import svd # Corrected path
    u, s, vh = svd(a_array)
    
    # Set default rcond if not provided
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    rcond_value = tensor.convert(1e-15, dtype=a_array.dtype)
    if rcond is None:
        max_dim = mx.max(tensor.convert(a_array.shape, dtype=a_array.dtype))
        max_s = mx.max(s)
        rcond_tensor = mx.multiply(mx.multiply(max_dim, max_s), rcond_value)
    else:
        rcond_tensor = tensor.convert(rcond, dtype=a_array.dtype)
    
    # Compute rank
    rank = mx.sum(mx.greater(s, rcond_tensor)).item()
    
    # Compute solution
    s_inv = mx.zeros_like(s)
    s_size = s.shape[0]  # Get the size of s
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    s_inv = mx.where(mx.greater(s, rcond_tensor), mx.divide(tensor.convert(1.0, dtype=s.dtype), s), mx.zeros_like(s))
    
    # Compute solution
    solution = mx.zeros((n, b_array.shape[1]), dtype=a_array.dtype)
    temp = mx.matmul(mx.transpose(u), b_array)
    temp = mx.multiply(temp, s_inv.reshape(-1, 1))
    solution = mx.matmul(mx.transpose(vh), temp)
    
    # Compute residuals
    residuals = mx.zeros((b_array.shape[1],), dtype=a_array.dtype)
    residuals = mx.sum(mx.square(mx.subtract(b_array, mx.matmul(a_array, solution))), axis=0)
    
    return solution, residuals, rank, s

