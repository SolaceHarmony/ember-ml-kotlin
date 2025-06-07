
"""
MLX matrix linear algebra operations for ember_ml.

This module provides MLX operations.
"""

import mlx.core as mx
from typing import Tuple, Optional

# Import from tensor_ops
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXDType


dtype_obj = MLXDType()

def solve(a: TensorLike, 
          b: TensorLike) -> mx.array:
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
    from ember_ml.backend.mlx.linearalg.ops.inverses_ops import inv
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
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    b_array = Tensor.convert_to_tensor(b, dtype=dtype_obj.float32)
                                       
    # Get matrix dimensions
    m, n = a_array.shape
    
    # Ensure b is a matrix
    if b_array.ndim == 1:  # Use ndim instead of len(shape)
        b_array = b_array.reshape(m, 1)
    
    # Compute SVD of A
    from ember_ml.backend.mlx.linearalg.ops.decomp_ops import svd
    u, s, vh = svd(a_array)
    
    # Set default rcond if not provided
    rcond_value = 1e-15
    if rcond is None:
        max_dim = mx.max(mx.array(a_array.shape))
        max_s = mx.max(s)
        rcond_tensor = mx.multiply(mx.multiply(max_dim, max_s), mx.array(rcond_value))
    else:
        rcond_tensor = mx.array(rcond)
    
    # Compute rank
    rank = mx.sum(mx.greater(s, rcond_tensor))
    
    # Compute solution
    s_inv = mx.zeros_like(s)
    s_size = s.shape[0]  # Get the size of s
    for i in range(s_size):
        if mx.greater(s[i], rcond_tensor).item():
            # Create a new array with the updated value
            temp = mx.zeros_like(s_inv)
            temp = temp.at[i].add(mx.divide(mx.array(1.0), s[i]))
            s_inv = s_inv + temp
    
    # Compute solution
    solution = mx.zeros((n, b_array.shape[1]), dtype=a_array.dtype)
    for i in range(b_array.shape[1]):
        temp = mx.matmul(mx.transpose(u), b_array[:, i])
        temp = mx.multiply(temp, s_inv)
        solution_col = mx.matmul(mx.transpose(vh), temp)
        solution[:, i] = solution_col
    
    # Compute residuals
    residuals = mx.zeros((b_array.shape[1],), dtype=a_array.dtype)
    for i in range(b_array.shape[1]):
        residual = mx.sum(mx.square(mx.subtract(b_array[:, i], mx.matmul(a_array, solution[:, i]))))
        residuals[i] = residual
    
    return solution, residuals, rank, s

