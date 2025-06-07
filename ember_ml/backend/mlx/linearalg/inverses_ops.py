"""
MLX solver operations for ember_ml.

This module provides MLX implementations of solver operations.
"""

import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike

dtype_obj = MLXDType()


def inv(A: TensorLike) -> mx.array:
    """
    Inverts a square matrix using Gauss-Jordan elimination.
    
    Args:
        A: Square matrix to invert
        
    Returns:
        Inverse of matrix A
    """
    # Convert input to MLX array with float32 dtype
    A = mx.array(A, dtype=mx.float32)
    
    # Get matrix dimensions
    n = A.shape[0]
    assert A.shape[1] == n, "Matrix must be square"
    
    # Create augmented matrix [A|I]
    I = mx.eye(n, dtype=A.dtype)
    aug = mx.concatenate([A, I], axis=1)
    
    # Create a copy of the augmented matrix that we can modify
    aug_copy = mx.array(aug)
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        pivot = aug_copy[i, i]
        
        # Scale pivot row
        pivot_row = mx.divide(aug_copy[i], pivot)
        
        # Create a new augmented matrix with the updated row
        rows = []
        for j in range(n):
            if j == i:
                rows.append(pivot_row)
            else:
                # Eliminate from other rows
                factor = aug_copy[j, i]
                rows.append(mx.subtract(aug_copy[j], mx.multiply(factor, pivot_row)))
        
        # Reconstruct the augmented matrix
        aug_copy = mx.stack(rows)
    
    # Extract inverse from augmented matrix
    inv_A = aug_copy[:, n:]
    
    return inv_A