"""
PyTorch inverse operations for ember_ml.

This module provides PyTorch implementations of matrix inverse operations.
"""

import torch
from typing import Any # Added Any
# Import from tensor_ops
from ember_ml.backend.torch.tensor import TorchDType
from ember_ml.backend.torch.types import TensorLike

dtype_obj = TorchDType()


def inv(A: TensorLike) -> Any: # Changed torch.Tensor to Any
    """
    Compute the inverse of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Inverse of the matrix
    """
       # Convert input to Torch array with float32 dtype
    A = torch.tensor(A, dtype=torch.float32)
    
    # Get matrix dimensions
    n = A.shape[0]
    assert A.shape[1] == n, "Matrix must be square"
    
    # Create augmented matrix [A|I]
    I = torch.eye(n, dtype=A.dtype)
    aug = torch.cat([A, I], dim=1)
    
    # Create a copy of the augmented matrix that we can modify
    aug_copy = aug.clone() # Use clone() for efficiency and clearer intent
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        pivot = aug_copy[i, i]
        
        # Scale pivot row
        pivot_row = torch.divide(aug_copy[i], pivot)
        
        # Create a new augmented matrix with the updated row
        rows = []
        for j in range(n):
            if j == i:
                rows.append(pivot_row)
            else:
                # Eliminate from other rows
                factor = aug_copy[j, i]
                rows.append(torch.subtract(aug_copy[j], torch.multiply(factor, pivot_row)))
        
        # Reconstruct the augmented matrix
        aug_copy = torch.stack(rows)
    
    # Extract inverse from augmented matrix
    inv_A = aug_copy[:, n:]

    return inv_A