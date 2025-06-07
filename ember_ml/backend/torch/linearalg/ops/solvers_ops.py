"""
PyTorch solver operations for ember_ml.

This module provides PyTorch implementations of linear system solver operations.
"""

import torch
from typing import Optional, Union, List, Tuple, Any

from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
TensorLike = Any


def solve(a: TensorLike, b: TensorLike) -> torch.Tensor:
    """
    Solve a linear system of equations Ax = b for x.
    
    Args:
        a: Coefficient matrix
        b: Ordinate or "dependent variable" values
        
    Returns:
        Solution to the system Ax = b
    """
    tensor_a = convert_to_tensor(a)
    tensor_b = convert_to_tensor(b)
    
    return torch.linalg.solve(tensor_a, tensor_b)


def lstsq(a: TensorLike, b: TensorLike, rcond: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the least-squares solution to a linear matrix equation.
    
    Args:
        a: Coefficient matrix
        b: Ordinate or "dependent variable" values
        rcond: Cut-off ratio for small singular values
        
    Returns:
        Tuple containing:
        - Solution to the least-squares problem
        - Sum of squared residuals
        - Rank of matrix a
        - Singular values of a
    """
    tensor_a = convert_to_tensor(a)
    tensor_b = convert_to_tensor(b)
    
    # PyTorch's lstsq has a different signature than NumPy's
    # It returns (solution, QR decomposition) rather than (solution, residuals, rank, singular values)
    # We'll compute these values manually
    
    # Compute the solution using torch.linalg.lstsq
    solution = torch.linalg.lstsq(tensor_a, tensor_b, rcond=rcond).solution
    
    # Compute residuals
    residuals = torch.linalg.norm(tensor_b - torch.matmul(tensor_a, solution), dim=0) ** 2
    
    # Compute rank and singular values
    U, S, Vh = torch.linalg.svd(tensor_a)
    rank = torch.sum(S > (rcond or 1e-15) * S[0])
    
    return solution, residuals, rank, S