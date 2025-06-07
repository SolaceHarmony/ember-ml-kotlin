"""
PyTorch inverse operations for ember_ml.

This module provides PyTorch implementations of matrix inverse operations.
"""

import torch
from typing import Any

from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
TensorLike = Any


def inv(a: TensorLike) -> torch.Tensor:
    """
    Compute the inverse of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Inverse of the matrix
    """
    tensor = convert_to_tensor(a)
    return torch.linalg.inv(tensor)