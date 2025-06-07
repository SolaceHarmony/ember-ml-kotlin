"""
PyTorch implementation of I/O operations.

This module provides PyTorch implementations of the ember_ml I/O operations interface.
"""

import os
import torch
from typing import Union, Dict
from ember_ml.backend.torch.types import TensorLike, PathLike, TorchArray
from ember_ml.backend.torch.tensor.tensor import TorchTensor

def save(filepath: PathLike, obj: TensorLike, allow_pickle: bool = True) -> None:
    """
    Save a tensor or dictionary of tensors to a file.
    
    Args:
        filepath: Path to save the object to
        obj: Tensor or dictionary of tensors to save
        allow_pickle: Whether to allow saving objects that can't be saved directly
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert input to PyTorch tensor
    tensor_converter = TorchTensor()
    obj_tensor = tensor_converter.convert_to_tensor(obj)
    
    # Save to file using PyTorch
    torch.save(obj_tensor, filepath)

def load(filepath: PathLike, allow_pickle: bool = True) -> Union[TorchArray, Dict[str, TorchArray]]:
    """
    Load a tensor or dictionary of tensors from a file.
    
    Args:
        filepath: Path to load the object from
        allow_pickle: Whether to allow loading objects that can't be loaded directly
        
    Returns:
        Loaded tensor or dictionary of tensors
    """
    # Load with map_location='cpu' to ensure compatibility
    return torch.load(filepath, map_location='cpu')

# Removed TorchIOOps class as it's redundant with standalone functions