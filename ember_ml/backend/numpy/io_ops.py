"""
NumPy implementation of I/O operations.

This module provides NumPy implementations of the ember_ml I/O operations interface.
"""

import os
import numpy as np
from typing import Union, Dict
from ember_ml.backend.numpy.types import TensorLike, PathLike
# Removed top-level import for NumpyTensor to enable lazy loading
# from ember_ml.backend.numpy.tensor.tensor import NumpyTensor

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
    
    # Convert input to NumPy array
    # Lazy load NumpyTensor inside the function
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    tensor_converter = NumpyTensor()
    obj_array = tensor_converter.convert_to_tensor(obj)
    
    # Save to file using NumPy
    np.save(filepath, obj_array, allow_pickle=allow_pickle)

def load(filepath: PathLike, allow_pickle: bool = True) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load a tensor or dictionary of tensors from a file.
    
    Args:
        filepath: Path to load the object from
        allow_pickle: Whether to allow loading objects that can't be loaded directly
        
    Returns:
        Loaded tensor or dictionary of tensors
    """
    # Load from file using NumPy
    return np.load(filepath, allow_pickle=allow_pickle)

# Removed NumpyIOOps class as it's redundant with standalone functions