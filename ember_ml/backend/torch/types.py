"""
Type definitions for PyTorch tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union, Literal,
    TYPE_CHECKING
)
import os

import torch

# Basic type aliases that don't require imports
Numeric = Union[int, float]
OrdLike = Optional[Union[int, str]]
Device = Optional[str]
PathLike = Union[str, os.PathLike[str]]
Shape = Sequence[int]
ShapeType = Union[int, Tuple[int, ...], List[int]]
ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]
DimSize = Union[int, 'torch.Tensor']
Axis = Optional[Union[int, Sequence[int]]]
IndexType = Union[int, Sequence[int], 'torch.Tensor']
Indices = Union[Sequence[int], 'torch.Tensor']

# PyTorch specific
TorchArray = Any # Changed from torch.Tensor due to AttributeError
DTypeClass = Any # Changed from torch.dtype due to AttributeError

# Precision related
default_int = Any # Changed from torch.int32 due to AttributeError
default_float = Any # Changed from torch.float32 due to AttributeError
default_bool = Any # Changed from torch.bool due to AttributeError
TensorLike = Any
ScalarLike = Any

# Default type for dtype
DType = Any

# Conditional type definitions
if TYPE_CHECKING == True:
    # These imports are for type checking only
    # Must be done inside TYPE_CHECKING block to avoid circular imports
    from typing import TypeVar
    from ember_ml.nn.modules.base_module import Parameter # Import Parameter here
    T = TypeVar('T')  # Used for generic type definitions
    
    # Define types that reference external modules
    TensorTypes = Union[
        TorchArray,
        Any,  # TorchTensor
        Any,  # EmberTensor
        Any,  # numpy.ndarray
        Parameter # Add Parameter here
    ]
    
    ArrayLike = Union[
        Any,  # TorchTensor
        TorchArray, 
        Numeric, 
        List[Any], 
        Tuple[Any, ...]
    ]
    
    DTypes = Union[
        torch.dtype,
        Any,  # numpy.dtype
    ]
    
    TensorLike = Optional[Union[
        Numeric,
        bool,
        List[Any],
        Tuple[Any, ...],
        'TensorTypes'
    ]]
    
    ScalarLike = Optional[Union[
        Numeric,
        bool,
        TorchArray,
        'TensorLike'
    ]]


__all__ = [
    'Numeric',
    'TensorLike',
    'Shape',
    'ShapeType', 
    'ShapeLike',
    'DTypeClass',
    'DTypes',
    'TorchArray',
    'ArrayLike', 
    'TensorTypes',
    'DimSize',
    'Axis',
    'ScalarLike',
    'OrdLike',
    'Device',
    'IndexType',
    'Indices',
    'PathLike'
]
