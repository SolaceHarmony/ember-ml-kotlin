"""
Type definitions for NumPy tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import (
    Any, List, Optional, Sequence, Tuple, Union, Literal,
    TYPE_CHECKING
)
import os

import numpy as np


# Basic type aliases that don't require imports
Numeric = Union[int, float]
OrdLike = Optional[Union[int, str]]
Device = Optional[str]
PathLike = Union[str, os.PathLike[str]]
Shape = Sequence[int]
ShapeType = Union[int, Tuple[int, ...], List[int]]
ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]
DimSize = Union[int, 'np.ndarray']
Axis = Optional[Union[int, Sequence[int]]]
IndexType = Union[int, Sequence[int], 'np.ndarray']
Indices = Union[Sequence[int], 'np.ndarray']
# Numpy specific
NumpyArray = np.ndarray
DTypeClass = np.dtype

# Precision related
default_int = np.int32
default_float = np.float32
default_bool = np.bool_ if hasattr(np, 'bool_') else Any


# Runtime definitions (simplified)
TensorTypes = Any # type: ignore
ArrayLike = Any # type: ignore
TensorLike = Any # type: ignore
ScalarLike = Any # type: ignore
DTypes = Any # type: ignore
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
        NumpyArray,
        Any,  # NumpyTensor
        Any,  # EmberTensor
        Any,  # numpy.ndarray
        Parameter # Add Parameter here
    ]
    
    ArrayLike = Union[
        Any,  # NumpyTensor
        NumpyArray, 
        Numeric, 
        List[Any], 
        Tuple[Any, ...]
    ]
    
    DTypes = Union[
        np.dtype,
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
        NumpyArray,
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
    'NumpyArray',
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


