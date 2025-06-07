"""
Type definitions for MLX tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""
import os
from typing import (
    Any, List, Optional, Sequence, Tuple, Union,
    TYPE_CHECKING
)

import mlx.core as mx

# Basic type aliases that don't require imports
Numeric = Union[int, float]
OrdLike = Optional[Union[int, str]]
Device = Optional[str]
PathLike = Union[str, os.PathLike[str]]
Shape = Sequence[int]
ShapeType = Union[int, Tuple[int, ...], List[int]]
ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]
DimSize = Union[int, 'mx.array']
Axis = Optional[Union[int, Sequence[int]]]
IndexType = Union[int, Sequence[int], 'mx.array']
Indices = Union[Sequence[int], 'mx.array']
TensorLike = Optional[Union[ # type: ignore
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    'MLXArray',
    'numpy.ndarray',
]]
ScalarLike = Optional[Union[ # type: ignore
    Numeric,
    bool,
    'MLXArray',
    'mx.array',
    'numpy.ndarray',
]]

# MLX specific
MLXArray = mx.array
DTypeClass = mx.Dtype

# Precision related
default_int = mx.int32
default_float = mx.float32
default_bool = mx.bool_ if hasattr(mx, 'bool_') else Any # type: ignore


# Default type for dtype
DType = Any

# Conditional type definitions
if TYPE_CHECKING == True:
    # These imports are for type checking only
    # Must be done inside TYPE_CHECKING block to avoid circular imports
    from typing import TypeVar
    # Import Parameter from its correct location
    from ember_ml.nn.modules import Parameter
    T = TypeVar('T')  # Used for generic type definitions
    
    # Define types that reference external modules
    TensorTypes = Union[
        MLXArray,
        Any,  # MLXTensor placeholder if exists
        Any,  # EmberTensor placeholder if exists
        Any,  # numpy.ndarray
        Parameter # Added Parameter
    ]
    
    ArrayLike = Union[
        Any,  # MLXTensor
        MLXArray, 
        Numeric, 
        List[Any], 
        Tuple[Any, ...]
    ]
    
    DTypes = Union[
        mx.Dtype,
        Any,  # numpy.dtype
    ]
    
    # Ensure TensorLike correctly uses the updated TensorTypes
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
        MLXArray,
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
    'MLXArray',
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


