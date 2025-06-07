"""Type definitions for tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import Union, Optional, Sequence, Any, List, Tuple, TYPE_CHECKING

# Import EmberTensor for type annotations
from ember_ml.nn.tensor.common.dtypes import EmberDType
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor

# Conditionally import backend types for type checking only
if TYPE_CHECKING:
    # These imports are only used for type checking and not at runtime
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    from ember_ml.backend.torch.tensor import TorchTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    # from ember_ml.nn.tensor.common.ember_tensor import EmberTensor # Moved up

# Moved TensorTypes outside of TYPE_CHECKING
TensorTypes = Union[
    Any,  # Placeholder for TensorLike, resolved during type checking
    Any,  # Placeholder for 'mlx.core.array'
    Any,  # Placeholder for MLXTensor
    Any,  # Placeholder for EmberTensor
    Any,  # Placeholder for TorchTensor
    Any   # Placeholder for NumpyTensor
]

# Basic type aliases
Numeric = Union[int, float]

# Standard type aliases for general tensor-like inputs
# This covers all possible input types that can be converted to a tensor
TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    EmberTensor,
    'TensorLike',
    'mlx.core.array',  # MLX array type
    'np.ndarray',      # NumPy array type
    'torch.Tensor',    # PyTorch tensor type
    'NumpyTensor',
    'TorchTensor',
    'MLXTensor',
    'torch.Tensor',
]]

# Dimension-specific tensor types
Scalar = Union[int, float, bool, EmberTensor]  # 0D tensors
Vector = Union[List[Union[int, float, bool]], Tuple[Union[int, float, bool], ...], EmberTensor]  # 1D tensors
Matrix = Union[List[List[Union[int, float, bool]]], EmberTensor]  # 2D tensors

# Shape definitions
Shape = Union[int, Sequence[int]]
ShapeLike = Union[int, Tuple[int, ...], List[int]]

# Dtype definitions
DType = Optional[Union[str, EmberDType, Any]]  # Any covers backend-specific dtype objects
Device = Optional[str]

# Dimension types
Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
ScalarLike = Optional[Union[
    Numeric,
    bool,
    TensorTypes
]]
