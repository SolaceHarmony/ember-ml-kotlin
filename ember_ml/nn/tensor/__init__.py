"""
Tensor module for ember_ml.

This module provides a backend-agnostic tensor implementation that works with
any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.
"""

# Import interfaces
from ember_ml.nn.tensor.interfaces import TensorInterface  # noqa
from ember_ml.nn.tensor.interfaces.dtype import DTypeInterface  # noqa

# Import directly from common implementation
from ember_ml.nn.tensor.common import EmberTensor  # noqa
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    EmberDType, DType, dtype as dtype_instance, # Alias the instance import
    get_dtype, to_dtype_str, from_dtype_str
)
# Import the dtype *function* separately to ensure it's available
from ember_ml.nn.tensor.common import dtype # noqa
# Import dtype objects directly from dtypes.py
from ember_ml.nn.tensor.common.dtypes import (  # noqa
    float32, float64, int32, int64, bool_,
    int8, int16, uint8, uint16, uint32, uint64, float16
)

# Import tensor operations from common
from ember_ml.nn.tensor.common import (  # noqa
    zeros, ones, eye, arange, linspace,
    zeros_like, ones_like, full, full_like,
    reshape, transpose, concatenate, stack, split, split_tensor,
    expand_dims, squeeze, tile, gather, scatter, tensor_scatter_nd_update,
    slice_tensor, slice_update, index_update, cast, copy, pad,
    to_numpy, item, shape,
    random_uniform, random_normal, maximum,
    random_bernoulli, random_gamma, random_exponential, random_poisson,
    random_categorical, random_permutation, shuffle, random_shuffle, set_seed, get_seed,
    meshgrid, nonzero, index # Add nonzero here
)

# Define array function as an alias for EmberTensor constructor
def array(data, dtype=None, device=None, requires_grad=False):
    """
    Create a tensor from data.
    
    Args:
        data: Input data (array, list, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        EmberTensor
    """
    return EmberTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

from typing import Any
 
def convert_to_tensor(data: Any, dtype=None, device=None, requires_grad=False):
    """
    Create a tensor from data.
    
    Args:
        data: Input data (array, list, scalar, or tensor)
        dtype: Optional data type
        device: Optional device to place the tensor on
        requires_grad: Whether the tensor requires gradients
        
    Returns:
        Backend tensor (e.g., TensorLike, tensor.convert_to_tensor, TensorLike)
    """
    # If already an EmberTensor, check if dtype/device/requires_grad match
    if isinstance(data, EmberTensor):
        # TODO: Add logic here to potentially re-wrap or cast if dtype/device/requires_grad differ?
        # For now, return as is, assuming the caller handles potential mismatches if needed.
        # A more robust implementation might create a new EmberTensor if properties differ significantly.
        return data

    # Create and return an EmberTensor instance.
    # The EmberTensor.__init__ method is responsible for handling the backend conversion,
    # dtype setting, device placement, and storing the backend tensor and EmberDType.
    return EmberTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

# Export all classes and functions
__all__ = [
    # Interfaces
    'TensorInterface',
    'DTypeInterface',
    
    # Implementations
    'EmberTensor',
    'EmberDType',
    'DType',
    'dtype', # This should now correctly refer to the function
    # 'dtype_instance' is not typically part of the public API, so omit from __all__
    
    # Tensor constructor
    'array',
    'convert_to_tensor',
    
    # Tensor operations
    'zeros', 'ones', 'eye', 'arange', 'linspace',
    'zeros_like', 'ones_like', 'full', 'full_like',
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'split_tensor',
    'expand_dims', 'squeeze', 'tile', 'gather', 'scatter', 'tensor_scatter_nd_update',
    'slice_tensor', 'slice_update', 'index_update', 'cast', 'copy', 'pad',
    'to_numpy', 'item', 'index','shape',
    'random_uniform', 'random_normal', 'maximum',
    'random_bernoulli', 'random_gamma', 'random_exponential', 'random_poisson',
    'random_categorical', 'random_permutation', 'shuffle', 'random_shuffle', 'set_seed', 'get_seed', 'meshgrid', 'nonzero', # Add nonzero here
    
    # Data types
    'float32', 'float64', 'int32', 'int64', 'bool_',
    'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'float16',
    
    # Data type operations
    'get_dtype', 'to_dtype_str', 'from_dtype_str',
]