"""
PyTorch tensor module for ember_ml.

This module provides PyTorch implementations of tensor operations.
"""

from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.tensor.tensor import TorchTensor
from ember_ml.backend.torch.tensor.ops import (
    # Casting operations
    cast,

    # Creation operations
    zeros, ones, eye, zeros_like, ones_like, full, full_like, arange, linspace,

    # Manipulation operations
    reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, pad,
    vstack, hstack,

    # Indexing operations
    slice_tensor, slice_update, gather, tensor_scatter_nd_update,

    # Utility operations
    convert_to_torch_tensor, to_numpy, item, shape, dtype, copy, var, sort, argsort, maximum,

    # Random operations
    random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
    random_poisson, random_categorical, random_permutation, shuffle, set_seed, get_seed,
)

__all__ = [
    # Classes
    'TorchDType',
    'TorchTensor',
    
    # Casting operations
    'cast',
    
    # Creation operations
    'zeros',
    'ones',
    'eye',
    'zeros_like',
    'ones_like',
    'full',
    'full_like',
    'arange',
    'linspace',
    
    # Manipulation operations
    'reshape',
    'transpose',
    'concatenate',
    'stack',
    'split',
    'expand_dims',
    'squeeze',
    'tile',
    'pad',
    'vstack',
    'hstack',
    
    # Indexing operations
    'slice_tensor',
    'slice_update',
    'gather',
    'tensor_scatter_nd_update',
    
    # Utility operations
    'convert_to_torch_tensor',
    'to_numpy',
    'item',
    'shape',
    'dtype',
    'copy',
    'var',
    'sort',
    'argsort',
    'maximum',
    
    # Random operations
    'random_normal',
    'random_uniform',
    'random_binomial',
    'random_gamma',
    'random_exponential',
    'random_poisson',
    'random_categorical',
    'random_permutation',
    'shuffle',
    'set_seed',
    'get_seed',
]