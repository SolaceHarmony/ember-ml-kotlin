"""
NumPy tensor module for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

from ember_ml.backend.numpy.tensor.dtype import NumpyDType
from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
from ember_ml.backend.numpy.tensor.ops import (
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
    to_numpy, item, shape, dtype, copy, var, sort, argsort, maximum,
    
    # Random operations
    random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
    random_poisson, random_categorical, random_permutation, shuffle, set_seed, get_seed
)

__all__ = [
    # Classes
    'NumpyDType',
    'NumpyTensor',
    
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