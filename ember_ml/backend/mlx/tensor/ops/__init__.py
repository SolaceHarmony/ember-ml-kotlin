"""
MLX tensor operations.

This module provides standalone functions for tensor operations using the MLX backend.
These functions can be called directly or through the MLXTensor class methods.
"""

# Import operations from modules
from ember_ml.backend.mlx.tensor.ops.casting import cast

from ember_ml.backend.mlx.tensor.ops.creation import (
    zeros, ones, zeros_like, ones_like, eye, full, full_like, arange, linspace
)

from ember_ml.backend.mlx.tensor.ops.indexing import (
    slice_tensor, slice_update, gather, tensor_scatter_nd_update,
    scatter, scatter_add, scatter_max, scatter_min, scatter_mean, scatter_softmax,
    meshgrid, nonzero, index_update # Add index_update
)

from ember_ml.backend.mlx.tensor.ops.manipulation import (
    reshape, transpose, concatenate, stack, split, split_tensor, expand_dims, squeeze, tile, pad,
    vstack, hstack
)

from ember_ml.backend.mlx.tensor.ops.random import (
    random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
    random_poisson, random_categorical, random_permutation, shuffle, random_shuffle, set_seed, get_seed
)

from ember_ml.backend.mlx.tensor.ops.utility import ( to_numpy, item, shape, dtype, copy, var, sort, argsort, maximum )

# Export all operations
__all__ = [
    # Casting operations
    'cast',
    
    # Creation operations
    'zeros', 
    'ones', 
    'zeros_like', 
    'ones_like', 
    'eye', 
    'full', 
    'full_like', 
    'arange', 
    'linspace',
    
    # Indexing operations
    'slice_tensor',
    'slice_update',
    'gather',
    'tensor_scatter_nd_update',
    'scatter',
    'scatter_add',
    'scatter_max',
    'scatter_min',
    'scatter_mean',
    'scatter_softmax',
    'meshgrid',
    'nonzero',
    'index_update', # Add index_update
    # Manipulation operations
    'reshape', 
    'transpose', 
    'concatenate', 
    'stack',
    'split',
    'split_tensor',
    'expand_dims',
    'squeeze', 
    'tile',
    'pad',
    'vstack',
    'hstack',
    
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
    'random_shuffle',
    'set_seed',
    'get_seed',
    
    # Utility operations
    'convert_to_mlx_tensor',
    'to_numpy', 
    'item', 
    'shape', 
    'dtype', 
    'copy', 
    'var', 
    'sort', 
    'argsort', 
    'maximum',
       
]