"""
PyTorch tensor operations.

This module provides standalone functions for tensor operations using the PyTorch backend.
These functions can be called directly or through the TorchTensor class methods.
"""
# Import functions defined within this directory (ops) using absolute paths
from ember_ml.backend.torch.tensor.ops.casting import cast
from ember_ml.backend.torch.tensor.ops.creation import (
    zeros, ones, eye, zeros_like, ones_like, full, full_like, arange, linspace, meshgrid # Add meshgrid
)
from ember_ml.backend.torch.tensor.ops.manipulation import (
    reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, pad,
    vstack, hstack
)
from ember_ml.backend.torch.tensor.ops.indexing import (
    slice_tensor, slice_update, gather, tensor_scatter_nd_update, scatter, nonzero, index_update
    # Note: scatter_* helpers might be internal to indexing.py
)
from ember_ml.backend.torch.tensor.ops.utility import (
    to_numpy, item, shape, dtype, copy, var, sort, argsort, maximum
)
from ember_ml.backend.torch.tensor.ops.random import (
    random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
    random_poisson, random_categorical, random_permutation, shuffle, random_shuffle, set_seed, get_seed
)

# Define the list of symbols to export
# Export all functions imported above, preserving formatting
__all__ = [
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
    'meshgrid', # Add meshgrid

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
    'scatter',
    'nonzero',
    'index_update',

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
    'random_shuffle',
    'set_seed',
    'get_seed',
]