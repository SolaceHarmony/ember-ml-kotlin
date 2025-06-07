"""
Common tensor implementations.

This module provides backend-agnostic implementations of tensor operations
using the backend abstraction layer.
"""

import importlib

from ember_ml.backend import get_backend

# Cache for imported backend tensor ops modules
_BACKEND_TENSOR_OPS_MODULES = {}

def _get_backend_tensor_ops_module():
    """Dynamically import and return the tensor ops module for the current backend."""
    backend_name = get_backend()
    if backend_name not in _BACKEND_TENSOR_OPS_MODULES:
        try:
            module_path = f"ember_ml.backend.{backend_name}.tensor.ops"
            _BACKEND_TENSOR_OPS_MODULES[backend_name] = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Could not import tensor ops module for backend '{backend_name}': {e}")
    return _BACKEND_TENSOR_OPS_MODULES[backend_name]

# _get_backend_module() might not be needed anymore if only used for tensor ops
# def _get_backend_module():
#     """Get the current backend module."""
#     try:
#         return get_backend_module() # This function is in ember_ml.backend, might cause issues if called here directly
#     except (ImportError, ModuleNotFoundError):
#         # If backend-specific implementation not found, use common implementation
#         return importlib.import_module('ember_ml.backend.numpy')

# _get_tensor_ops is replaced by _get_backend_tensor_ops_module
# def _get_tensor_ops():
#    """Get the tensor operations for the current backend."""
#    backend = get_backend()
#    # This relied on the old mechanism of attaching Tensor class instances
#    # return _CURRENT_INSTANCES[backend] # Old implementation
#    ops_module = _get_backend_tensor_ops_module()
#    # How to get the ops instance/functions from the module? Needs clarification.
#    # For now, assume the module itself has the functions directly.
#    return ops_module

# Define tensor operations using lambda functions
zeros = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros(*args, **kwargs)
ones = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones(*args, **kwargs)
zeros_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().zeros_like(*args, **kwargs)
ones_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().ones_like(*args, **kwargs)
eye = lambda *args, **kwargs: _get_backend_tensor_ops_module().eye(*args, **kwargs)
arange = lambda *args, **kwargs: _get_backend_tensor_ops_module().arange(*args, **kwargs)
linspace = lambda *args, **kwargs: _get_backend_tensor_ops_module().linspace(*args, **kwargs)
nonzero = lambda *args, **kwargs: _get_backend_tensor_ops_module().nonzero(*args, **kwargs)
full = lambda *args, **kwargs: _get_backend_tensor_ops_module().full(*args, **kwargs)
full_like = lambda *args, **kwargs: _get_backend_tensor_ops_module().full_like(*args, **kwargs)
reshape = lambda *args, **kwargs: _get_backend_tensor_ops_module().reshape(*args, **kwargs)
transpose = lambda *args, **kwargs: _get_backend_tensor_ops_module().transpose(*args, **kwargs)
concatenate = lambda *args, **kwargs: _get_backend_tensor_ops_module().concatenate(*args, **kwargs)
stack = lambda *args, **kwargs: _get_backend_tensor_ops_module().stack(*args, **kwargs)
split = lambda *args, **kwargs: _get_backend_tensor_ops_module().split(*args, **kwargs)
split_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().split_tensor(*args, **kwargs)
expand_dims = lambda *args, **kwargs: _get_backend_tensor_ops_module().expand_dims(*args, **kwargs)
squeeze = lambda *args, **kwargs: _get_backend_tensor_ops_module().squeeze(*args, **kwargs)
tile = lambda *args, **kwargs: _get_backend_tensor_ops_module().tile(*args, **kwargs)
gather = lambda *args, **kwargs: _get_backend_tensor_ops_module().gather(*args, **kwargs)
scatter = lambda *args, **kwargs: _get_backend_tensor_ops_module().scatter(*args, **kwargs)
tensor_scatter_nd_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().tensor_scatter_nd_update(*args, **kwargs)
index_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().index_update(*args, **kwargs)
# Use slice_tensor directly to avoid conflicts with built-in slice
slice_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_tensor(*args, **kwargs)
slice_update = lambda *args, **kwargs: _get_backend_tensor_ops_module().slice_update(*args, **kwargs)

# Rename the current function to indicate it's internal
# Dynamically get the correct convert_to_<backend>_tensor function
_convert_to_backend_tensor = lambda *args, **kwargs: getattr(
    # Dynamically import the backend's utility module
    importlib.import_module(f"ember_ml.backend.{get_backend()}.tensor.ops.utility"),
    # Call the standardized internal function name
    "_convert_to_tensor"
)(*args, **kwargs)
# _convert_to_backend_tensor = lambda *args, **kwargs: _get_backend_tensor_ops_module().convert_to_tensor(*args, **kwargs) # Placeholder comment
shape = lambda *args, **kwargs: _get_backend_tensor_ops_module().shape(*args, **kwargs)
# Define dtype as a function that handles both callable and non-callable backend dtypes
def dtype(data):
    """
    Get the data type of a tensor.
    
    Args:
        data: Input tensor
        
    Returns:
        Data type of the tensor
    """
    # Use the dtype function directly from the backend ops module
    return _get_backend_tensor_ops_module().dtype(data)
cast = lambda *args, **kwargs: _get_backend_tensor_ops_module().cast(*args, **kwargs)
copy = lambda *args, **kwargs: _get_backend_tensor_ops_module().copy(*args, **kwargs)
pad = lambda *args, **kwargs: _get_backend_tensor_ops_module().pad(*args, **kwargs)
item = lambda *args, **kwargs: _get_backend_tensor_ops_module().item(*args, **kwargs)
to_numpy = lambda *args, **kwargs: _get_backend_tensor_ops_module().to_numpy(*args, **kwargs)
# tolist might not be present in all backend ops, handle potentially missing attribute
tolist = lambda *args, **kwargs: getattr(_get_backend_tensor_ops_module(), 'tolist', lambda x: x.tolist())(*args, **kwargs)
random_uniform = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_uniform(*args, **kwargs)
random_normal = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_normal(*args, **kwargs)
maximum = lambda *args, **kwargs: _get_backend_tensor_ops_module().maximum(*args, **kwargs)

# Add missing random operations
def random_bernoulli(*args, **kwargs):
    """Generates Bernoulli random values."""
    seed = kwargs.pop('seed', None)
    if seed is not None:
        # Use the common set_seed function, which dispatches to the backend
        set_seed(seed)
    # Call backend implementation without the seed argument
    # Note: some backends might call this 'random_bernoulli' or 'random_binomial'
    ops_module = _get_backend_tensor_ops_module()
    func = getattr(ops_module, 'random_binomial', getattr(ops_module, 'random_bernoulli', None))
    if func:
        return func(*args, **kwargs)
    else:
        raise AttributeError(f"Backend '{get_backend()}' tensor ops module does not have a 'random_binomial' or 'random_bernoulli' function.")
random_gamma = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_gamma(*args, **kwargs)
random_exponential = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_exponential(*args, **kwargs)
random_poisson = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_poisson(*args, **kwargs)
random_categorical = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_categorical(*args, **kwargs)
random_permutation = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_permutation(*args, **kwargs)
shuffle = lambda *args, **kwargs: _get_backend_tensor_ops_module().shuffle(*args, **kwargs)
random_shuffle = lambda *args, **kwargs: _get_backend_tensor_ops_module().random_shuffle(*args, **kwargs)
set_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().set_seed(*args, **kwargs)
get_seed = lambda *args, **kwargs: _get_backend_tensor_ops_module().get_seed(*args, **kwargs)
meshgrid = lambda *args, **kwargs: _get_backend_tensor_ops_module().meshgrid(*args, **kwargs) # Add meshgrid lambda

# Define a simple Index class that just returns the key when indexed
class Index:
    def __getitem__(self, key):
        return key

# Create a singleton instance of the Index class
index = Index()

# Import EmberTensor class for use in __all__ but don't import it directly
# This avoids the unused import warning
from ember_ml.nn.tensor.common import ember_tensor
EmberTensor = ember_tensor.EmberTensor

__all__ = [
    # Implementations
    'EmberTensor',
    
    # Operations
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'eye',
    'arange',
    'linspace',
    'full',
    'full_like',
    'reshape',
    'transpose',
    'concatenate',
    'stack',
    'split',
    'split_tensor',
    'expand_dims',
    'squeeze',
    'tile',
    'gather',
    'scatter',
    'tensor_scatter_nd_update',
    'index_update',
    'slice_tensor',
    'slice_update',
    'shape',
    'dtype',
    'cast',
    'copy',
    'pad',
    'item',
    'to_numpy',
    'tolist',
    'random_uniform',
    'random_normal',
    'maximum',
    
    # Additional random operations
    'random_bernoulli',
    'random_gamma',
    'random_exponential',
    'random_poisson',
    'random_categorical',
    'random_permutation',
    'shuffle',
    'random_shuffle',
    'set_seed',
    'get_seed',
    'meshgrid',
    'nonzero', # Add nonzero export
    # Note: _convert_to_backend_tensor is intentionally not exported
    'index',
]