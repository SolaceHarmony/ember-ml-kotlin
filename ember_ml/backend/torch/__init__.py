"""
PyTorch backend for ember_ml.

This module provides PyTorch implementations of tensor operations.
"""

# Define the list of symbols to export
__all__ = [
    
    # Math operations
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'mean', 'sum', 'max', 'min', 'exp', 'log', 'log10', 'log2',
    'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'sigmoid', 'relu', 'softmax', 'clip',
    'var', 'pi', 'power', 'negative', 'mod', 'floor_divide', 'floor', 'ceil', 'sort', 'gradient', 'cumsum', 'eigh', # Added missing math ops
    
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'where', 'isnan', # Added missing comparison ops
    # Device operations
    'to_device', 'get_device', 'get_available_devices', 'memory_usage',
    'memory_info', 'synchronize', 'set_default_device', 'get_default_device',
    'is_available',
    
    # Linear Algebra operations
    'solve', 'inv', 'svd', 'eig', 'eigvals', 'det', 'norm', 'qr',
    'cholesky', 'lstsq', 'diag', 'diagonal', # Added missing linear algebra ops
    # I/O operations
    'save', 'load',
    
    # Feature operations
    'pca', 'transform', 'inverse_transform', 'standardize', 'normalize', # Added missing feature ops
    # Vector operations
    'normalize_vector', 'euclidean_distance', 'cosine_similarity',
    'exponential_decay', 'compute_energy_stability', # Removed gaussian from here
    'compute_interference_strength', 'compute_phase_coherence',
    'partial_interference', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn',
    'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', # Removed gaussian
    # Loss Ops Functions (added)
    'mse', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',
    'sparse_categorical_crossentropy', 'huber_loss', 'log_cosh_loss',

    # Functional Tensor Ops (from tensor.ops - added)
    'cast', 'zeros', 'ones', 'eye', 'zeros_like', 'ones_like', 'full', 'full_like', 'arange', 'linspace',
    'reshape', 'transpose', 'concatenate', 'stack', 'split', 'expand_dims', 'squeeze', 'tile', 'pad',
    'vstack', 'hstack',
    'slice_tensor', 'slice_update', 'gather', 'tensor_scatter_nd_update', 'scatter',
    'convert_to_torch_tensor', 'to_numpy', 'item', 'shape', 'dtype', 'copy', 'argsort', 'maximum',
    'random_normal', 'random_uniform', 'random_binomial', 'random_gamma', 'random_exponential',
    'random_poisson', 'random_categorical', 'random_permutation', 'shuffle', 'set_seed', 'get_seed',
   # Stats Ops Functions (added)
   'gaussian',
   # Add other stats ops like median, std, percentile here if needed for aliasing
   # 'median', 'std', 'percentile',

   # Activation Ops Functions (added)
   'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
]

# Import specific functions from math_ops
from ember_ml.backend.torch.math_ops import (
    add,
    subtract,
    multiply,
    divide,
    matmul,
    dot,
    mean,
    sum,
    max,
    min,
    exp,
    log,
    log10,
    log2,
    pow,
    sqrt,
    square,
    abs,
    sign,
    sin,
    cos,
    tan,
    sinh,
    cosh,
   clip,
   var,
   pi,
   negative, # Added import
   mod,      # Added import
   floor_divide, # Added import
   sort,     # Added import
   gradient, # Added import
   cumsum,   # Added import
   eigh,     # Added import
   floor,    # Added import
   ceil      # Added import
)

# Import specific functions from comparison_ops
from ember_ml.backend.torch.comparison_ops import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    logical_and,
   logical_or,
   logical_not,
   logical_xor,
   allclose, # Added import
   isclose,  # Added import
   all,      # Added import
   where,    # Added import
   isnan     # Added import
)

# Import specific functions from device_ops
from ember_ml.backend.torch.device_ops import (
    to_device,
    get_device,
    get_available_devices,
    memory_usage,
    memory_info,
    synchronize,
    set_default_device,
    get_default_device,
    is_available
)

# Import specific functions from linearalg
from ember_ml.backend.torch.linearalg import (
    solve,
    inv,
    svd,
    eig,
    eigvals,
    det,
    norm,
    qr,
    cholesky,
    lstsq,
    diag,
    diagonal
)

# Import specific functions from io_ops
from ember_ml.backend.torch.io_ops import (
    save,
   load
)

# Import specific functions from loss_ops (Added)
from ember_ml.backend.torch.loss_ops import (
   mse, mean_absolute_error, binary_crossentropy, categorical_crossentropy,
   sparse_categorical_crossentropy, huber_loss, log_cosh_loss
)


# Import specific functions from vector_ops
from ember_ml.backend.torch.vector_ops import (
    normalize_vector,
    euclidean_distance,
    cosine_similarity,
    exponential_decay,
    compute_energy_stability,
    compute_interference_strength,
    compute_phase_coherence,
    partial_interference,
    fft,
    ifft,
    fft2,
    ifft2,
    fftn,
    ifftn,
    rfft,
    irfft,
    rfft2,
    irfft2,
  rfftn,
  irfftn
) # End of vector_ops import

# Import specific functions from stats (Corrected path)
from ember_ml.backend.torch.stats.descriptive import gaussian # Corrected import path after flattening
# from ember_ml.backend.torch.stats import median, std, percentile # Example if importing from stats/__init__ later

# Import functional tensor ops directly (Added)
from ember_ml.backend.torch.tensor.ops import (
   cast, zeros, ones, eye, zeros_like, ones_like, full, full_like, arange, linspace,
   reshape, transpose, concatenate, stack, split, expand_dims, squeeze, tile, pad,
   vstack, hstack,
   slice_tensor, slice_update, gather, tensor_scatter_nd_update, scatter,
   to_numpy, item, shape, dtype, copy, var as tensor_var, sort as tensor_sort, argsort, maximum, # aliased var/sort to avoid name clash
   random_normal, random_uniform, random_binomial, random_gamma, random_exponential,
   random_poisson, random_categorical, random_permutation, shuffle, set_seed, get_seed
)

# Set power function
power = pow

# Import activation functions (Added)
from ember_ml.backend.torch.activations.ops import relu, sigmoid, tanh, softmax, softplus
