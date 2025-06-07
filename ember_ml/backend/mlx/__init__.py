"""
MLX backend for ember_ml.

This module provides MLX implementations of tensor operations.
"""

# Import configuration variables
from ember_ml.backend.mlx.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE
)


# Import specific functions from math_ops
from ember_ml.backend.mlx.math_ops import (
    add,
    subtract,
    multiply,
    divide,
    matmul,
    dot,
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
    negative,
    mod,
    floor_divide,
    floor,
    ceil,
    gradient
)


# Import specific functions from comparison_ops
from ember_ml.backend.mlx.comparison_ops import (
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
    allclose,
    isclose,
    all,
    any,
    where,
    isnan
)

# Import specific functions from device_ops
from ember_ml.backend.mlx.device_ops import (
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
# Import specific functions from io_ops
from ember_ml.backend.mlx.io_ops import (
    save,
    load
)

# Import specific functions from vector_ops
from ember_ml.backend.mlx.vector_ops import (
    normalize_vector,
    compute_energy_stability,
    compute_interference_strength,
    compute_phase_coherence,
    partial_interference,
    euclidean_distance,
    cosine_similarity,
    exponential_decay,
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
)

# Import specific functions from loss_ops
from ember_ml.backend.mlx.loss_ops import (
    mse,
    mean_absolute_error,
    binary_crossentropy,
    categorical_crossentropy,
    sparse_categorical_crossentropy,
    huber_loss,
    log_cosh_loss
)

# Set power function
power = pow

# Define the list of symbols to export
__all__ = [
    # Configuration variables
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',

    # Math operations
    'add',
    'subtract',
    'multiply',
    'divide',
    'matmul',
    'dot',

    'exp',
    'log',
    'log10',
    'log2',
    'pow',
    'sqrt',
    'square',
    'abs',
    'sign',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'negative',
    'clip',
    'mod',
    'floor_divide',
    'floor',
    'ceil',
    'gradient',

    # Comparison operations
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    'logical_and',
    'logical_or',
    'logical_not',
    'logical_xor',
    'allclose',
    'isclose',
    'all',
    'any',
    'where',
    'isnan',
    
    # Device operations
    'to_device',
    'get_device',
    'get_available_devices',
    'memory_usage',
    'memory_info',
    'synchronize',
    'set_default_device',
    'get_default_device',
    'is_available',
    
    # I/O operations
    'save',
    'load',
    
    # Vector operations
    'normalize_vector',
    'compute_energy_stability',
    'compute_interference_strength',
    'compute_phase_coherence',
    'partial_interference',
    'euclidean_distance',
    'cosine_similarity',
    'exponential_decay',
    # 'gaussian', # Handled by ops.stats
    'fft',
    'ifft',
    'fft2',
    'ifft2',
    'fftn',
    'ifftn',
    'rfft',
    'irfft',
    'rfft2',
    'irfft2',
    'rfftn',
    'irfftn',
    # Loss Ops Functions (added)
    'mse',
    'mean_absolute_error',
    'binary_crossentropy',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
    'huber_loss',
    'log_cosh_loss',


]