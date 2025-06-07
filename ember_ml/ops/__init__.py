"""
Operations module.

This module provides a unified interface to operations from the current backend
(NumPy, PyTorch, MLX) using a proxy module pattern. It dynamically forwards
attribute access to the appropriate backend module, avoiding the need for
explicit alias updates.
"""

import logging
from typing import Optional, Any, Union, TypeVar, Protocol, runtime_checkable, List

# Setup basic logging configuration if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import types for type annotations
@runtime_checkable
class TensorLike(Protocol):
    """Protocol for tensor-like objects"""
    def __array__(self) -> Any: ...

type Tensor = Any  # Placeholder for actual tensor types
type DType = Any  # Placeholder for actual data types

# Define type variables for function signatures
T = TypeVar('T', bound=TensorLike)
D = TypeVar('D', bound=Union[str, DType])

# Import the ops module from the proxy module
from ember_ml.ops.proxy import ops_module

# Import backend control functions from the ops module
get_backend = ops_module.get_backend
set_backend = ops_module.set_backend
auto_select_backend = ops_module.auto_select_backend

# Import pi from the ops module
pi = ops_module.pi

# Import submodules from the ops module
stats = ops_module.stats
linearalg = ops_module.linearalg
bitwise = ops_module.bitwise

# Import all operations from the ops module
# Math operations
add = ops_module.add
subtract = ops_module.subtract
multiply = ops_module.multiply
divide = ops_module.divide
matmul = ops_module.matmul
dot = ops_module.dot
exp = ops_module.exp
log = ops_module.log
log10 = ops_module.log10
log2 = ops_module.log2
pow = ops_module.pow
sqrt = ops_module.sqrt
square = ops_module.square
abs = ops_module.abs
sign = ops_module.sign
sin = ops_module.sin
cos = ops_module.cos
tan = ops_module.tan
sinh = ops_module.sinh
cosh = ops_module.cosh
clip = ops_module.clip
negative = ops_module.negative
mod = ops_module.mod
floor_divide = ops_module.floor_divide
floor = ops_module.floor
ceil = ops_module.ceil
gradient = ops_module.gradient
power = ops_module.power

# Comparison operations
equal = ops_module.equal
not_equal = ops_module.not_equal
less = ops_module.less
less_equal = ops_module.less_equal
greater = ops_module.greater
greater_equal = ops_module.greater_equal
logical_and = ops_module.logical_and
logical_or = ops_module.logical_or
logical_not = ops_module.logical_not
logical_xor = ops_module.logical_xor
allclose = ops_module.allclose
isclose = ops_module.isclose
all = ops_module.all
any = ops_module.any
where = ops_module.where
isnan = ops_module.isnan

# Device operations
to_device = ops_module.to_device
get_device = ops_module.get_device
get_available_devices = ops_module.get_available_devices
memory_usage = ops_module.memory_usage
memory_info = ops_module.memory_info
synchronize = ops_module.synchronize
set_default_device = ops_module.set_default_device
get_default_device = ops_module.get_default_device
is_available = ops_module.is_available

# IO operations
save = ops_module.save
load = ops_module.load

# Loss operations
mse = ops_module.mse
mean_absolute_error = ops_module.mean_absolute_error
binary_crossentropy = ops_module.binary_crossentropy
categorical_crossentropy = ops_module.categorical_crossentropy
sparse_categorical_crossentropy = ops_module.sparse_categorical_crossentropy
huber_loss = ops_module.huber_loss
log_cosh_loss = ops_module.log_cosh_loss

# Vector operations
normalize_vector = ops_module.normalize_vector
compute_energy_stability = ops_module.compute_energy_stability
compute_interference_strength = ops_module.compute_interference_strength
compute_phase_coherence = ops_module.compute_phase_coherence
partial_interference = ops_module.partial_interference
euclidean_distance = ops_module.euclidean_distance
cosine_similarity = ops_module.cosine_similarity
exponential_decay = ops_module.exponential_decay
fft = ops_module.fft
ifft = ops_module.ifft
fft2 = ops_module.fft2
ifft2 = ops_module.ifft2
fftn = ops_module.fftn
ifftn = ops_module.ifftn
rfft = ops_module.rfft
irfft = ops_module.irfft
rfft2 = ops_module.rfft2
irfft2 = ops_module.irfft2
rfftn = ops_module.rfftn
irfftn = ops_module.irfftn

# Array manipulation
vstack = ops_module.vstack
hstack = ops_module.hstack

# Master list of all operations for __all__
_MASTER_OPS_LIST = [
    # Math operations
    'add', 'subtract', 'multiply', 'divide', 'matmul', 'dot',
    'exp', 'log', 'log10', 'log2', 'pow', 'sqrt', 'square', 'abs', 'sign', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'clip', 'negative', 'mod', 'floor_divide', 'floor', 'ceil', 'gradient',
    'power',
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'logical_and', 'logical_or',
    'logical_not', 'logical_xor', 'allclose', 'isclose', 'all', 'any', 'where', 'isnan',
    # Device operations
    'to_device', 'get_device', 'get_available_devices', 'memory_usage', 'memory_info', 'synchronize',
    'set_default_device', 'get_default_device', 'is_available',
    # IO operations
    'save', 'load',
    # Loss operations
    'mse', 'mean_absolute_error', 'binary_crossentropy', 'categorical_crossentropy',
    'sparse_categorical_crossentropy', 'huber_loss', 'log_cosh_loss',
    # Vector operations
    'normalize_vector', 'compute_energy_stability', 'compute_interference_strength', 'compute_phase_coherence',
    'partial_interference', 'euclidean_distance', 'cosine_similarity', 'exponential_decay', 'fft', 'ifft',
    'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    # Array manipulation
    'vstack', 'hstack',
]

# Define __all__ to include backend controls, pi, submodules, and all operations
__all__ = [
    'set_backend', 'get_backend', 'auto_select_backend',  # Backend controls
    'pi',  # Constants
    'stats', 'linearalg', 'bitwise',  # Submodules
] + _MASTER_OPS_LIST  # All operations
