"""
Asynchronous operations module.

This module provides a unified interface to asynchronous operations from the current backend
(NumPy, PyTorch, MLX) using the proxy module pattern. It dynamically forwards
attribute access to the appropriate backend module, wrapping functions in asynchronous
wrappers using asyncio.to_thread.
"""

from typing import Any, List

# Import the async ops module from the proxy module
from ember_ml.asyncml.ops.proxy import async_ops_module

# Import backend control functions from the async ops module
get_backend = async_ops_module.get_backend
set_backend = async_ops_module.set_backend
auto_select_backend = async_ops_module.auto_select_backend

# Import pi from the async ops module
pi = async_ops_module.pi

# Import submodules from the async ops module
stats = async_ops_module.stats
linearalg = async_ops_module.linearalg
bitwise = async_ops_module.bitwise

# Import all operations from the async ops module
# Math operations
add = async_ops_module.add
subtract = async_ops_module.subtract
multiply = async_ops_module.multiply
divide = async_ops_module.divide
matmul = async_ops_module.matmul
dot = async_ops_module.dot
exp = async_ops_module.exp
log = async_ops_module.log
log10 = async_ops_module.log10
log2 = async_ops_module.log2
pow = async_ops_module.pow
sqrt = async_ops_module.sqrt
square = async_ops_module.square
abs = async_ops_module.abs
sign = async_ops_module.sign
sin = async_ops_module.sin
cos = async_ops_module.cos
tan = async_ops_module.tan
sinh = async_ops_module.sinh
cosh = async_ops_module.cosh
clip = async_ops_module.clip
negative = async_ops_module.negative
mod = async_ops_module.mod
floor_divide = async_ops_module.floor_divide
floor = async_ops_module.floor
ceil = async_ops_module.ceil
gradient = async_ops_module.gradient
power = async_ops_module.power

# Comparison operations
equal = async_ops_module.equal
not_equal = async_ops_module.not_equal
less = async_ops_module.less
less_equal = async_ops_module.less_equal
greater = async_ops_module.greater
greater_equal = async_ops_module.greater_equal
logical_and = async_ops_module.logical_and
logical_or = async_ops_module.logical_or
logical_not = async_ops_module.logical_not
logical_xor = async_ops_module.logical_xor
allclose = async_ops_module.allclose
isclose = async_ops_module.isclose
all = async_ops_module.all
any = async_ops_module.any
where = async_ops_module.where
isnan = async_ops_module.isnan

# Device operations
to_device = async_ops_module.to_device
get_device = async_ops_module.get_device
get_available_devices = async_ops_module.get_available_devices
memory_usage = async_ops_module.memory_usage
memory_info = async_ops_module.memory_info
synchronize = async_ops_module.synchronize
set_default_device = async_ops_module.set_default_device
get_default_device = async_ops_module.get_default_device
is_available = async_ops_module.is_available

# IO operations
save = async_ops_module.save
load = async_ops_module.load

# Loss operations
mse = async_ops_module.mse
mean_absolute_error = async_ops_module.mean_absolute_error
binary_crossentropy = async_ops_module.binary_crossentropy
categorical_crossentropy = async_ops_module.categorical_crossentropy
sparse_categorical_crossentropy = async_ops_module.sparse_categorical_crossentropy
huber_loss = async_ops_module.huber_loss
log_cosh_loss = async_ops_module.log_cosh_loss

# Vector operations
normalize_vector = async_ops_module.normalize_vector
compute_energy_stability = async_ops_module.compute_energy_stability
compute_interference_strength = async_ops_module.compute_interference_strength
compute_phase_coherence = async_ops_module.compute_phase_coherence
partial_interference = async_ops_module.partial_interference
euclidean_distance = async_ops_module.euclidean_distance
cosine_similarity = async_ops_module.cosine_similarity
exponential_decay = async_ops_module.exponential_decay
fft = async_ops_module.fft
ifft = async_ops_module.ifft
fft2 = async_ops_module.fft2
ifft2 = async_ops_module.ifft2
fftn = async_ops_module.fftn
ifftn = async_ops_module.ifftn
rfft = async_ops_module.rfft
irfft = async_ops_module.irfft
rfft2 = async_ops_module.rfft2
irfft2 = async_ops_module.irfft2
rfftn = async_ops_module.rfftn
irfftn = async_ops_module.irfftn

# Array manipulation
vstack = async_ops_module.vstack
hstack = async_ops_module.hstack

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
