"""
Proxy module for operations.

This module provides proxy classes for dynamically forwarding operations
to the current backend. It uses the BackendRegistry and ProxyModule pattern
to avoid the need for explicit alias updates.
"""

import importlib
import sys
from typing import Dict, List, Set, Callable, Any, Optional, Type, TypeVar

from ember_ml.backend.registry import BackendRegistry, ProxyModule, create_proxy_module

# Create proxy module classes for each operation category
MathOpsProxy = create_proxy_module("math_ops", "{backend}.math_ops")
ComparisonOpsProxy = create_proxy_module("comparison_ops", "{backend}.comparison_ops")
DeviceOpsProxy = create_proxy_module("device_ops", "{backend}.device_ops")
IOOpsProxy = create_proxy_module("io_ops", "{backend}.io_ops")
LossOpsProxy = create_proxy_module("loss_ops", "{backend}.loss_ops")
VectorOpsProxy = create_proxy_module("vector_ops", "{backend}.vector_ops")
StatsProxy = create_proxy_module("stats", "{backend}.stats")
LinearAlgProxy = create_proxy_module("linearalg", "{backend}.linearalg")
BitwiseProxy = create_proxy_module("bitwise", "{backend}.bitwise")

# Create instances of the proxy modules
math_ops = MathOpsProxy("math_ops", "ember_ml.ops")
comparison_ops = ComparisonOpsProxy("comparison_ops", "ember_ml.ops")
device_ops = DeviceOpsProxy("device_ops", "ember_ml.ops")
io_ops = IOOpsProxy("io_ops", "ember_ml.ops")
loss_ops = LossOpsProxy("loss_ops", "ember_ml.ops")
vector_ops = VectorOpsProxy("vector_ops", "ember_ml.ops")
stats = StatsProxy("stats", "ember_ml.ops")
linearalg = LinearAlgProxy("linearalg", "ember_ml.ops")
bitwise = BitwiseProxy("bitwise", "ember_ml.ops")

# Initialize the proxy modules with the current backend
from ember_ml.backend import get_backend, get_backend_module
backend_name = get_backend()
backend_module = get_backend_module()
registry = BackendRegistry()
registry.set_backend(backend_name, backend_module)

# Create a dictionary mapping operation names to their proxy modules
_OPS_MAPPING = {
    # Math operations
    'add': math_ops,
    'subtract': math_ops,
    'multiply': math_ops,
    'divide': math_ops,
    'matmul': math_ops,
    'dot': math_ops,
    'exp': math_ops,
    'log': math_ops,
    'log10': math_ops,
    'log2': math_ops,
    'pow': math_ops,
    'sqrt': math_ops,
    'square': math_ops,
    'abs': math_ops,
    'sign': math_ops,
    'sin': math_ops,
    'cos': math_ops,
    'tan': math_ops,
    'sinh': math_ops,
    'cosh': math_ops,
    'clip': math_ops,
    'negative': math_ops,
    'mod': math_ops,
    'floor_divide': math_ops,
    'floor': math_ops,
    'ceil': math_ops,
    'gradient': math_ops,
    'power': math_ops,

    # Comparison operations
    'equal': comparison_ops,
    'not_equal': comparison_ops,
    'less': comparison_ops,
    'less_equal': comparison_ops,
    'greater': comparison_ops,
    'greater_equal': comparison_ops,
    'logical_and': comparison_ops,
    'logical_or': comparison_ops,
    'logical_not': comparison_ops,
    'logical_xor': comparison_ops,
    'allclose': comparison_ops,
    'isclose': comparison_ops,
    'all': comparison_ops,
    'any': comparison_ops,
    'where': comparison_ops,
    'isnan': comparison_ops,

    # Device operations
    'to_device': device_ops,
    'get_device': device_ops,
    'get_available_devices': device_ops,
    'memory_usage': device_ops,
    'memory_info': device_ops,
    'synchronize': device_ops,
    'set_default_device': device_ops,
    'get_default_device': device_ops,
    'is_available': device_ops,

    # IO operations
    'save': io_ops,
    'load': io_ops,

    # Loss operations
    'mse': loss_ops,
    'mean_absolute_error': loss_ops,
    'binary_crossentropy': loss_ops,
    'categorical_crossentropy': loss_ops,
    'sparse_categorical_crossentropy': loss_ops,
    'huber_loss': loss_ops,
    'log_cosh_loss': loss_ops,

    # Vector operations
    'normalize_vector': vector_ops,
    'compute_energy_stability': vector_ops,
    'compute_interference_strength': vector_ops,
    'compute_phase_coherence': vector_ops,
    'partial_interference': vector_ops,
    'euclidean_distance': vector_ops,
    'cosine_similarity': vector_ops,
    'exponential_decay': vector_ops,
    'fft': vector_ops,
    'ifft': vector_ops,
    'fft2': vector_ops,
    'ifft2': vector_ops,
    'fftn': vector_ops,
    'ifftn': vector_ops,
    'rfft': vector_ops,
    'irfft': vector_ops,
    'rfft2': vector_ops,
    'irfft2': vector_ops,
    'rfftn': vector_ops,
    'irfftn': vector_ops,

    # Stats operations
    'mean': stats,
    'var': stats,
    'median': stats,
    'std': stats,
    'percentile': stats,
    'max': stats,
    'min': stats,
    'sum': stats,
    'cumsum': stats,
    'argmax': stats,
    'sort': stats,
    'argsort': stats,
    'gaussian': stats,

    # Linear algebra operations
    'solve': linearalg,
    'inv': linearalg,
    'svd': linearalg,
    'eig': linearalg,
    'eigh': linearalg,
    'eigvals': linearalg,
    'det': linearalg,
    'norm': linearalg,
    'qr': linearalg,
    'cholesky': linearalg,
    'lstsq': linearalg,
    'diag': linearalg,
    'diagonal': linearalg,
    'orthogonal': linearalg,
    'HPC16x8': linearalg,

    # Bitwise operations
    'bitwise_and': bitwise,
    'bitwise_or': bitwise,
    'bitwise_xor': bitwise,
    'bitwise_not': bitwise,
    'left_shift': bitwise,
    'right_shift': bitwise,
    'rotate_left': bitwise,
    'rotate_right': bitwise,
    'count_ones': bitwise,
    'count_zeros': bitwise,
    'get_bit': bitwise,
    'set_bit': bitwise,
    'toggle_bit': bitwise,
    'binary_wave_interference': bitwise,
    'binary_wave_propagate': bitwise,
    'create_duty_cycle': bitwise,
    'generate_blocky_sin': bitwise,

    # Array manipulation
    'vstack': math_ops,
    'hstack': math_ops,
}

# Create a class to handle dynamic attribute access for the ops module
class OpsModule:
    """
    Class that dynamically forwards attribute access to the appropriate proxy module.

    This class is used to implement the ember_ml.ops module, which provides a unified
    interface to operations from the current backend.
    """
    def __init__(self):
        """Initialize the ops module."""
        self._proxy_modules = {
            'math_ops': math_ops,
            'comparison_ops': comparison_ops,
            'device_ops': device_ops,
            'io_ops': io_ops,
            'loss_ops': loss_ops,
            'vector_ops': vector_ops,
            'stats': stats,
            'linearalg': linearalg,
            'bitwise': bitwise,
        }

        # Expose submodules directly
        self.stats = stats
        self.linearalg = linearalg
        self.bitwise = bitwise

        # Import backend control functions
        from ember_ml.backend import get_backend, auto_select_backend
        self.get_backend = get_backend
        self.auto_select_backend = auto_select_backend

        # Define set_backend to update the registry
        def set_backend(backend: str):
            """Set the backend and update all proxy modules."""
            from ember_ml.backend import set_backend as original_set_backend
            original_set_backend(backend)

        self.set_backend = set_backend

        # Import pi from math as a fallback
        import math
        self.pi = math.pi

        # Try to get pi from the current backend
        try:
            from ember_ml.backend import get_backend_module
            backend_module = get_backend_module()
            if hasattr(backend_module, 'math_ops') and hasattr(backend_module.math_ops, 'pi'):
                self.pi = backend_module.math_ops.pi
            elif hasattr(backend_module, 'pi'):
                self.pi = backend_module.pi
        except (ImportError, AttributeError):
            pass  # Keep math.pi as fallback

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the appropriate proxy module.

        Args:
            name: The name of the attribute to access

        Returns:
            The attribute from the appropriate proxy module

        Raises:
            AttributeError: If the attribute is not found in any proxy module
        """
        # Check if the attribute is in the ops mapping
        if name in _OPS_MAPPING:
            proxy_module = _OPS_MAPPING[name]
            return getattr(proxy_module, name)

        # Check if the attribute is a proxy module
        if name in self._proxy_modules:
            return self._proxy_modules[name]

        # If not found, raise AttributeError
        raise AttributeError(f"'ember_ml.ops' has no attribute '{name}'")

# Create an instance of the ops module
ops_module = OpsModule()
