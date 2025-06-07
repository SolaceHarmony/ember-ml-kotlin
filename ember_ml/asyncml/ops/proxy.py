"""
Proxy module for asynchronous operations.

This module provides proxy classes for dynamically forwarding operations
to the current backend, wrapping them in asynchronous wrappers. It uses
the BackendRegistry and ProxyModule pattern to avoid the need for explicit
alias updates.
"""

import importlib
import sys
import asyncio
import functools
from typing import Dict, List, Set, Callable, Any, Optional, Type, TypeVar

from ember_ml.backend.registry import BackendRegistry, ProxyModule, create_proxy_module
from ember_ml.ops.proxy import _OPS_MAPPING as SYNC_OPS_MAPPING

# Helper to create an asynchronous wrapper for a synchronous function
def _make_async_wrapper(sync_func):
    """Wraps a synchronous function in an async function using asyncio.to_thread."""
    @functools.wraps(sync_func)
    async def async_wrapper(*args, **kwargs):
        # Execute the synchronous function in a separate thread
        return await asyncio.to_thread(sync_func, *args, **kwargs)
    return async_wrapper

# Create a subclass of ProxyModule that wraps synchronous operations with asynchronous wrappers
class AsyncProxyModule(ProxyModule):
    """
    Proxy module that wraps synchronous operations with asynchronous wrappers.
    
    This class extends the ProxyModule class to wrap synchronous operations
    from the backend with asynchronous wrappers.
    """
    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the current backend module,
        wrapping functions in asynchronous wrappers.
        
        Args:
            name: The name of the attribute to access
            
        Returns:
            The attribute from the backend module, wrapped in an asynchronous
            wrapper if it's a callable
            
        Raises:
            AttributeError: If the attribute is not found in the backend module
        """
        if self._backend_module is None:
            raise AttributeError(f"No backend module set for {self._full_name}")
            
        try:
            attr = getattr(self._backend_module, name)
            # If the attribute is callable, wrap it in an asynchronous wrapper
            if callable(attr) and not name.startswith('_'):
                return _make_async_wrapper(attr)
            # Otherwise, return it as is
            return attr
        except AttributeError:
            raise AttributeError(f"'{self._full_name}' has no attribute '{name}' in backend {self._backend_module.__name__}")

# Create a function to create async proxy module classes
def create_async_proxy_module(module_name: str, backend_path_template: str) -> Type[AsyncProxyModule]:
    """
    Create an async proxy module class for a specific module.
    
    This function creates a subclass of AsyncProxyModule that implements the
    _get_backend_module method to return the appropriate backend module based
    on the backend_path_template.
    
    Args:
        module_name: The name of the module
        backend_path_template: A template string for the backend module path,
            e.g., "{backend}.{module}" where {backend} will be replaced with
            the backend name and {module} with the module name
            
    Returns:
        A subclass of AsyncProxyModule
    """
    class SpecificAsyncProxyModule(AsyncProxyModule):
        def _get_backend_module(self, backend_name: str, backend_module: Any) -> Any:
            """Get the backend module for this proxy."""
            # Replace placeholders in the template
            backend_path = backend_path_template.format(
                backend=backend_module.__name__,
                module=module_name
            )
            
            try:
                module = importlib.import_module(backend_path)
                return module
            except ImportError as e:
                print(f"Warning: Could not import backend module {backend_path}: {e}")
                return None
                
    return SpecificAsyncProxyModule

# Create async proxy module classes for each operation category
AsyncMathOpsProxy = create_async_proxy_module("math_ops", "{backend}.math_ops")
AsyncComparisonOpsProxy = create_async_proxy_module("comparison_ops", "{backend}.comparison_ops")
AsyncDeviceOpsProxy = create_async_proxy_module("device_ops", "{backend}.device_ops")
AsyncIOOpsProxy = create_async_proxy_module("io_ops", "{backend}.io_ops")
AsyncLossOpsProxy = create_async_proxy_module("loss_ops", "{backend}.loss_ops")
AsyncVectorOpsProxy = create_async_proxy_module("vector_ops", "{backend}.vector_ops")
AsyncStatsProxy = create_async_proxy_module("stats", "{backend}.stats")
AsyncLinearAlgProxy = create_async_proxy_module("linearalg", "{backend}.linearalg")
AsyncBitwiseProxy = create_async_proxy_module("bitwise", "{backend}.bitwise")

# Create instances of the async proxy modules
async_math_ops = AsyncMathOpsProxy("math_ops", "ember_ml.asyncml.ops")
async_comparison_ops = AsyncComparisonOpsProxy("comparison_ops", "ember_ml.asyncml.ops")
async_device_ops = AsyncDeviceOpsProxy("device_ops", "ember_ml.asyncml.ops")
async_io_ops = AsyncIOOpsProxy("io_ops", "ember_ml.asyncml.ops")
async_loss_ops = AsyncLossOpsProxy("loss_ops", "ember_ml.asyncml.ops")
async_vector_ops = AsyncVectorOpsProxy("vector_ops", "ember_ml.asyncml.ops")
async_stats = AsyncStatsProxy("stats", "ember_ml.asyncml.ops")
async_linearalg = AsyncLinearAlgProxy("linearalg", "ember_ml.asyncml.ops")
async_bitwise = AsyncBitwiseProxy("bitwise", "ember_ml.asyncml.ops")

# Initialize the async proxy modules with the current backend
from ember_ml.backend import get_backend, get_backend_module
backend_name = get_backend()
backend_module = get_backend_module()
registry = BackendRegistry()
registry.set_backend(backend_name, backend_module)

# Create a dictionary mapping operation names to their async proxy modules
_ASYNC_OPS_MAPPING = {
    # Use the same mapping as the synchronous ops, but with async proxy modules
    # Math operations
    'add': async_math_ops,
    'subtract': async_math_ops,
    'multiply': async_math_ops,
    'divide': async_math_ops,
    'matmul': async_math_ops,
    'dot': async_math_ops,
    'exp': async_math_ops,
    'log': async_math_ops,
    'log10': async_math_ops,
    'log2': async_math_ops,
    'pow': async_math_ops,
    'sqrt': async_math_ops,
    'square': async_math_ops,
    'abs': async_math_ops,
    'sign': async_math_ops,
    'sin': async_math_ops,
    'cos': async_math_ops,
    'tan': async_math_ops,
    'sinh': async_math_ops,
    'cosh': async_math_ops,
    'clip': async_math_ops,
    'negative': async_math_ops,
    'mod': async_math_ops,
    'floor_divide': async_math_ops,
    'floor': async_math_ops,
    'ceil': async_math_ops,
    'gradient': async_math_ops,
    'power': async_math_ops,
    
    # Comparison operations
    'equal': async_comparison_ops,
    'not_equal': async_comparison_ops,
    'less': async_comparison_ops,
    'less_equal': async_comparison_ops,
    'greater': async_comparison_ops,
    'greater_equal': async_comparison_ops,
    'logical_and': async_comparison_ops,
    'logical_or': async_comparison_ops,
    'logical_not': async_comparison_ops,
    'logical_xor': async_comparison_ops,
    'allclose': async_comparison_ops,
    'isclose': async_comparison_ops,
    'all': async_comparison_ops,
    'any': async_comparison_ops,
    'where': async_comparison_ops,
    'isnan': async_comparison_ops,
    
    # Device operations
    'to_device': async_device_ops,
    'get_device': async_device_ops,
    'get_available_devices': async_device_ops,
    'memory_usage': async_device_ops,
    'memory_info': async_device_ops,
    'synchronize': async_device_ops,
    'set_default_device': async_device_ops,
    'get_default_device': async_device_ops,
    'is_available': async_device_ops,
    
    # IO operations
    'save': async_io_ops,
    'load': async_io_ops,
    
    # Loss operations
    'mse': async_loss_ops,
    'mean_absolute_error': async_loss_ops,
    'binary_crossentropy': async_loss_ops,
    'categorical_crossentropy': async_loss_ops,
    'sparse_categorical_crossentropy': async_loss_ops,
    'huber_loss': async_loss_ops,
    'log_cosh_loss': async_loss_ops,
    
    # Vector operations
    'normalize_vector': async_vector_ops,
    'compute_energy_stability': async_vector_ops,
    'compute_interference_strength': async_vector_ops,
    'compute_phase_coherence': async_vector_ops,
    'partial_interference': async_vector_ops,
    'euclidean_distance': async_vector_ops,
    'cosine_similarity': async_vector_ops,
    'exponential_decay': async_vector_ops,
    'fft': async_vector_ops,
    'ifft': async_vector_ops,
    'fft2': async_vector_ops,
    'ifft2': async_vector_ops,
    'fftn': async_vector_ops,
    'ifftn': async_vector_ops,
    'rfft': async_vector_ops,
    'irfft': async_vector_ops,
    'rfft2': async_vector_ops,
    'irfft2': async_vector_ops,
    'rfftn': async_vector_ops,
    'irfftn': async_vector_ops,
    
    # Stats operations
    'mean': async_stats,
    'var': async_stats,
    'median': async_stats,
    'std': async_stats,
    'percentile': async_stats,
    'max': async_stats,
    'min': async_stats,
    'sum': async_stats,
    'cumsum': async_stats,
    'argmax': async_stats,
    'sort': async_stats,
    'argsort': async_stats,
    'gaussian': async_stats,
    
    # Linear algebra operations
    'solve': async_linearalg,
    'inv': async_linearalg,
    'svd': async_linearalg,
    'eig': async_linearalg,
    'eigh': async_linearalg,
    'eigvals': async_linearalg,
    'det': async_linearalg,
    'norm': async_linearalg,
    'qr': async_linearalg,
    'cholesky': async_linearalg,
    'lstsq': async_linearalg,
    'diag': async_linearalg,
    'diagonal': async_linearalg,
    'orthogonal': async_linearalg,
    'HPC16x8': async_linearalg,
    
    # Bitwise operations
    'bitwise_and': async_bitwise,
    'bitwise_or': async_bitwise,
    'bitwise_xor': async_bitwise,
    'bitwise_not': async_bitwise,
    'left_shift': async_bitwise,
    'right_shift': async_bitwise,
    'rotate_left': async_bitwise,
    'rotate_right': async_bitwise,
    'count_ones': async_bitwise,
    'count_zeros': async_bitwise,
    'get_bit': async_bitwise,
    'set_bit': async_bitwise,
    'toggle_bit': async_bitwise,
    'binary_wave_interference': async_bitwise,
    'binary_wave_propagate': async_bitwise,
    'create_duty_cycle': async_bitwise,
    'generate_blocky_sin': async_bitwise,
    
    # Array manipulation
    'vstack': async_math_ops,
    'hstack': async_math_ops,
}

# Create a class to handle dynamic attribute access for the async ops module
class AsyncOpsModule:
    """
    Class that dynamically forwards attribute access to the appropriate async proxy module.
    
    This class is used to implement the ember_ml.asyncml.ops module, which provides
    a unified interface to asynchronous operations from the current backend.
    """
    def __init__(self):
        """Initialize the async ops module."""
        self._proxy_modules = {
            'math_ops': async_math_ops,
            'comparison_ops': async_comparison_ops,
            'device_ops': async_device_ops,
            'io_ops': async_io_ops,
            'loss_ops': async_loss_ops,
            'vector_ops': async_vector_ops,
            'stats': async_stats,
            'linearalg': async_linearalg,
            'bitwise': async_bitwise,
        }
        
        # Expose submodules directly
        self.stats = async_stats
        self.linearalg = async_linearalg
        self.bitwise = async_bitwise
        
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
        Dynamically forward attribute access to the appropriate async proxy module.
        
        Args:
            name: The name of the attribute to access
            
        Returns:
            The attribute from the appropriate async proxy module
            
        Raises:
            AttributeError: If the attribute is not found in any async proxy module
        """
        # Check if the attribute is in the async ops mapping
        if name in _ASYNC_OPS_MAPPING:
            proxy_module = _ASYNC_OPS_MAPPING[name]
            return getattr(proxy_module, name)
            
        # Check if the attribute is a proxy module
        if name in self._proxy_modules:
            return self._proxy_modules[name]
            
        # If not found, raise AttributeError
        raise AttributeError(f"'ember_ml.asyncml.ops' has no attribute '{name}'")
        
# Create an instance of the async ops module
async_ops_module = AsyncOpsModule()