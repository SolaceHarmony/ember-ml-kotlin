"""
Ember ML: A backend-agnostic neural network library.

This library provides a unified interface for neural network operations
that can work with different backends (NumPy, PyTorch, MLX).
"""

import importlib
from typing import Union, Literal

# Default backend
_CURRENT_BACKEND = None
_BACKEND_MODULE = None

def set_backend(backend_name: Union[str, Literal['numpy', 'torch', 'mlx']]) -> None:
    """
    Set the backend for neural network operations.

    Args:
        backend_name: Name of the backend ('numpy', 'torch', 'mlx')
    """
    global _CURRENT_BACKEND, _BACKEND_MODULE

    if backend_name == 'torch':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.torch')
    elif backend_name == 'numpy':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.numpy')
    elif backend_name == 'mlx':
        _BACKEND_MODULE = importlib.import_module('ember_ml.backend.mlx')
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    _CURRENT_BACKEND = backend_name

    # Import all functions from the backend module into the current namespace
    for name in dir(_BACKEND_MODULE):
        if not name.startswith('_'):
            globals()[name] = getattr(_BACKEND_MODULE, name)

# Import auto_select_backend from the backend module
from ember_ml.backend import auto_select_backend

# Set default backend using auto_select_backend
# This will respect the backend configuration and choose the best available backend
try:
    backend_name, _ = auto_select_backend()
    if backend_name:
        set_backend(backend_name)
    else:
        print("Warning: No default backend could be selected. Imports may fail if backend operations are used without calling set_backend().")
except Exception as e:
    print(f"Warning: Error selecting default backend: {e}. Imports may fail if backend operations are used without calling set_backend().")
    pass # Allow import to proceed without a default backend set

# Import submodules
# from ember_ml import benchmarks # Removed - moved out of package
from ember_ml import data
from ember_ml import models
from ember_ml import nn
from ember_ml import ops
from ember_ml import training
from ember_ml import visualization
from ember_ml import wave
from ember_ml import utils

# Use lazy import for asyncml to avoid importing ray until it's needed
asyncml = None
def __getattr__(name):
    global asyncml
    if name == 'asyncml':
        if asyncml is None:
            try:
                import ember_ml.asyncml as _asyncml
                asyncml = _asyncml
            except ImportError:
                raise ImportError("Could not import ember_ml.asyncml. Make sure 'ray' is installed.")
        return asyncml
    raise AttributeError(f"module 'ember_ml' has no attribute '{name}'")
# Version of the Ember ML package
__version__ = '0.2.0'

# List of public objects exported by this module
__all__ = [
    'set_backend',
    # 'auto_select_backend', # Removed - moved to ops
    'data',
    'models',
    'nn',
    'ops',
    'training',
    'visualization',
    'wave',
    'utils',
    'asyncml',
    '__version__'
]
