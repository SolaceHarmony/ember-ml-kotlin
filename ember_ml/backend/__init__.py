"""
Backend module.

This module provides backend implementations,
including PyTorch, NumPy, and MLX.
"""

import importlib
import os
import platform
from pathlib import Path
import json
from typing import Optional, Any, Dict, List, Tuple

# Import the registry module
from ember_ml.backend.registry import BackendRegistry

# We'll use lazy imports for tensor classes to avoid circular dependencies
# Initialize variables for lazy loading
# TorchDType = None # Removed DType lazy load
# NumpyDType = None # Removed DType lazy load
# MLXDType = None # Removed DType lazy load

# Removed get_torch_dtype() function


# Removed get_numpy_dtype() function


# Removed get_mlx_dtype() function


# Available backends - will be populated based on config and successful imports
_BACKENDS = {}
_AVAILABLE_BACKENDS = [] # To store successfully loaded and enabled backends


# Path to the .ember directory in the user's home directory
EMBER_CONFIG_DIR = Path.home() / '.ember'
EMBER_BACKEND_FILE = EMBER_CONFIG_DIR / 'backend'

# Path to the backend_config.json file in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Adjust based on actual file location
BACKEND_CONFIG_FILE = PROJECT_ROOT / 'backend_config.json'


# Current backend
_CURRENT_BACKEND = None
_CURRENT_BACKEND_MODULE = None
# _backend_change_callbacks = [] # Removed Callback registry

def load_backend_config():
    """Loads the backend configuration from backend_config.json.

    Returns:
        dict: The backend configuration.
    """
    default_config = {
        "numpy": True, # NumPy is always a fallback
        "torch": True,
        "mlx": True
    }
    if BACKEND_CONFIG_FILE.exists():
        try:
            with open(BACKEND_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load backend_config.json: {e}. Using default config.")
            return default_config
    return default_config

def _initialize_backends():
    """Initializes the _BACKENDS and _AVAILABLE_BACKENDS lists based on config and imports."""
    global _BACKENDS, _AVAILABLE_BACKENDS

    config = load_backend_config()

    potential_backends = {
        'numpy': 'ember_ml.backend.numpy',
        'torch': 'ember_ml.backend.torch',
        'mlx': 'ember_ml.backend.mlx'
    }

    loaded_backends = {}
    available_and_enabled_backends = []

    for name, module_path in potential_backends.items():
        if config.get(name, False): # Check if backend is enabled in config
            try:
                importlib.import_module(module_path) # Try to import
                loaded_backends[name] = module_path
                available_and_enabled_backends.append(name)
                print(f"Successfully imported backend: {name}")
            except ImportError:
                print(f"Warning: Backend '{name}' is enabled in config but failed to import. It will not be available.")
        else:
            print(f"Info: Backend '{name}' is disabled in config.")

    _BACKENDS = loaded_backends
    _AVAILABLE_BACKENDS = available_and_enabled_backends

    # Ensure NumPy is always available as a fallback if it imported successfully
    if 'numpy' not in _BACKENDS and 'numpy' in potential_backends:
        try:
            importlib.import_module(potential_backends['numpy'])
            _BACKENDS['numpy'] = potential_backends['numpy']
            if 'numpy' not in _AVAILABLE_BACKENDS:
                 _AVAILABLE_BACKENDS.append('numpy') # Should already be there if config had it true
            print("Info: NumPy backend loaded as a fallback.")
        except ImportError:
            print("Critical Warning: NumPy backend failed to import. Ember ML might not function correctly.")


_initialize_backends() # Initialize backends when the module is loaded


def _get_backend_from_file():
    """Get the backend from the .ember/backend file."""
    if EMBER_BACKEND_FILE.exists():
        try:
            return EMBER_BACKEND_FILE.read_text().strip()
        except Exception:
            return None
    return None

def _save_backend_to_file(backend):
    """Save the backend to the .ember/backend file."""
    try:
        EMBER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        EMBER_BACKEND_FILE.write_text(backend)
    except Exception as e:
        print(f"Warning: Could not save backend preference to file: {e}")


def _reload_ops_module():
    """
    This function is kept for backward compatibility but is no longer needed.

    With the new proxy module pattern, backend changes are automatically propagated
    to all proxy modules via the BackendRegistry, so explicit reloading is not required.
    """
    # No action needed - the registry handles backend changes automatically
    pass

def get_backend():
    """Get the current backend, performing auto-selection if necessary."""
    global _CURRENT_BACKEND
    if _CURRENT_BACKEND is not None and _CURRENT_BACKEND in _AVAILABLE_BACKENDS:
        return _CURRENT_BACKEND

    # Try persisted backend first, then environment variable
    persisted_backend = _get_backend_from_file()
    env_backend = os.environ.get('EMBER_ML_BACKEND')

    if persisted_backend and persisted_backend in _AVAILABLE_BACKENDS:
        print(f"Info: Using persisted backend '{persisted_backend}'.")
        set_backend(persisted_backend)
        return _CURRENT_BACKEND

    if env_backend and env_backend in _AVAILABLE_BACKENDS:
        print(f"Info: Using EMBER_ML_BACKEND environment variable '{env_backend}'.")
        set_backend(env_backend)
        return _CURRENT_BACKEND

    # If no valid persisted or env backend, auto-select
    print("Info: No valid persisted or environment backend found. Auto-selecting backend.")
    selected_backend, _ = auto_select_backend() # auto_select_backend now considers _AVAILABLE_BACKENDS

    if selected_backend:
        set_backend(selected_backend)
    else:
        # This case should ideally not be reached if NumPy is a guaranteed fallback
        print("Critical Error: No backend could be set. Defaulting to a non-functional state.")
        _CURRENT_BACKEND = None # Explicitly set to None or a dummy backend

    return _CURRENT_BACKEND


def set_backend(backend: str):
    """Set the current backend."""
    global _CURRENT_BACKEND, _CURRENT_BACKEND_MODULE

    if backend not in _BACKENDS: # Check against successfully loaded backends
        # Try to load it if it was missed by initial load but is in config
        config = load_backend_config()
        potential_backends = {
            'numpy': 'ember_ml.backend.numpy',
            'torch': 'ember_ml.backend.torch',
            'mlx': 'ember_ml.backend.mlx'
        }
        if backend in potential_backends and config.get(backend, False):
            try:
                importlib.import_module(potential_backends[backend])
                _BACKENDS[backend] = potential_backends[backend]
                if backend not in _AVAILABLE_BACKENDS:
                    _AVAILABLE_BACKENDS.append(backend)
                print(f"Info: Late loading of backend '{backend}' successful.")
            except ImportError:
                raise ValueError(f"Backend '{backend}' is enabled but failed to import. Cannot set as current backend.")
        else:
            raise ValueError(f"Invalid backend: {backend}. Available and enabled backends: {_AVAILABLE_BACKENDS}")

    if backend == _CURRENT_BACKEND:
        return

    _CURRENT_BACKEND = backend
    _save_backend_to_file(backend)
    os.environ['EMBER_ML_BACKEND'] = backend # Keep env var in sync
    _CURRENT_BACKEND_MODULE = None  # Force reload of module on next get_backend_module()

    # Get the backend module
    backend_module = get_backend_module()

    # Update the registry
    registry = BackendRegistry()
    registry.set_backend(backend, backend_module)

    print(f"Info: Backend set to '{backend}'.")
    _reload_ops_module()


def get_backend_module():
    """Get the current backend module."""
    global _CURRENT_BACKEND_MODULE

    current_backend_name = get_backend() # Ensures backend is initialized

    if current_backend_name is None:
        # This means no backend could be loaded, not even NumPy.
        # This should be handled by get_backend() itself, but as a safeguard:
        raise RuntimeError("No backend is available or could be loaded.")

    if _CURRENT_BACKEND_MODULE is None or _CURRENT_BACKEND_MODULE.__name__ != _BACKENDS[current_backend_name]:
        try:
            _CURRENT_BACKEND_MODULE = importlib.import_module(_BACKENDS[current_backend_name])
        except ImportError:
            # This should ideally be caught by _initialize_backends or set_backend
            print(f"Critical Error: Failed to import module for backend '{current_backend_name}' which was previously available.")
            # Attempt to fall back to NumPy if possible
            if current_backend_name != 'numpy' and 'numpy' in _BACKENDS:
                print("Attempting to fall back to NumPy backend.")
                set_backend('numpy') # This will recall get_backend_module
                return _CURRENT_BACKEND_MODULE # Return the (now NumPy) module
            else:
                raise # Re-raise if NumPy also fails or was the failing one

    return _CURRENT_BACKEND_MODULE


def get_device(tensor=None):
    """
    Get the current device.

    Args:
        tensor: Optional tensor to get the device from

    Returns:
        Device name as a string
    """
    backend_name = get_backend()

    if tensor is not None and backend_name != 'mlx': # MLX handles device differently for now
        if hasattr(tensor, 'device'):
            return str(tensor.device)

    if backend_name == 'numpy':
        return 'cpu'
    elif backend_name == 'torch':
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                return 'cuda'
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        except ImportError:
            return 'cpu' # PyTorch not available
    elif backend_name == 'mlx':
        try:
            import mlx.core as mx
            return mx.default_device().type
        except ImportError:
            return 'cpu' # MLX not available

    return 'cpu' # Default fallback


def set_device(device_identifier: str):
    """
    Set the default device for the current backend.
    This is primarily for MLX, as PyTorch uses torch.set_default_device
    or tensor.to(device) and NumPy is CPU-only.

    Args:
        device_identifier: Device string (e.g., "cpu", "gpu", "mps" for torch, or specific for mlx)
    """
    backend_name = get_backend()
    if backend_name == 'mlx':
        try:
            import mlx.core as mx
            # MLX uses specific device objects, map common names
            if device_identifier.lower() == "gpu":
                mx.set_default_device(mx.gpu)  # type: ignore
            elif device_identifier.lower() == "cpu":
                mx.set_default_device(mx.cpu) # type: ignore
            else:
                # Assuming device_identifier might be a more specific MLX device string
                # This part might need more robust parsing if MLX has complex device IDs
                print(f"Info: Attempting to set MLX device to '{device_identifier}'. Exact mapping may vary.")
                # mx.set_default_device(device_identifier) # This might not be how MLX sets arbitrary devices
        except ImportError:
            print("Warning: MLX backend selected but MLX library not found for set_device.")
        except Exception as e:
            print(f"Warning: Could not set MLX device to '{device_identifier}': {e}")

    elif backend_name == 'torch':
        try:
            import torch
            # PyTorch uses torch.set_default_device or tensor.to(device)
            # This function could be a wrapper if a global default is desired here.
            # For now, it's a no-op for torch, users should manage device via torch methods.
            print("Info: For PyTorch, manage devices using tensor.to(device) or torch.set_default_device().")
        except ImportError:
             print("Warning: PyTorch backend selected but PyTorch library not found for set_device.")
    # NumPy is CPU-only, no device setting needed.


def auto_select_backend():
    """Automatically select the best *available and enabled* backend."""

    # Helper to check PyTorch capabilities safely
    def _check_torch_capability(capability_check_func):
        if 'torch' not in _AVAILABLE_BACKENDS: # Check if torch was even loaded
            return False
        try:
            import torch # Import it again, as this is a local scope
            return capability_check_func(torch)
        except ImportError: # Should not happen if _AVAILABLE_BACKENDS is correct
            return False
        except AttributeError: # For safety, if torch is incomplete
             print(f"Warning: PyTorch is missing an expected attribute during capability check.")
             return False

    if _check_torch_capability(lambda torch_module: hasattr(torch_module, 'cuda') and torch_module.cuda.is_available()):
        print("Info: Auto-selected PyTorch (CUDA) backend.")
        return 'torch', 'cuda'
    if _check_torch_capability(lambda torch_module: hasattr(torch_module, 'backends') and hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available()):
        print("Info: Auto-selected PyTorch (MPS) backend.")
        return 'torch', 'mps'
    if 'torch' in _AVAILABLE_BACKENDS: # If torch loaded but no GPU
        print("Info: Auto-selected PyTorch (CPU) backend.")
        return 'torch', 'cpu'

    if 'mlx' in _AVAILABLE_BACKENDS and platform.system() == 'Darwin': # MLX typically on Darwin
        print("Info: Auto-selected MLX backend.")
        return 'mlx', None # MLX device is handled internally by MLX

    if 'numpy' in _AVAILABLE_BACKENDS: # Fallback to NumPy
        print("Info: Auto-selected NumPy backend.")
        return 'numpy', 'cpu'

    # Should not be reached if _initialize_backends ensures NumPy is always attempted
    print("Critical Error: No available backends found, including NumPy.")
    return None, None
