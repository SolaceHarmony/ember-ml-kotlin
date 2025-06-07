"""
Activation function modules and operations for ember_ml's neural network library.

This module provides the activation Module classes (e.g., ReLU, Sigmoid)
and dynamically aliases the corresponding functional operations from the active backend
upon import.
"""
import importlib
import sys
import os
from typing import List, Optional, Callable, Any

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# Import Module base class and activation module classes using absolute paths
# Import Module needs to be fixed later - potential circular dependency
# from ember_ml.nn.modules import Module # Causes ImportError
from ember_ml.nn.modules.activations.relu_module import ReLU
from ember_ml.nn.modules.activations.sigmoid_module import Sigmoid
from ember_ml.nn.modules.activations.tanh_module import Tanh
from ember_ml.nn.modules.activations.softmax_module import Softmax
from ember_ml.nn.modules.activations.softplus_module import Softplus
from ember_ml.nn.modules.activations.lecun_tanh_module import LeCunTanh
from ember_ml.nn.modules.activations.dropout_module import Dropout # Assuming Dropout is also considered an activation module here


# --- Dynamic Backend Function Aliasing ---

# Master list of activation functions expected to be aliased *in this module*
_ACTIVATION_OPS_LIST = [
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
    'softplus',
    'lecun_tanh', 
]

# Placeholder initialization
for _op_name in _ACTIVATION_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track if aliases have been set for the current backend
_aliased_backend_activations: Optional[str] = None

def get_activations_module():
    """Imports the activation functions from the active backend module."""
    # This function is not used in this module but can be used for testing purposes
    # or to ensure that the backend module is imported correctly.
    # Reload the backend module to ensure the latest version is use
    backend_name = get_backend()
    module_name = get_backend_module().__name__ + '.activations'
    module = importlib.import_module(module_name)
    return module

def _update_activation_aliases():
    """Dynamically updates this module's namespace with backend activation functions."""
    global _aliased_backend_activations
    backend_name = get_backend()

    # Avoid re-aliasing if backend hasn't changed since last update for this module
    if backend_name == _aliased_backend_activations:
        return

    backend_module = get_activations_module()
    current_module = sys.modules[__name__]
    missing_ops = []

    for func_name in _ACTIVATION_OPS_LIST:
        try:
            backend_activations = get_activations_module()

            backend_function = getattr(backend_activations, func_name)
            setattr(current_module, func_name, backend_function)
            globals()[func_name] = backend_function # Update globals too
        except AttributeError:
            setattr(current_module, func_name, None)
            globals()[func_name] = None
            missing_ops.append(func_name)

    if missing_ops:
        # Suppress warning here as ops/__init__ will also warn
        # print(f"Warning: Backend '{backend_name}' does not implement the following activation ops: {', '.join(missing_ops)}")
        pass
    _aliased_backend_activations = backend_name

# --- Initial alias setup ---
# Populate aliases when this module is first imported.
_update_activation_aliases()


def get_activation(name: str) -> Callable:
    """
    Retrieves an activation function by name from the currently aliased backend functions.

    Args:
        name: The string name of the activation function (e.g., 'relu', 'tanh').

    Returns:
        The corresponding activation function callable.

    Raises:
        AttributeError: If the activation function name is not found in the
                       currently aliased functions for the active backend, or if
                       the retrieved attribute is not callable.
    """
    current_module = sys.modules[__name__]
    try:
        # Ensure aliases are up-to-date before lookup
        _update_activation_aliases()
        func = getattr(current_module, name)
        if func is None:
             raise AttributeError(f"Activation function '{name}' is not implemented or aliased by the current backend '{get_backend()}'.")
        if not callable(func):
            # This shouldn't happen if aliasing works correctly, but good sanity check
            raise TypeError(f"Retrieved attribute '{name}' for backend '{get_backend()}' is not callable.")
        return func
    except AttributeError:
        raise AttributeError(f"Activation function '{name}' not found or not aliased in {__name__} for backend '{get_backend()}'.") from None


# --- Define __all__ ---
__all__ = [
    # Module Classes
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'Softplus',
    'LeCunTanh',
    'Dropout',
    # Aliased Functions (matching _ACTIVATION_OPS_LIST)
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
    'softplus',
    'lecun_tanh', # Add to __all__ as well
    'get_activation',
]
