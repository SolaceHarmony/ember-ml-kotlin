"""
NumPy device operations for ember_ml.
"""

import numpy as np
from typing import Any, Optional, List, Dict
from ember_ml.backend.numpy.types import TensorLike # Use TensorLike from numpy types


# Module-level variable for default device (always 'cpu' for numpy)
_default_device = 'cpu'

def get_device(tensor: Optional[Any] = None) -> str:
    """
    Get the device ('cpu' for NumPy).

    Args:
        tensor: Ignored for NumPy backend.

    Returns:
        Always returns 'cpu'.
    """
    return 'cpu'

def set_device(device: Any) -> None:
    """
    Set the device (ignored for NumPy).

    Args:
        device: Ignored for NumPy backend.

    Raises:
        ValueError: If the device is not 'cpu'.
    """
    if str(device).lower() != 'cpu':
        raise ValueError("NumPy backend only supports 'cpu' device")
    # No actual device setting is needed for NumPy

def to_device(x: TensorLike, device: str) -> np.ndarray:
    """
    Move a tensor to the specified device (returns input for NumPy).

    Args:
        x: Input tensor (NumPy array)
        device: Target device (must be 'cpu')

    Returns:
        The original NumPy array if device is 'cpu'.

    Raises:
        ValueError: If the target device is not 'cpu'.
    """
    set_device(device) # Validate device
    # NumPy arrays are always on CPU, so just return the input
    # Ensure input is converted if needed
    
    return x




def get_available_devices() -> List[str]:
    """
    Get a list of available devices (always ['cpu'] for NumPy).

    Returns:
        List containing only 'cpu'.
    """
    return ['cpu']


def set_default_device(device: str) -> None:
    """
    Set the default device (ignored for NumPy, but validates).

    Args:
        device: Default device (must be 'cpu').

    Raises:
        ValueError: If the device is not 'cpu'.
    """
    global _default_device
    set_device(device) # Validate
    _default_device = 'cpu' # It can only ever be cpu


def get_default_device() -> str:
    """
    Get the default device (always 'cpu' for NumPy).

    Returns:
        Always returns 'cpu'.
    """
    return _default_device


def is_available(device: str) -> bool:
    """
    Check if the specified device is available ('cpu' is always available for NumPy).

    Args:
        device: Device to check.

    Returns:
        True if the device is 'cpu', False otherwise.
    """
    return str(device).lower() == 'cpu'


def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information (returns zeros for NumPy/CPU).

    Args:
        device: Ignored for NumPy backend (must be 'cpu').

    Returns:
        Dictionary with all memory values set to 0.
    """
    if device is not None:
        set_device(device) # Validate
    # Basic memory info isn't readily available for CPU via numpy alone
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}


def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information (returns zeros for NumPy/CPU).

    Args:
        device: Ignored for NumPy backend (must be 'cpu').

    Returns:
        Dictionary with all memory values set to 0.
    """
    return memory_usage(device)


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device (no-op for NumPy/CPU).

    Args:
        device: Ignored for NumPy backend (must be 'cpu').
    """
    if device is not None:
        set_device(device) # Validate
    # No synchronization needed for NumPy CPU operations
    pass

# Removed the NumpyDeviceOps class