"""
MLX device operations for ember_ml.
"""

from typing import Any, Optional, List, Dict
import mlx.core as mx
from ember_ml.backend.mlx.types import TensorLike # Use TensorLike from mlx types

# Module-level variable for default device consistency
# It's mainly set via mx.set_default_device, but we can track it
_default_device = str(mx.default_device().type).lower()

def get_device(tensor: Optional[Any] = None) -> str:
    """
    Get the current default device or the device of a given tensor.

    Args:
        tensor: Optional tensor to get the device from.

    Returns:
        Device name as a string ('cpu' or 'gpu').
    """
    if tensor is not None:
        # If a tensor is provided, try to get its device
        if hasattr(tensor, 'device'):
            # Handle both string devices and MLX device objects
            if isinstance(tensor.device, str):
                return tensor.device.lower()
            else:
                # MLX device is a DeviceType object, convert to string
                return str(tensor.device.type).lower() # e.g., 'cpu', 'gpu'

    # Return the default MLX device type as a string
    return str(mx.default_device().type).lower()

def set_device(device: Any) -> None:
    """
    Set the default device for MLX operations.

    Args:
        device: Device name as a string ('cpu', 'gpu') or an mx.Device object.
               Empty string will use GPU if available, otherwise CPU.

    Raises:
        ValueError: If the device is not valid for MLX.
    """
    global _default_device
    if isinstance(device, mx.Device):
        target_device_obj = device
        target_device_str = str(device.type).lower()
    elif isinstance(device, str):
        device_str = device.lower()
        if device_str == 'cpu':
            target_device_obj = mx.Device(mx.DeviceType.cpu)
            target_device_str = 'cpu'
        elif device_str in ['gpu', 'metal']: # Accept 'gpu' or 'metal'
            target_device_obj = mx.Device(mx.DeviceType.gpu)
            target_device_str = 'gpu'
        elif device_str == '':
            # Empty string - try GPU first, fall back to CPU
            try:
                target_device_obj = mx.Device(mx.DeviceType.gpu)
                target_device_str = 'gpu'
            except:
                target_device_obj = mx.Device(mx.DeviceType.cpu)
                target_device_str = 'cpu'
        else:
            raise ValueError(f"Invalid device string for MLX: {device}. Use 'cpu' or 'gpu'.")
    else:
        raise ValueError(f"Invalid device type for MLX: {type(device)}. Use str or mx.Device.")

    # Set the default device for subsequent MLX operations
    try:
        mx.set_default_device(target_device_obj)
        _default_device = target_device_str # Update tracked default
    except Exception as e:
        # Catch potential errors if the device isn't available
        raise ValueError(f"Failed to set MLX default device to {target_device_obj}: {e}")

def to_device(x: TensorLike, device: str) -> mx.array:
    """
    Move a tensor to the specified device (effectively a no-op for MLX default).

    Args:
        x: Input tensor (MLX array)
        device: Target device ('cpu' or 'gpu')

    Returns:
        The original MLX array (as default device handles placement).

    Raises:
        ValueError: If the target device is invalid.
    """
    # Validate the target device string, but MLX handles placement implicitly
    set_device(device) # This validates and sets the default if possible
    # Ensure input is converted if needed
    from ember_ml.backend.mlx.tensor import MLXTensor
    tensor = MLXTensor()
    x_tensor = tensor.convert_to_tensor(x)
    # In MLX, tensors generally reside on the default device.
    # Explicit movement isn't the primary mechanism like in PyTorch.
    # We return the tensor, assuming it's now on the (new) default device.
    return x_tensor

def get_available_devices() -> List[str]:
    """
    Get a list of available devices for MLX.

    Returns:
        List containing 'cpu' and potentially 'gpu'.
    """
    devices = ['cpu']
    # MLX doesn't have a direct equivalent to torch.cuda.is_available() easily accessible
    # We might need a try-except block attempting to set the GPU device
    try:
        set_device('gpu') # Try setting to GPU
        devices.append('gpu')
        set_device('cpu') # Set back to CPU if GPU was available
    except ValueError:
        pass # GPU not available or setting failed
    return devices

def set_default_device(device: str) -> None:
    """
    Set the default device for MLX operations (calls set_device).

    Args:
        device: Default device ('cpu' or 'gpu').
    """
    set_device(device) # Use the main set_device function

def get_default_device() -> str:
    """
    Get the default device for MLX operations.

    Returns:
        Default device ('cpu' or 'gpu').
    """
    # Return the tracked default or query MLX again
    # return _default_device
    return str(mx.default_device().type).lower()


def is_available(device: str) -> bool:
    """
    Check if the specified device is available for MLX.

    Args:
        device: Device to check ('cpu' or 'gpu').

    Returns:
        True if the device seems available, False otherwise.
    """
    device_str = str(device).lower()
    if device_str == 'cpu':
        return True
    elif device_str in ['gpu', 'metal']:
        # Try setting the device; if it fails, it's not available
        current_default = get_default_device()
        try:
            set_device(device_str)
            set_device(current_default) # Set back
            return True
        except ValueError:
            return False
    return False

def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information (limited info available for MLX).

    Args:
        device: Target device ('cpu' or 'gpu').

    Returns:
        Dictionary with memory usage (currently returns zeros).
    """
    # MLX doesn't provide detailed memory usage APIs like PyTorch CUDA yet
    if device is not None:
        set_device(device) # Validate
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0} # Placeholder

def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information (limited info available for MLX).

    Args:
        device: Target device ('cpu' or 'gpu').

    Returns:
        Dictionary with memory information (currently returns zeros).
    """
    return memory_usage(device) # Same placeholder


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device (uses mx.eval).

    Args:
        device: Target device ('cpu' or 'gpu').
    """
    # MLX uses lazy evaluation. mx.eval() forces computation.
    # It doesn't target a specific device like torch.cuda.synchronize.
    if device is not None:
        set_device(device) # Validate device, though eval isn't device specific
    mx.eval() # Evaluate all pending computations

# Removed the MLXDeviceOps class