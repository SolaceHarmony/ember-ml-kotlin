"""
PyTorch device operations for ember_ml.
"""

import torch
from typing import Any, Optional, List, Dict # Import List and Dict for type hints
from ember_ml.backend.torch.types import TensorLike, default_int # Use TensorLike instead of ArrayLike
# Module-level variable to store the pseudo-default device
# Initialize by determining the best available device

_default_device = 'cpu' # Default to CPU
try:
    if torch.cuda.is_available():
        _default_device = 'cuda'
except AttributeError:
    pass # CUDA not available or torch.cuda doesn't have is_available

try:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _default_device = 'mps'
except AttributeError:
    pass # MPS not available or torch.backends doesn't have mps


# Removed TorchDeviceOps class as it's redundant with standalone functions

def to_device(x: TensorLike, device: str) -> Any: # Changed torch.Tensor to Any
    """
    Move a tensor to the specified device.
    
    Args:
        x: Input tensor
        device: Target device
        
    Returns:
        Tensor on the target device
    """
    from ember_ml.backend.torch.tensor import convert_to_tensor # Lazy load
    # Check if the device is available
    if not is_available(device):
        raise ValueError(f"Device {device} is not available.")
    # Convert the input to a tensor if it's not already
    if not isinstance(x, torch.Tensor):
        # Use convert_to_tensor to ensure the input is a tensor
        from ember_ml.backend.torch.tensor import TorchTensor
        x_tensor = TorchTensor().convert_to_tensor(data=x)
    else:
        # If it's already a tensor, just use it
        x_tensor = x
    # Move the tensor to the specified device
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                from ember_ml.backend.torch.tensor import TorchTensor
                device_idx_tensor = TorchTensor().convert_to_tensor(device_idx_str)
                device_idx = device_idx_tensor.to(torch.int32).item()
            x_tensor = x_tensor.to(device_idx)
        else:
            raise ValueError("CUDA is not available.")
    elif device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            x_tensor = x_tensor.to(device)
        else:
            raise ValueError("MPS is not available.")
    elif device == 'cpu':
        x_tensor = x_tensor.to(device)
    else:
        raise ValueError(f"Invalid device: {device}")
    # Return the tensor on the specified device
    # Use convert_to_tensor to ensure the input is a tensor
    x_tensor = convert_to_tensor(x)
    return x_tensor.to(device)


def get_device(tensor: Optional[Any] = None) -> str: # Changed x to tensor, make optional
    """
    Get the current default device or the device of a given tensor.

    Args:
        tensor: Optional tensor to get the device from. If None, returns the default device.

    Returns:
        Device name as a string (e.g., 'cpu', 'cuda', 'mps').
    """
    if tensor is not None:
        # If a tensor is provided, get its device
        from ember_ml.backend.torch.tensor import TorchTensor # Lazy load
        x_tensor = tensor
        # Ensure it's a torch tensor before accessing .device
        if not isinstance(x_tensor, torch.Tensor):
            # Attempt conversion if not already a tensor
             x_tensor = TorchTensor().convert_to_tensor(data=tensor)

        if hasattr(x_tensor, 'device'):
            return str(x_tensor.device)
        else:
             # Fallback if it's not a tensor or doesn't have device attr
             # (Should not happen with proper conversion)
             return get_default_device()
    else:
        # If no tensor is provided, return the default device
        return get_default_device()


def get_available_devices() -> List[str]:
    """
    Get a list of available devices.
    
    Returns:
        List of available devices
    """
    devices = ['cpu']
    try:
        if torch.cuda.is_available():
            devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    except AttributeError:
        pass # CUDA not available
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
    except AttributeError:
        pass # MPS not available
    return devices


def set_default_device(device: str) -> None:
    """
    Set the default device for PyTorch operations.
    
    Args:
        device: Default device
    """
    global _default_device
    _default_device = device
    
    # Set the default device for PyTorch
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            device_idx = 0
            if ':' in device:
                # Use convert_to_tensor and cast instead of int()
                device_idx_str = device.split(':')[1]
                from ember_ml.backend.torch.tensor import TorchTensor # Lazy load
                device_idx_tensor = TorchTensor().convert_to_tensor(device_idx_str)
                device_idx = device_idx_tensor.to(torch.int32).item()
            torch.cuda.set_device(device_idx)


def get_default_device() -> str:
    """
    Get the default device for PyTorch operations.
    
    Returns:
        Default device
    """
    return _default_device


def is_available(device: str) -> bool:
    """
    Check if the specified device is available.
    
    Args:
        device: Device to check
        
    Returns:
        True if the device is available, False otherwise
    """
    if device == 'cpu':
        return True
    elif device.startswith('cuda'):
        try:
            return torch.cuda.is_available()
        except AttributeError:
            return False # CUDA not available
    elif device == 'mps':
        try:
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except AttributeError:
            return False # MPS not available
    return False


def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory usage information
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        try:
            if torch.cuda.is_available():
                device_idx = 0
                if ':' in device:
                    # Use convert_to_tensor and cast instead of int()
                    device_idx_str = device.split(':')[1]
                    from ember_ml.backend.torch.tensor import TorchTensor
                    # Use convert_to_tensor to ensure the input is a tensor
                    # and cast to int32
                    device_idx_tensor = TorchTensor().convert_to_tensor(device_idx_str)
                    device_idx = device_idx_tensor.to(torch.int32).item()

                # Get memory information
                allocated = torch.cuda.memory_allocated(device_idx)
                reserved = torch.cuda.memory_reserved(device_idx)

                # Get total memory
                total = torch.cuda.get_device_properties(device_idx).total_memory

                # Calculate free memory using torch.subtract instead of direct subtraction
                free = torch.subtract(torch.tensor(total), torch.tensor(reserved)).item()

                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'free': free,
                    'total': total
                }
        except AttributeError:
            pass # CUDA not available
    
    # For CPU or other devices, return zeros
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}


def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory information
    """
    return memory_usage(device)


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.
    
    Args:
        device: Target device
    """
    if device is None:
        device = _default_device
        
    if device.startswith('cuda'):
        try:
            if torch.cuda.is_available():
                device_idx = 0
                if ':' in device:
                    # Use convert_to_tensor and cast instead of int()
                    device_idx_str = device.split(':')[1]
                    from ember_ml.backend.torch.tensor import TorchTensor # Lazy load
                    device_idx_tensor = TorchTensor().convert_to_tensor(device_idx_str)
                    device_idx = device_idx_tensor.to(default_int).item()
                torch.cuda.synchronize(device_idx)
        except AttributeError:
            pass # CUDA not available

# Removed duplicate function definitions for get_device and set_device