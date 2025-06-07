"""
Simple test for tensor device information.

This module provides a simple test for the device property of tensors.
"""

import pytest
from ember_ml.backend import get_backend, set_backend
from ember_ml.nn.tensor import EmberTensor


def test_tensor_device_simple():
    """Test that tensor device information is accessible."""
    # Store the original backend
    original_backend = get_backend()
    
    try:
        # Set the backend to mlx for testing
        set_backend('mlx')
        
        # Create a tensor
        t = EmberTensor([[1, 2, 3], [4, 5, 6]])
        
        # Check tensor properties
        assert t.shape == (2, 3)
        assert t.dtype is not None
        assert hasattr(t, 'device')
        
        # Print tensor information for debugging
        print(f'Tensor shape: {t.shape}, Tensor dtype: {t.dtype}, Tensor device: {t.device}')
    finally:
        # Restore the original backend
        if original_backend is not None:
            set_backend(original_backend)
        else:
            # Default to 'numpy' if original is None
            set_backend('numpy')


if __name__ == "__main__":
    test_tensor_device_simple()
    print("Test passed!")