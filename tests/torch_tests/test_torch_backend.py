# tests/torch_tests/test_backend.py
import pytest
from ember_ml.ops import set_backend, get_backend # Keep backend functions
from ember_ml import ops # Import ops module for device functions

# Note: Assumes conftest.py provides the torch_backend fixture

def test_set_and_get_backend_torch(torch_backend): # Use torch_backend fixture
    """
    Tests if get_backend retrieves the currently set PyTorch backend.
    """
    # Backend is set by torch_backend fixture
    assert get_backend() == 'torch', "Backend should be torch"

def test_get_available_devices_torch(torch_backend): # Use torch_backend fixture
    """
    Tests ops.get_available_devices with PyTorch backend.
    """
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available for torch"
    # Check for GPU availability if torch is installed
    try:
        import torch
        if torch.cuda.is_available():
             assert 'cuda:0' in available or any(x.startswith('cuda:') for x in available), "CUDA should be listed if available"
        if torch.backends.mps.is_available():
             assert 'mps' in available or 'mps:0' in available, "MPS should be listed if available" # Accommodate variations
    except ImportError:
        pass # Should have been skipped by fixture if torch not installed