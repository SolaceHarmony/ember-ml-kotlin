# tests/numpy_tests/test_backend.py
import pytest
from ember_ml.ops import set_backend, get_backend # Keep backend functions
from ember_ml import ops # Import ops module for device functions

# Note: Assumes conftest.py provides the numpy_backend fixture

def test_set_and_get_backend_numpy(numpy_backend): # Use numpy_backend fixture
    """
    Tests if get_backend retrieves the currently set NumPy backend.
    """
    # Backend is set by numpy_backend fixture
    assert get_backend() == 'numpy', "Backend should be numpy"

def test_get_available_devices_numpy(numpy_backend): # Use numpy_backend fixture
    """
    Tests ops.get_available_devices with NumPy backend.
    """
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available for numpy"
    # NumPy backend typically only shows 'cpu'
    assert len(available) == 1 or 'mps' not in available, "NumPy shouldn't list GPU devices directly usually"