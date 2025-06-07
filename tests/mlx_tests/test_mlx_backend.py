# tests/mlx_tests/test_backend.py
import pytest
import platform
from ember_ml.ops import set_backend, get_backend # Keep backend functions
from ember_ml import ops # Import ops module for device functions

# Note: Assumes conftest.py provides the mlx_backend fixture

def test_set_and_get_backend_mlx(mlx_backend): # Use mlx_backend fixture
    """
    Tests if get_backend retrieves the currently set MLX backend.
    """
    # Backend is set by mlx_backend fixture
    assert get_backend() == 'mlx', "Backend should be mlx"

def test_get_available_devices_mlx(mlx_backend): # Use mlx_backend fixture
    """
    Tests ops.get_available_devices with MLX backend.
    """
    available = ops.get_available_devices()
    assert isinstance(available, list), "Should return a list"
    assert 'cpu' in available, "CPU device should always be available for mlx"
    # MLX typically runs on Metal (MPS) on macOS
    if platform.system() == "Darwin":
         assert 'gpu' in available or 'mps' in available, "GPU/MPS device should be available for mlx on macOS"