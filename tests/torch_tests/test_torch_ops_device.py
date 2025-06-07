import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend, get_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for ops.device functions

def test_get_backend():
    # Test getting the current backend name
    current_backend = ops.get_backend()
    assert current_backend == "torch"

def test_set_backend():
    # Test setting the backend
    initial_backend = ops.get_backend()
    try:
        # Attempt to set to a different backend and verify
        ops.set_backend("numpy")
        assert ops.get_backend() == "numpy"
    finally:
        # Restore original backend
        ops.set_backend(initial_backend)
        assert ops.get_backend() == initial_backend

def test_get_device():
    # Test getting the device of a tensor
    x_cpu = tensor.convert_to_tensor([1.0, 2.0], device="cpu")
    assert ops.get_device(x_cpu) == "cpu"

    # Test with a tensor on a different device if available
    available_devices = ops.get_available_devices()
    if "cuda" in available_devices:
        x_gpu = tensor.convert_to_tensor([1.0, 2.0], device="cuda")
        assert ops.get_device(x_gpu) == "cuda"
    if "mps" in available_devices:
        x_mps = tensor.convert_to_tensor([1.0, 2.0], device="mps")
        assert ops.get_device(x_mps) == "mps"

    # Test with a tensor created without explicit device (should use default)
    y = tensor.convert_to_tensor([3.0, 4.0])
    default_device = ops.get_default_device()
    assert ops.get_device(y) == default_device

def test_get_available_devices():
    # Test getting the list of available devices
    available_devices = ops.get_available_devices()
    assert "cpu" in available_devices
    # Check for 'cuda' and 'mps' based on torch availability
    if torch.cuda.is_available():
        assert "cuda" in available_devices
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        assert "mps" in available_devices

# Add more test functions for other ops.device functions:
# test_to_device(), test_memory_usage(), test_memory_info(),
# test_synchronize(), test_set_default_device(), test_get_default_device(),
# test_is_available()

# Example structure for test_to_device
# def test_to_device():
#     x_cpu = tensor.convert_to_tensor([1.0, 2.0], device="cpu")
#     # Attempt to move to a different device if available
#     available_devices = ops.get_available_devices()
#     target_device = None
#     if "cuda" in available_devices:
#         target_device = "cuda"
#     elif "mps" in available_devices:
#         target_device = "mps"
#
#     if target_device:
#         y = ops.to_device(x_cpu, target_device)
#         assert ops.get_device(y) == target_device
#         # Ensure data is preserved
#         assert ops.allclose(tensor.to_numpy(y), tensor.to_numpy(x_cpu))
#     else:
#         pytest.skip("No alternative device available for testing to_device")