import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend, get_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for ops.device functions

def test_get_backend():
    # Test getting the current backend name
    current_backend = ops.get_backend()
    assert current_backend == "numpy"

def test_set_backend():
    # Test setting the backend
    initial_backend = ops.get_backend()
    try:
        # Attempt to set to a different backend and verify
        # NumPy backend can only switch to itself or potentially other installed backends
        # For a robust test, we might need to check if other backends are available
        # For now, a simple self-set test
        ops.set_backend("numpy")
        assert ops.get_backend() == "numpy"
    finally:
        # Restore original backend
        ops.set_backend(initial_backend)
        assert ops.get_backend() == initial_backend

def test_get_device():
    # Test getting the device of a tensor
    # NumPy only has 'cpu'
    x = tensor.convert_to_tensor([1.0, 2.0])
    device = ops.get_device(x)
    assert device == "cpu"

def test_get_available_devices():
    # Test getting the list of available devices
    available_devices = ops.get_available_devices()
    # NumPy should only have 'cpu'
    assert available_devices == ["cpu"]

# Add more test functions for other ops.device functions:
# test_to_device(), test_memory_usage(), test_memory_info(),
# test_synchronize(), test_set_default_device(), test_get_default_device(),
# test_is_available()

# Example structure for test_to_device (NumPy specific)
# def test_to_device():
#     x = tensor.convert_to_tensor([1.0, 2.0], device="cpu")
#     # Attempt to move to cpu (should stay on cpu)
#     y = ops.to_device(x, "cpu")
#     assert ops.get_device(y) == "cpu"
#     assert ops.allclose(tensor.to_numpy(y), tensor.to_numpy(x))
#
#     # Attempt to move to a non-cpu device (should raise an error or return cpu tensor depending on implementation)
#     # This test might need to be more specific to the expected behavior for unsupported devices
#     # with pytest.raises(ExpectedError): # Replace ExpectedError with the actual exception
#     #     ops.to_device(x, "gpu")