import pytest
import numpy as np # For comparison with known correct results
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.utils import backend_utils # Import backend_utils
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

# Test cases for utils.backend_utils functions

def test_get_current_backend():
    # Test get_current_backend
    assert backend_utils.get_current_backend() == "torch"

def test_set_preferred_backend():
    # Test set_preferred_backend
    initial_backend = backend_utils.get_current_backend()
    try:
        # Attempt to set to a different backend and verify
        set_backend("numpy") # Temporarily switch to ensure set_preferred_backend can switch back
        preferred = backend_utils.set_preferred_backend("torch")
        assert backend_utils.get_current_backend() == "torch"
        assert preferred == "torch" # Should return the backend that was set

        # Test setting to an unavailable backend (should fall back to default or current)
        # Assuming 'invalid_backend' is not installed.
        # The behavior might be to stay on the current backend or fall back to a default.
        # Let's test that it doesn't raise an error and stays on the current backend.
        current_backend = backend_utils.get_current_backend()
        preferred_invalid = backend_utils.set_preferred_backend("invalid_backend")
        assert backend_utils.get_current_backend() == current_backend
        assert preferred_invalid == current_backend

    finally:
        # Restore original backend
        set_backend(initial_backend)
        assert backend_utils.get_current_backend() == initial_backend


def test_initialize_random_seed():
    # Test initialize_random_seed for reproducibility
    shape = (10,)
    seed1 = 123
    seed2 = 123
    seed3 = 456

    backend_utils.initialize_random_seed(seed1)
    result1 = tensor.random_uniform(shape)

    backend_utils.initialize_random_seed(seed2)
    result2 = tensor.random_uniform(shape)

    backend_utils.initialize_random_seed(seed3)
    result3 = tensor.random_uniform(shape)

    # Results with the same seed should be equal
    assert ops.allclose(result1, result2).item()
    # Results with different seeds should be different
    assert not ops.allclose(result1, result3).item()


def test_convert_to_tensor_safe():
    # Test convert_to_tensor_safe
    data_list = [[1, 2], [3, 4]]
    data_np = tensor.convert_to_tensor(data_list)
    data_tensor = tensor.convert_to_tensor(data_list) # Already an EmberTensor

    result_list = backend_utils.convert_to_tensor_safe(data_list)
    result_np_input = backend_utils.convert_to_tensor_safe(data_np)
    result_tensor_input = backend_utils.convert_to_tensor_safe(data_tensor)

    assert isinstance(result_list, tensor.EmberTensor)
    assert isinstance(result_np_input, tensor.EmberTensor)
    assert isinstance(result_tensor_input, tensor.EmberTensor)

    assert ops.allclose(result_list, tensor.convert_to_tensor(data_list)).item()
    assert ops.allclose(result_np_input, tensor.convert_to_tensor(data_np)).item()
    assert ops.allclose(result_tensor_input, data_tensor).item() # Should be the same tensor or a copy


def test_tensor_to_numpy_safe():
    # Test tensor_to_numpy_safe
    data = [[1.1, 2.2], [3.3, 4.4]]
    t_ember = tensor.convert_to_tensor(data)

    np_array = backend_utils.tensor_to_numpy_safe(t_ember)

    assert isinstance(np_array, TensorLike)
    assert ops.allclose(np_array, tensor.convert_to_tensor(data))
    assert np_array.shape == (2, 2)
    assert np_array.dtype == tensor.float32 # Assuming default float32


def test_print_backend_info(capsys):
    # Test print_backend_info (capturing stdout)
    backend_utils.print_backend_info()
    captured = capsys.readouterr()
    # Check if the output contains expected information
    assert "Current backend:" in captured.out
    assert "Torch" in captured.out # Should mention Torch as it's the set backend
    assert "Default device:" in captured.out
    assert "Test operation result:" in captured.out


# Add more test functions for other backend_utils functions:
# test_random_uniform(), test_sin_cos_transform(), test_vstack_safe(),
# test_get_backend_info()

# Note: test_random_uniform, test_sin_cos_transform, test_vstack_safe might be redundant
# if they are just wrappers around ops/tensor functions that are already tested.
# Need to check implementation details.

# Example structure for test_get_backend_info
# def test_get_backend_info():
#     info = backend_utils.get_backend_info()
#     assert isinstance(info, dict)
#     assert 'name' in info
#     assert 'device' in info
#     assert info['name'] == 'torch'
#     # Device check might be more complex depending on hardware