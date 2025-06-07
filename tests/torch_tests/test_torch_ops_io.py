import pytest
import numpy as np # For comparison with known correct results
import os
import torch # Import torch for device checks if needed

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("torch")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_torch_backend():
    set_backend("torch")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Define a fixture for cleanup
@pytest.fixture
def cleanup_file():
    file_path = "test_ops_io_file.pkl" # Use a consistent test file name
    yield file_path
    # Clean up the created file after the test
    if os.path.exists(file_path):
        os.remove(file_path)

# Test cases for ops.io functions

def test_save_and_load_tensor(cleanup_file):
    # Test saving and loading a tensor
    original_tensor = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    file_path = cleanup_file

    # Save the tensor
    ops.save(original_tensor, file_path)

    # Load the tensor
    loaded_tensor = ops.load(file_path)

    # Convert to numpy for assertion
    original_np = tensor.to_numpy(original_tensor)
    loaded_np = tensor.to_numpy(loaded_tensor)

    # Assert correctness
    assert ops.allclose(original_np, loaded_np)
    # Also check dtype and shape
    assert original_tensor.dtype == loaded_tensor.dtype
    assert original_tensor.shape == loaded_tensor.shape

# Add more test functions for other ops.io scenarios:
# test_save_and_load_dict_of_tensors(), test_save_and_load_model_state(),
# test_save_and_load_with_different_dtypes(), test_save_and_load_empty_tensor()

# Example structure for test_save_and_load_dict_of_tensors
# def test_save_and_load_dict_of_tensors(cleanup_file):
#     # Test saving and loading a dictionary of tensors
#     original_dict = {
#         "tensor1": tensor.convert_to_tensor([1, 2, 3]),
#         "tensor2": tensor.convert_to_tensor([[4.0, 5.0], [6.0, 7.0]])
#     }
#     file_path = cleanup_file
#
#     # Save the dictionary
#     ops.save(original_dict, file_path)
#
#     # Load the dictionary
#     loaded_dict = ops.load(file_path)
#
#     # Assert correctness
#     assert isinstance(loaded_dict, dict)
#     assert set(original_dict.keys()) == set(loaded_dict.keys())
#
#     for key in original_dict:
#         original_np = tensor.to_numpy(original_dict[key])
#         loaded_np = tensor.to_numpy(loaded_dict[key])
#         assert ops.allclose(original_np, loaded_np)
#         assert original_dict[key].dtype == loaded_dict[key].dtype
#         assert original_dict[key].shape == loaded_dict[key].shape