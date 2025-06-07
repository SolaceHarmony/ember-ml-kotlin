import pytest
import numpy as np # For comparison with known correct results
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

# Test cases for nn.tensor manipulation functions

def test_reshape():
    # Test tensor.reshape
    x = tensor.arange(6) # Shape (6,)
    result = tensor.reshape(x, (2, 3))

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (2, 3)
    assert tensor.convert_to_tensor_equal(result_np, tensor.arange(6).reshape((2, 3)))

    # Test with -1 for inferred dimension
    result_inferred = tensor.reshape(x, (-1, 2))
    assert tensor.shape(result_inferred) == (3, 2)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_inferred), tensor.arange(6).reshape((3, 2)))

def test_transpose():
    # Test tensor.transpose
    x = tensor.convert_to_tensor([[1, 2], [3, 4]]) # Shape (2, 2)
    result = tensor.transpose(x) # Should transpose last two dimensions by default

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (2, 2)
    assert tensor.convert_to_tensor_equal(result_np, tensor.convert_to_tensor([[1, 3], [2, 4]]))

    # Test with explicit axes
    y = tensor.arange(24).reshape((2, 3, 4)) # Shape (2, 3, 4)
    result_axes = tensor.transpose(y, axes=(1, 0, 2)) # Swap axes 0 and 1
    assert tensor.shape(result_axes) == (3, 2, 4)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_axes), np.transpose(tensor.arange(24).reshape((2, 3, 4)), axes=(1, 0, 2)))

def test_concatenate():
    # Test tensor.concatenate
    a = tensor.convert_to_tensor([[1, 2], [3, 4]]) # Shape (2, 2)
    b = tensor.convert_to_tensor([[5, 6], [7, 8]]) # Shape (2, 2)

    result_axis0 = tensor.concatenate([a, b], axis=0)
    assert isinstance(result_axis0, tensor.EmberTensor)
    assert tensor.shape(result_axis0) == (4, 2)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_axis0), np.concatenate([tensor.to_numpy(a), tensor.to_numpy(b)], axis=0))

    result_axis1 = tensor.concatenate([a, b], axis=1)
    assert isinstance(result_axis1, tensor.EmberTensor)
    assert tensor.shape(result_axis1) == (2, 4)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_axis1), np.concatenate([tensor.to_numpy(a), tensor.to_numpy(b)], axis=1))

def test_stack():
    # Test tensor.stack
    a = tensor.convert_to_tensor([1, 2]) # Shape (2,)
    b = tensor.convert_to_tensor([3, 4]) # Shape (2,)

    result_axis0 = tensor.stack([a, b], axis=0)
    assert isinstance(result_axis0, tensor.EmberTensor)
    assert tensor.shape(result_axis0) == (2, 2)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_axis0), tensor.stack([tensor.to_numpy(a), tensor.to_numpy(b)], axis=0))

    result_axis1 = tensor.stack([a, b], axis=1)
    assert isinstance(result_axis1, tensor.EmberTensor)
    assert tensor.shape(result_axis1) == (2, 2)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_axis1), tensor.stack([tensor.to_numpy(a), tensor.to_numpy(b)], axis=1))

def test_split():
    # Test tensor.split
    x = tensor.arange(10) # Shape (10,)
    result_num = tensor.split(x, 2) # Split into 2 equal parts
    assert isinstance(result_num, list)
    assert len(result_num) == 2
    assert tensor.shape(result_num[0]) == (5,)
    assert tensor.shape(result_num[1]) == (5,)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_num[0]), tensor.arange(5))
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_num[1]), tensor.arange(5, 10))

    result_size_splits = tensor.split(x, [3, 7]) # Split after indices 3 and 7
    assert isinstance(result_size_splits, list)
    assert len(result_size_splits) == 3
    assert tensor.shape(result_size_splits[0]) == (3,)
    assert tensor.shape(result_size_splits[1]) == (4,)
    assert tensor.shape(result_size_splits[2]) == (3,)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_size_splits[0]), tensor.arange(3))
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_size_splits[1]), tensor.arange(3, 7))
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_size_splits[2]), tensor.arange(7, 10))

# Add more test functions for other manipulation functions:
# test_expand_dims(), test_squeeze(), test_tile()

# Example structure for test_expand_dims
# def test_expand_dims():
#     x = tensor.arange(5) # Shape (5,)
#     result = tensor.expand_dims(x, axis=0) # Add new dimension at axis 0
#     assert isinstance(result, tensor.EmberTensor)
#     assert tensor.shape(result) == (1, 5)
#     assert tensor.convert_to_tensor_equal(tensor.to_numpy(result), tensor.arange(5)[np.newaxis, :])
#
#     result_middle = tensor.expand_dims(x, axis=1) # Add new dimension at axis 1
#     assert tensor.shape(result_middle) == (5, 1)
#     assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_middle), tensor.arange(5)[:, np.newaxis])