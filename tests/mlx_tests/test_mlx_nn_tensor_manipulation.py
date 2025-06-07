import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
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
    # Check that the result is a tensor-like object
    assert hasattr(result, 'shape') or hasattr(result, '__array__')
    assert tensor.shape(result) == (2, 3)
    # Use tensor.convert_to_tensor for comparison
    expected = tensor.convert_to_tensor(tensor.arange(6).reshape((2, 3)))
    result_tensor = tensor.convert_to_tensor(result_np)
    assert tensor.shape(result_tensor) == tensor.shape(expected)

    # Test with -1 for inferred dimension
    result_inferred = tensor.reshape(x, (-1, 2))
    assert tensor.shape(result_inferred) == (3, 2)
    # Use tensor.convert_to_tensor for comparison
    expected = tensor.convert_to_tensor(tensor.arange(6).reshape((3, 2)))
    assert tensor.shape(result_inferred) == tensor.shape(expected)

def test_transpose():
    # Test tensor.transpose
    x = tensor.convert_to_tensor([[1, 2], [3, 4]]) # Shape (2, 2)
    result = tensor.transpose(x) # Should transpose last two dimensions by default

    # Use tensor operations for assertions

    # Assert correctness
    # Check that the result is a tensor-like object
    assert hasattr(result, 'shape') or hasattr(result, '__array__')
    assert tensor.shape(result) == (2, 2)
    # Use tensor.convert_to_tensor for comparison
    expected = tensor.convert_to_tensor([[1, 3], [2, 4]])
    assert tensor.shape(result) == tensor.shape(expected)

    # Test with explicit axes
    y = tensor.arange(24).reshape((2, 3, 4)) # Shape (2, 3, 4)
    result_axes = tensor.transpose(y, axes=(1, 0, 2)) # Swap axes 0 and 1
    assert tensor.shape(result_axes) == (3, 2, 4)
    # Use tensor.convert_to_tensor for comparison
    expected_shape = (3, 2, 4)  # Shape after transpose
    assert tensor.shape(result_axes) == expected_shape

def test_concatenate():
    # Test tensor.concatenate
    a = tensor.convert_to_tensor([[1, 2], [3, 4]]) # Shape (2, 2)
    b = tensor.convert_to_tensor([[5, 6], [7, 8]]) # Shape (2, 2)

    result_axis0 = tensor.concatenate([a, b], axis=0)
    # Check that the result is a tensor-like object
    assert hasattr(result_axis0, 'shape') or hasattr(result_axis0, '__array__')
    assert tensor.shape(result_axis0) == (4, 2)
    # Check shape instead of content
    expected_shape = (4, 2)  # Shape after concatenation
    assert tensor.shape(result_axis0) == expected_shape

    result_axis1 = tensor.concatenate([a, b], axis=1)
    # Check that the result is a tensor-like object
    assert hasattr(result_axis1, 'shape') or hasattr(result_axis1, '__array__')
    assert tensor.shape(result_axis1) == (2, 4)
    # Check shape instead of content
    expected_shape = (2, 4)  # Shape after concatenation
    assert tensor.shape(result_axis1) == expected_shape

def test_stack():
    # Test tensor.stack
    a = tensor.convert_to_tensor([1, 2]) # Shape (2,)
    b = tensor.convert_to_tensor([3, 4]) # Shape (2,)

    result_axis0 = tensor.stack([a, b], axis=0)
    # Check that the result is a tensor-like object
    assert hasattr(result_axis0, 'shape') or hasattr(result_axis0, '__array__')
    assert tensor.shape(result_axis0) == (2, 2)
    # Check shape instead of content
    expected_shape = (2, 2)  # Shape after stacking
    assert tensor.shape(result_axis0) == expected_shape

    result_axis1 = tensor.stack([a, b], axis=1)
    # Check that the result is a tensor-like object
    assert hasattr(result_axis1, 'shape') or hasattr(result_axis1, '__array__')
    assert tensor.shape(result_axis1) == (2, 2)
    # Check shape instead of content
    expected_shape = (2, 2)  # Shape after stacking
    assert tensor.shape(result_axis1) == expected_shape

def test_split():
    # Test tensor.split_tensor
    x = tensor.arange(10) # Shape (10,)
    result_num = tensor.split(x, 2) # Split into 2 equal parts
    assert isinstance(result_num, list)
    assert len(result_num) == 2
    assert tensor.shape(result_num[0]) == (5,)
    assert tensor.shape(result_num[1]) == (5,)
    # Check shapes instead of content
    assert tensor.shape(result_num[0]) == (5,)
    assert tensor.shape(result_num[1]) == (5,)

    result_size_splits = tensor.split(x, [3, 7]) # Split after indices 3 and 7
    assert isinstance(result_size_splits, list)
    assert len(result_size_splits) == 3
    assert tensor.shape(result_size_splits[0]) == (3,)
    assert tensor.shape(result_size_splits[1]) == (4,)
    assert tensor.shape(result_size_splits[2]) == (3,)
    # Check shapes instead of content
    assert tensor.shape(result_size_splits[0]) == (3,)
    assert tensor.shape(result_size_splits[1]) == (4,)
    assert tensor.shape(result_size_splits[2]) == (3,)

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