import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import features # Import features module
from ember_ml.ops import set_backend

# Set the backend for these tests
set_backend("numpy")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_numpy_backend():
    set_backend("numpy")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("mlx")

# Test cases for nn.features stateless operations

def test_one_hot():
    # Test features.one_hot
    indices = tensor.convert_to_tensor([0, 2, 1, 0])
    depth = 3
    result = features.one_hot(indices, num_classes=depth) # Use num_classes as per docs

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (4, depth)
    expected_np = tensor.convert_to_tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert ops.allclose(result_np, expected_np)

    # Test with different dtype for indices
    indices_int64 = tensor.convert_to_tensor([0, 2, 1, 0], dtype=tensor.int64)
    result_int64 = features.one_hot(indices_int64, num_classes=depth)
    assert tensor.shape(result_int64) == (4, depth)
    assert ops.allclose(tensor.to_numpy(result_int64), expected_np)

    # Test with different output dtype
    result_int = features.one_hot(indices, num_classes=depth, dtype=tensor.int32)
    assert tensor.dtype(result_int) == tensor.int32
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result_int), expected_np.astype(tensor.int32))

    # Test with invalid indices (out of range) - should raise an error
    indices_invalid = tensor.convert_to_tensor([0, 3, 1]) # Index 3 is out of range for depth 3
    with pytest.raises(Exception): # Expecting an exception
        features.one_hot(indices_invalid, num_classes=depth)

# Add more test functions for other stateless feature operations if any exist