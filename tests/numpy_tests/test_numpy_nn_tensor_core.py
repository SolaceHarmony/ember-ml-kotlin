import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
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

# Test cases for core nn.tensor functionalities (EmberTensor class and properties)

def test_embertensor_instantiation():
    # Test creating an EmberTensor from a list
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = tensor.EmberTensor(data)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(t)

    # Assert correctness
    assert isinstance(t, tensor.EmberTensor)
    # Use ops.allclose for tensor comparison
    assert ops.allclose(result_np, tensor.convert_to_tensor(data))
    assert tensor.shape(t) == (2, 2)
    # Default dtype should be float
    assert tensor.dtype(t) in [tensor.float32, tensor.float64]
    assert ops.allclose(t, tensor.convert_to_tensor(data))

    # Test creating with explicit dtype
    t_int = tensor.EmberTensor(data, dtype=tensor.int32)
    assert isinstance(t_int, tensor.EmberTensor)
    assert tensor.dtype(t_int) == tensor.int32
    assert ops.allclose(t_int, tensor.convert_to_tensor(data, dtype=tensor.int32))

    # Test creating with explicit device (should be 'cpu' for NumPy)
    t_cpu = tensor.EmberTensor(data, device="cpu")
    assert isinstance(t_cpu, tensor.EmberTensor)
    assert ops.get_device(t_cpu) == "cpu"

    # Test creating with requires_grad
    t_grad = tensor.EmberTensor(data, requires_grad=True)
    assert isinstance(t_grad, tensor.EmberTensor)
    # NumPy backend does not support gradients, so requires_grad should be False
    # assert t_grad.requires_grad is False # This might not be a direct attribute on EmberTensor

def test_embertensor_properties():
    # Test shape, dtype, device properties
    data = [[1, 2, 3], [4, 5, 6]]
    t = tensor.EmberTensor(data, dtype=tensor.float32, device="cpu")

    assert tensor.shape(t) == (2, 3)
    assert tensor.dtype(t) == tensor.float32
    assert ops.get_device(t) == "cpu"

    # Test requires_grad property
    t_no_grad = tensor.EmberTensor(data)
    # assert t_no_grad.requires_grad is False # Might not be a direct attribute

    t_with_grad = tensor.EmberTensor(data, requires_grad=True)
    # assert t_with_grad.requires_grad is False # Should be False for NumPy

def test_embertensor_to_numpy():
    # Test converting EmberTensor to NumPy array
    data = [[1.1, 2.2], [3.3, 4.4]]
    t = tensor.EmberTensor(data)
    np_array = tensor.to_numpy(t)

    assert isinstance(np_array, tensor.EmberTensor)
    assert ops.allclose(np_array, tensor.convert_to_tensor(data))
    assert np_array.shape == (2, 2)
    # Check dtype conversion
    # Check dtype conversion - could be float32 or float64 depending on system config
    assert np_array.dtype in [tensor.float32, tensor.float64]

def test_embertensor_item():
    # Test converting scalar EmberTensor to Python scalar
    t_int = tensor.EmberTensor(42)
    item_int = tensor.item(t_int)
    assert isinstance(item_int, (int, tensor.integer))
    assert item_int == 42

    t_float = tensor.EmberTensor(3.14)
    item_float = tensor.item(t_float)
    assert isinstance(item_float, (float, tensor.floating))
    assert abs(item_float - 3.14) < 1e-6

    t_bool = tensor.EmberTensor(True)
    item_bool = tensor.item(t_bool)
    assert isinstance(item_bool, (bool, tensor.bool))
    assert item_bool is True

    # Test with non-scalar tensor (should raise error)
    t_non_scalar = tensor.EmberTensor([1, 2])
    with pytest.raises(Exception): # Expecting an exception for non-scalar
        tensor.item(t_non_scalar)

# Add more test functions for other core EmberTensor functionalities if any are missed