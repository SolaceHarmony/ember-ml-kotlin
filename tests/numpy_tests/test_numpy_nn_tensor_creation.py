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

# Test cases for nn.tensor creation functions

def test_array():
    # Test tensor.array creation
    data = [[1, 2], [3, 4]]
    result = tensor.array(data)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert isinstance(result, tensor.EmberTensor)
    assert ops.allclose(result_np, tensor.convert_to_tensor(data))
    assert tensor.shape(result) == (2, 2)
    # Default dtype should be inferred (likely int or float depending on backend default)
    # We can check if it's one of the expected types
    assert tensor.dtype(result) in [tensor.int32, tensor.int64]

    # Test with explicit dtype
    result_float = tensor.array(data, dtype=tensor.float32)
    assert tensor.dtype(result_float) == tensor.float32
    assert ops.allclose(tensor.to_numpy(result_float), tensor.convert_to_tensor(data, dtype=tensor.float32))

def test_convert_to_tensor():
    # Test tensor.convert_to_tensor
    data_list = [[1, 2], [3, 4]]
    data_np = tensor.convert_to_tensor(data_list)

    result_list = tensor.convert_to_tensor(data_list)
    result_np_input = tensor.convert_to_tensor(data_np)

    assert isinstance(result_list, tensor.EmberTensor)
    assert isinstance(result_np_input, tensor.EmberTensor)
    assert ops.allclose(tensor.to_numpy(result_list), tensor.convert_to_tensor(data_list))
    assert ops.allclose(tensor.to_numpy(result_np_input), data_np)
    assert tensor.shape(result_list) == (2, 2)
    assert tensor.shape(result_np_input) == (2, 2)

    # Test with explicit dtype
    result_float = tensor.convert_to_tensor(data_list, dtype=tensor.float32)
    assert tensor.dtype(result_float) == tensor.float32
    assert ops.allclose(tensor.to_numpy(result_float), tensor.convert_to_tensor(data_list, dtype=tensor.float32))

def test_zeros():
    # Test tensor.zeros
    shape = (2, 3)
    # Get the raw NumPy array
    raw_result = tensor.zeros(shape)
    # Wrap it in an EmberTensor
    result = tensor.EmberTensor(raw_result)

    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.allclose(tensor.to_numpy(result), tensor.zeros(shape))
    # Default dtype should be float
    assert tensor.dtype(result) in [tensor.float32, tensor.float64]

    # Test with explicit dtype
    result_int = tensor.zeros(shape, dtype=tensor.int32)
    assert tensor.dtype(result_int) == tensor.int32
    assert ops.allclose(tensor.to_numpy(result_int), tensor.zeros(shape, dtype=tensor.int32))

def test_ones():
    # Test tensor.ones
    shape = (3, 2)
    # Get the raw NumPy array
    raw_result = tensor.ones(shape)
    # Wrap it in an EmberTensor
    result = tensor.EmberTensor(raw_result)

    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.allclose(tensor.to_numpy(result), tensor.ones(shape))
    # Default dtype should be float
    assert tensor.dtype(result) in [tensor.float32, tensor.float64]

    # Test with explicit dtype
    result_int = tensor.ones(shape, dtype=tensor.int32)
    assert tensor.dtype(result_int) == tensor.int32
    assert ops.allclose(tensor.to_numpy(result_int), tensor.ones(shape, dtype=tensor.int32))

def test_arange():
    # Test tensor.arange
    # Get the raw NumPy array
    raw_result = tensor.arange(5)
    # Wrap it in an EmberTensor
    result = tensor.EmberTensor(raw_result)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (5,)
    assert ops.allclose(tensor.to_numpy(result), tensor.convert_to_tensor(tensor.arange(5)))

    # Get the raw NumPy array
    raw_result_start_stop = tensor.arange(2, 7)
    # Wrap it in an EmberTensor
    result_start_stop = tensor.EmberTensor(raw_result_start_stop)
    assert tensor.shape(result_start_stop) == (5,)
    assert ops.allclose(tensor.to_numpy(result_start_stop), tensor.convert_to_tensor(tensor.arange(2, 7)))

    # Get the raw NumPy array
    raw_result_start_stop_step = tensor.arange(1, 10, 2)
    # Wrap it in an EmberTensor
    result_start_stop_step = tensor.EmberTensor(raw_result_start_stop_step)
    assert tensor.shape(result_start_stop_step) == (5,)
    assert ops.allclose(tensor.to_numpy(result_start_stop_step), tensor.convert_to_tensor(tensor.arange(1, 10, 2)))

    # Test with float step
    # Get the raw NumPy array
    raw_result_float_step = tensor.arange(1.0, 5.0, 0.5)
    # Wrap it in an EmberTensor
    result_float_step = tensor.EmberTensor(raw_result_float_step)
    assert ops.allclose(tensor.to_numpy(result_float_step), tensor.arange(1.0, 5.0, 0.5))

def test_linspace():
    # Test tensor.linspace
    # Get the raw NumPy array
    raw_result = tensor.linspace(0.0, 1.0, 5)
    # Wrap it in an EmberTensor
    result = tensor.EmberTensor(raw_result)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == (5,)
    assert ops.allclose(tensor.to_numpy(result), tensor.linspace(0.0, 1.0, 5))

    # Get the raw NumPy array
    raw_result_int = tensor.linspace(0, 10, 3, dtype=tensor.int32)
    # Wrap it in an EmberTensor
    result_int = tensor.EmberTensor(raw_result_int)
    assert tensor.dtype(result_int) == tensor.int32
    assert ops.allclose(tensor.to_numpy(result_int), tensor.linspace(0, 10, 3, dtype=tensor.int32))

# Add more test functions for other creation functions:
# test_eye(), test_zeros_like(), test_ones_like(), test_full(), test_full_like()

# Example structure for test_eye
# def test_eye():
#     n = 3
#     result = tensor.eye(n)
#     assert isinstance(result, tensor.EmberTensor)
#     assert tensor.shape(result) == (n, n)
#     assert ops.allclose(tensor.to_numpy(result), tensor.convert_to_tensor(ops.eye(n)))
#
#     m = 4
#     result_nm = tensor.eye(n, m)
#     assert tensor.shape(result_nm) == (n, m)
#     assert ops.allclose(tensor.to_numpy(result_nm), tensor.convert_to_tensor(ops.eye(n, m)))