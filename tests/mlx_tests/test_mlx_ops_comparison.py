import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
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

# Test cases for ops.comparison functions

def test_equal():
    # Test element-wise equality
    a = tensor.convert_to_tensor([[1, 2], [3, 4]])
    b = tensor.convert_to_tensor([[1, 5], [3, 4]])
    result = ops.equal(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert tensor.convert_to_tensor_equal(result_np, [[True, False], [True, True]])

def test_greater():
    # Test element-wise greater than
    a = tensor.convert_to_tensor([[1, 5], [3, 4]])
    b = tensor.convert_to_tensor([[1, 2], [3, 3]])
    result = ops.greater(a, b)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result), [[False, True], [False, True]])

def test_logical_and():
    # Test element-wise logical AND
    a = tensor.convert_to_tensor([[True, True], [False, False]], dtype=tensor.bool_)
    b = tensor.convert_to_tensor([[True, False], [True, False]], dtype=tensor.bool_)
    result = ops.logical_and(a, b)
    assert tensor.convert_to_tensor_equal(tensor.to_numpy(result), [[True, False], [False, False]])

# Add more test functions for other ops.comparison functions:
# test_not_equal(), test_less(), test_less_equal(), test_greater_equal(),
# test_logical_or(), test_logical_not(), test_logical_xor(), test_allclose(),
# test_isclose(), test_all(), test_where(), test_isnan()

# Example structure for test_where
# def test_where():
#     condition = tensor.convert_to_tensor([[True, False], [False, True]], dtype=tensor.bool_)
#     x = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
#     y = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
#     result = ops.where(condition, x, y)
#     assert ops.allclose(tensor.to_numpy(result), [[1.0, 6.0], [7.0, 4.0]])