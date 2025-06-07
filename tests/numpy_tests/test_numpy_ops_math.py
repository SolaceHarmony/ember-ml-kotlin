import pytest
import numpy as np # For comparison with known correct results
import math # For comparison with known correct results

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
    # set_backend("numpy")

# Test cases for ops.math functions (core ops)

def test_add():
    # Test element-wise addition
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = ops.add(a, b)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness
    assert ops.allclose(result_np, [[6.0, 8.0], [10.0, 12.0]])

    # Test broadcasting
    c = tensor.convert_to_tensor([1.0, 2.0]) # Shape (2,)
    result_broadcast = ops.add(a, c)
    assert ops.allclose(tensor.to_numpy(result_broadcast), [[2.0, 4.0], [4.0, 6.0]])

def test_subtract():
    # Test element-wise subtraction
    a = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    b = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = ops.subtract(a, b)
    assert ops.allclose(tensor.to_numpy(result), [[4.0, 4.0], [4.0, 4.0]])

def test_multiply():
    # Test element-wise multiplication
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = ops.multiply(a, b)
    assert ops.allclose(tensor.to_numpy(result), [[5.0, 12.0], [21.0, 32.0]])

def test_divide():
    # Test element-wise division
    a = tensor.convert_to_tensor([[10.0, 12.0], [14.0, 16.0]])
    b = tensor.convert_to_tensor([[2.0, 3.0], [4.0, 5.0]])
    result = ops.divide(a, b)
    assert ops.allclose(tensor.to_numpy(result), [[5.0, 4.0], [3.5, 3.2]])

def test_sin():
    # Test sine function
    x = tensor.convert_to_tensor([0.0, ops.pi / 2, ops.pi, 3 * ops.pi / 2, 2 * ops.pi])
    result = ops.sin(x)
    # Use numpy.sin for expected values
    expected_np = ops.sin(tensor.to_numpy(x))
    assert ops.allclose(tensor.to_numpy(result), expected_np)

def test_cos():
    # Test cosine function
    x = tensor.convert_to_tensor([0.0, ops.pi / 2, ops.pi, 3 * ops.pi / 2, 2 * ops.pi])
    result = ops.cos(x)
    # Use numpy.cos for expected values
    expected_np = ops.cos(tensor.to_numpy(x))
    assert ops.allclose(tensor.to_numpy(result), expected_np)

# Add more test functions for other core ops functions:
# test_floor_divide(), test_mod(), test_dot(), test_exp(), test_log(),
# test_log10(), test_log2(), test_pow(), test_sqrt(), test_square(),
# test_abs(), test_negative(), test_sign(), test_clip(), test_gradient()
# test_tan(), test_sinh(), test_cosh(), test_tanh()

# Example structure for test_pow
# def test_pow():
#     x = tensor.convert_to_tensor([[2.0, 3.0], [4.0, 5.0]])
#     y = tensor.convert_to_tensor([[2.0, 3.0], [1.0, 0.0]])
#     result = ops.pow(x, y)
#     # Use numpy.power for expected values
#     expected_np = np.power(tensor.to_numpy(x), tensor.to_numpy(y))
#     assert ops.allclose(tensor.to_numpy(result), expected_np)

# Example structure for test_abs
# def test_abs():
#     x = tensor.convert_to_tensor([[-1.0, 2.0], [-3.0, 4.0]])
#     result = ops.abs(x)
#     # Use numpy.abs for expected values
#     expected_np = ops.abs(tensor.to_numpy(x))
#     assert ops.allclose(tensor.to_numpy(result), expected_np)