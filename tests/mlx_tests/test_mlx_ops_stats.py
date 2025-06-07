import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.ops.stats import sum # Explicitly import sum

# Set the backend for these tests
set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for ops.stats functions

def test_mean():
    # Test mean calculation
    # Using the new pattern: from ember_ml import ops
    x = tensor.array([[1.0, 2.0], [3.0, 4.0]])
    result_all = ops.stats.mean(x)
    result_axis0 = ops.stats.mean(x, axis=0)
    result_axis1 = ops.stats.mean(x, axis=1)

    # Convert to numpy for assertion
    result_all_np = tensor.to_numpy(result_all)
    result_axis0_np = tensor.to_numpy(result_axis0)
    result_axis1_np = tensor.to_numpy(result_axis1)

    # Assert correctness (using numpy for expected values)
    assert ops.allclose(result_all_np, 2.5)
    assert ops.allclose(result_axis0_np, [2.0, 3.0])
    assert ops.allclose(result_axis1_np, [1.5, 3.5])

    # Test with different dtype using the new pattern
    x_int = tensor.array([[1, 2], [3, 4]])
    result_int = ops.stats.mean(x_int, dtype=tensor.int32)
    assert ops.allclose(tensor.to_numpy(result_int), 2)  # Integer division should result in 2, not 2.5
    
    # Test with float dtype explicitly specified
    result_float = ops.stats.mean(x_int, dtype=tensor.float32)
    assert ops.allclose(tensor.to_numpy(result_float), 2.5)  # Should be 2.5 with float dtype

def test_sum():
    # Test sum calculation
    # Using the new pattern: from ember_ml import ops
    x = tensor.array([[1.0, 2.0], [3.0, 4.0]])
    result_all = ops.stats.sum(x)
    result_axis0 = ops.stats.sum(x, axis=0)
    result_axis1 = ops.stats.sum(x, axis=1)

    # Convert to numpy for assertion
    result_all_np = tensor.to_numpy(result_all)
    result_axis0_np = tensor.to_numpy(result_axis0)
    result_axis1_np = tensor.to_numpy(result_axis1)

    # Assert correctness
    assert ops.allclose(result_all_np, 10.0)
    assert ops.allclose(result_axis0_np, [4.0, 6.0])
    assert ops.allclose(result_axis1_np, [3.0, 7.0])

    # Test with different dtype using the new pattern
    x_int = tensor.array([[1, 2], [3, 4]])
    result_int = ops.stats.sum(x_int, dtype=tensor.int32)
    assert ops.allclose(tensor.to_numpy(result_int), 10)
    
    # Test with float dtype explicitly specified
    result_float = ops.stats.sum(x_int, dtype=tensor.float32)
    assert ops.allclose(tensor.to_numpy(result_float), 10.0)

# Add more test functions for other ops.stats functions:
# test_var(), test_std(), test_min(), test_max(), test_median(),
# test_percentile(), test_cumsum(), test_argmax(), test_sort(), test_argsort()

# Example structure for test_var
# def test_var():
#     x = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
#     result_all = ops.stats.var(x)
#     result_axis0 = ops.stats.var(x, axis=0)
#     result_axis1 = ops.stats.var(x, axis=1)
#
#     result_all_np = tensor.to_numpy(result_all)
#     result_axis0_np = tensor.to_numpy(result_axis0)
#     result_axis1_np = tensor.to_numpy(result_axis1)
#
#     # Assert correctness (compare with numpy.var)
#     assert ops.allclose(result_all_np, np.var(tensor.to_numpy(x)))
#     assert ops.allclose(result_axis0_np, np.var(tensor.to_numpy(x), axis=0))
#     assert ops.allclose(result_axis1_np, np.var(tensor.to_numpy(x), axis=1))

# Example structure for test_sort
# def test_sort():
#     x = tensor.convert_to_tensor([[3, 1, 2], [6, 5, 4]])
#     result_default = ops.stats.sort(x)
#     result_axis0 = ops.stats.sort(x, axis=0)
#     result_descending = ops.stats.sort(x, descending=True)
#
#     result_default_np = tensor.to_numpy(result_default)
#     result_axis0_np = tensor.to_numpy(result_axis0)
#     result_descending_np = tensor.to_numpy(result_descending)
#
#     # Assert correctness (compare with numpy.sort)
#     assert ops.allclose(result_default_np, tensor.sort(tensor.to_numpy(x)))
#     assert ops.allclose(result_axis0_np, tensor.sort(tensor.to_numpy(x), axis=0))
#     assert ops.allclose(result_descending_np, tensor.sort(tensor.to_numpy(x), axis=-1)[::-1]) # Simple descending check