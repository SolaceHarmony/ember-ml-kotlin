import pytest
import numpy as np # For comparison with known correct results

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn import initializers # Import initializers module
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

# Test cases for initializers functions

def test_zeros_initializer():
    # Test zeros initializer
    shape = (5, 5)
    initializer = initializers.zeros()
    result = initializer(shape)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (should be all zeros)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.allclose(result_np, tensor.zeros(shape))

def test_ones_initializer():
    # Test ones initializer
    shape = (5, 5)
    initializer = initializers.ones()
    result = initializer(shape)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (should be all ones)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.allclose(result_np, tensor.ones(shape))

def test_constant_initializer():
    # Test constant initializer
    shape = (5, 5)
    constant_value = 3.14
    initializer = initializers.constant(constant_value)
    result = initializer(shape)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert correctness (should be filled with constant value)
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.allclose(result_np, np.full(shape, constant_value))

def test_random_uniform_initializer():
    # Test random_uniform initializer
    shape = (100, 100) # Use a larger shape for statistical checks
    minval = -1.0
    maxval = 1.0
    initializer = initializers.random_uniform(minval=minval, maxval=maxval, seed=42)
    result = initializer(shape)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of uniform distribution
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.all(result_np >= minval)
    assert ops.all(result_np < maxval)
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), (minval + maxval) / 2.0)), 0.05).item()
    assert ops.less(ops.abs(ops.subtract(ops.stats.std(result), tensor.convert_to_tensor(ops.sqrt((maxval - minval)**2 / 12.0)))), 0.05).item()

def test_random_normal_initializer():
    # Test random_normal initializer
    shape = (100, 100) # Use a larger shape for statistical checks
    mean = 0.0
    stddev = 1.0
    initializer = initializers.random_normal(mean=mean, stddev=stddev, seed=42)
    result = initializer(shape)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of normal distribution
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), mean)), 0.05).item()
    assert ops.less(ops.abs(ops.subtract(ops.stats.std(result), tensor.convert_to_tensor(stddev))), 0.05).item()

# Add more test functions for other initializers:
# test_truncated_normal_initializer(), test_variance_scaling_initializer(),
# test_glorot_uniform_initializer(), test_glorot_normal_initializer(),
# test_he_uniform_initializer(), test_he_normal_initializer(),
# test_orthogonal_initializer(), test_identity_initializer(), test_binomial_initializer()

# Example structure for test_glorot_uniform_initializer
# def test_glorot_uniform_initializer():
#     shape = (100, 100) # fan_in = 100, fan_out = 100
#     initializer = initializers.glorot_uniform(seed=42)
#     result = initializer(shape)
#
#     # Convert to numpy for assertion
#     result_np = tensor.to_numpy(result)
#
#     # Assert properties (mean close to 0, variance related to fan_in/fan_out)
#     fan_in, fan_out = shape
#     limit = ops.sqrt(6.0 / (fan_in + fan_out))
#     assert isinstance(result, tensor.EmberTensor)
#     assert tensor.shape(result) == shape
#     assert ops.all(result_np >= -limit)
#     assert ops.all(result_np <= limit)
#     assert ops.less(ops.abs(ops.stats.mean(result)), 0.05).item()
#     # Variance check might be more complex depending on exact implementation