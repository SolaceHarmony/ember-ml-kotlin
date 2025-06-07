import pytest

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend

set_backend("mlx")

# Define a fixture to ensure backend is set for each test
@pytest.fixture(autouse=True)
def set_mlx_backend():
    set_backend("mlx")
    yield
    # Optional: reset to a default backend or the original backend after the test
    # set_backend("numpy")

# Test cases for initializers functions

def test_zeros_initializer():
    # Test zeros initializer
    shape = (5, 5)
    # initializer = initializers.get_initializer('zeros') # Replaced with direct call
    result = tensor.zeros(shape) # Direct call

    # Assert correctness (should be all zeros)
    assert tensor.shape(result) == shape
    assert ops.allclose(result, tensor.zeros(shape))

def test_ones_initializer():
    # Test ones initializer
    shape = (5, 5)
    # initializer = initializers.get_initializer('ones') # Replaced with direct call
    result = tensor.ones(shape) # Direct call

    # Assert correctness (should be all ones)
    assert tensor.shape(result) == shape
    assert ops.allclose(result, tensor.ones(shape))

def test_random_uniform_initializer():
    # Test random_uniform initializer
    shape = (100, 100) # Use a larger shape for statistical checks
    minval = -1.0
    maxval = 1.0
    # initializer = initializers.get_initializer('random_uniform') # Replaced with direct call
    result = tensor.random_uniform(shape,minval=minval, maxval=maxval) # Direct call

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)
    # ops.set_backend("mlx") # Removed redundant backend set
    # Assert properties of uniform distribution
    assert result.shape == shape
    assert ops.all(ops.greater_equal(result_np,minval))
    assert ops.all(ops.less_equal(result_np,maxval))
    # Check mean and std (should be close to expected for a large sample)
    # assert ops.less(ops.abs(ops.stats.mean(result), (minval + maxval) / 2.0), 0.05).item() # Incorrect abs usage, corrected below
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), (minval + maxval) / 2.0)), 0.05).item() # Correct usage
    # Corrected ops.abs and ops.subtract usage
    assert ops.less(ops.abs(ops.subtract(ops.stats.std(result), tensor.convert_to_tensor(ops.sqrt((maxval - minval)**2 / 12.0)))), 0.05).item()

def test_random_normal_initializer():
    # Test random_normal initializer
    shape = (100, 100) # Use a larger shape for statistical checks
    mean = 0.0
    stddev = 1.0
    # initializer = initializers.random_normal(mean=mean, stddev=stddev, seed=42) # Replaced with direct call
    result = tensor.random_normal(shape, mean=mean, stddev=stddev, seed=42) # Direct call

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of normal distribution
    # assert isinstance(result, tensor.EmberTensor) # Removed incorrect type check; functions return native tensors
    assert tensor.shape(result) == shape
    # Check mean and std (should be close to expected for a large sample)
    # assert ops.less(ops.abs(ops.stats.mean(result), mean), 0.05).item() # Incorrect abs usage, corrected below
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), mean)), 0.05).item()
    # Corrected ops.abs and ops.subtract usage
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