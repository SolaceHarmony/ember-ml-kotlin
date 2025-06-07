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

# Test cases for nn.tensor random functions

def test_random_uniform():
    # Test tensor.random_uniform
    shape = (1000,)
    minval = 0.0
    maxval = 1.0
    result = tensor.random_uniform(shape, minval=minval, maxval=maxval, seed=42)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of uniform distribution
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.all(result_np >= minval)
    assert ops.all(result_np < maxval)
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), (minval + maxval) / 2.0)), 0.05).item()
    assert ops.less(ops.abs(ops.subtract(ops.stats.std(result), ops.sqrt((maxval - minval)**2 / 12.0))), 0.05).item()

def test_random_normal():
    # Test tensor.random_normal
    shape = (1000,)
    mean = 0.0
    stddev = 1.0
    result = tensor.random_normal(shape, mean=mean, stddev=stddev, seed=42)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of normal distribution
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    # Check mean and std (should be close to expected for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), mean)), 0.05).item()
    assert ops.less(ops.abs(ops.subtract(ops.stats.std(result), stddev)), 0.05).item()

def test_random_bernoulli():
    # Test tensor.random_bernoulli
    shape = (1000,)
    p = 0.7
    result = tensor.random_bernoulli(shape, p=p, seed=42)

    # Convert to numpy for assertion
    result_np = tensor.to_numpy(result)

    # Assert properties of Bernoulli distribution
    assert isinstance(result, tensor.EmberTensor)
    assert tensor.shape(result) == shape
    assert ops.all(ops.logical_or(result_np == 0, result_np == 1)) # Should contain only 0s and 1s
    # Check mean (should be close to p for a large sample)
    assert ops.less(ops.abs(ops.subtract(ops.stats.mean(result), p)), 0.05).item()

def test_set_seed():
    # Test tensor.set_seed for reproducibility
    shape = (10,)
    seed1 = 123
    seed2 = 123
    seed3 = 456

    tensor.set_seed(seed1)
    result1 = tensor.random_uniform(shape)

    tensor.set_seed(seed2)
    result2 = tensor.random_uniform(shape)

    tensor.set_seed(seed3)
    result3 = tensor.random_uniform(shape)

    # Results with the same seed should be equal
    assert ops.allclose(result1, result2).item()
    # Results with different seeds should be different
    assert not ops.allclose(result1, result3).item()

# Add more test functions for other random functions:
# test_random_gamma(), test_random_exponential(), test_random_poisson(),
# test_random_categorical(), test_random_permutation(), test_shuffle(), test_get_seed()

# Example structure for test_random_categorical
# def test_random_categorical():
#     logits = tensor.convert_to_tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]) # Shape (batch_size, num_classes)
#     num_samples = 1
#     result = tensor.random_categorical(logits, num_samples, seed=42) # Shape (batch_size, num_samples)
#
#     assert isinstance(result, tensor.EmberTensor)
#     assert tensor.shape(result) == (2, 1)
#     # Check that the sampled values are within the range of class indices
#     assert ops.all(ops.greater_equal(result, 0)).item()
#     assert ops.all(ops.less(result, tensor.shape(logits)[1])).item()