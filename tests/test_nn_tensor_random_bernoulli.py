"""
Test the random_bernoulli function in the tensor module.

This module tests the random_bernoulli function to ensure it works correctly
across different backends.
"""

import pytest

from ember_ml import ops
from ember_ml.nn import tensor


@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = ops.get_backend()
    yield original
    # Ensure original is not None before setting it
    if original is not None:
        ops.set_backend(original)
    else:
        # Default to 'numpy' if original is None
        ops.set_backend('numpy')


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_random_bernoulli_shape(backend_name, original_backend):
    """Test that random_bernoulli returns a tensor with the correct shape."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Test with a simple shape
        shape = (5, 5)
        result = tensor.random_bernoulli(shape, 0.5)
        
        # Check that the result has the correct shape
        assert tensor.shape(result) == shape
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_random_bernoulli_values(backend_name, original_backend):
    """Test that random_bernoulli returns values in the correct range."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Test with a simple shape and probability
        shape = (100, 100)
        p = 0.5
        result = tensor.random_bernoulli(shape, p)
        
        # Check that all values are either 0 or 1
        zeros = ops.equal(result, tensor.convert_to_tensor(0))
        ones = ops.equal(result, tensor.convert_to_tensor(1))
        all_valid = ops.all(ops.logical_or(zeros, ones))
        assert all_valid
        
        # Check that the mean is approximately equal to p
        # (with some tolerance due to randomness)
        mean = ops.mean(tensor.convert_to_tensor(result, dtype=tensor.float32))
        assert ops.abs(mean - p) < 0.1
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "mlx"])
def test_random_bernoulli_different_p(backend_name, original_backend):
    """Test random_bernoulli with different probability values."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Test with different probability values
        shape = (100, 100)
        
        # Test with p = 0 (should be all zeros)
        result_zeros = tensor.random_bernoulli(shape, 0.0)
        all_zeros = ops.all(ops.equal(result_zeros, tensor.convert_to_tensor(0)))
        assert all_zeros
        
        # Test with p = 1 (should be all ones)
        result_ones = tensor.random_bernoulli(shape, 1.0)
        all_ones = ops.all(ops.equal(result_ones, tensor.convert_to_tensor(1)))
        assert all_ones
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")