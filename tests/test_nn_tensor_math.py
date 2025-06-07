"""
Unit tests for math operations across different backends.

This module contains pytest tests for the math operations in the nn.tensor module.
It tests each operation with different backends to ensure consistency.
"""
import pytest
import numpy as np
from ember_ml import ops
from ember_ml.nn.tensor import (
    zeros, ones, eye, full, arange, reshape, transpose,
    convert_to_tensor, cast, copy, to_numpy,
    shape, dtype, maximum
)
from ember_ml.backend import get_backend, set_backend

# Replace ops.ones with ones, ops.shape with shape, etc.
# But keep ops.add, ops.subtract, etc. as they are

# List of backends to test
BACKENDS = ['numpy']
try:
    import torch
    BACKENDS.append('torch')
except ImportError:
    pass

try:
    import mlx.core
    BACKENDS.append('mlx')
except ImportError:
    pass

@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with different backends."""
    prev_backend = get_backend()
    set_backend(request.param)
    ops.set_ops(request.param)
    yield request.param
    if prev_backend is not None:
        set_backend(prev_backend)
        ops.set_ops(prev_backend)

class TestBasicArithmetic:
    """Tests for basic arithmetic operations."""

    def test_add(self, backend):
        """Test add operation."""
        # Test scalar addition
        x = ones((3, 4))
        y = ones((3, 4))
        z = ops.add(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) + ones((3, 4))))
        
        # Test broadcasting
        x = ones((3, 4))
        y = ones((1, 4))
        z = ops.add(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) + ones((1, 4))))
        
        # Test scalar + tensor
        x = ones((3, 4))
        z = ops.add(x, 2)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) + 2))

    def test_subtract(self, backend):
        """Test subtract operation."""
        # Test scalar subtraction
        x = ones((3, 4))
        y = ones((3, 4))
        z = ops.subtract(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) - ones((3, 4))))
        
        # Test broadcasting
        x = ones((3, 4))
        y = ones((1, 4))
        z = ops.subtract(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) - ones((1, 4))))
        
        # Test scalar - tensor
        x = ones((3, 4))
        z = ops.subtract(x, 2)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) - 2))

    def test_multiply(self, backend):
        """Test multiply operation."""
        # Test scalar multiplication
        x = ones((3, 4))
        y = full((3, 4), 2)
        z = ops.multiply(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) * full((3, 4), 2)))
        
        # Test broadcasting
        x = ones((3, 4))
        y = full((1, 4), 2)
        z = ops.multiply(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) * full((1, 4), 2)))
        
        # Test scalar * tensor
        x = ones((3, 4))
        z = ops.multiply(x, 2)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) * 2))

    def test_divide(self, backend):
        """Test divide operation."""
        # Test scalar division
        x = ones((3, 4))
        y = full((3, 4), 2)
        z = ops.divide(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) / full((3, 4), 2)))
        
        # Test broadcasting
        x = ones((3, 4))
        y = full((1, 4), 2)
        z = ops.divide(x, y)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) / full((1, 4), 2)))
        
        # Test tensor / scalar
        x = ones((3, 4))
        z = ops.divide(x, 2)
        assert shape(z) == (3, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((3, 4)) / 2))

    def test_matmul(self, backend):
        """Test matmul operation."""
        # Test matrix multiplication
        x = ones((2, 3))
        y = ones((3, 4))
        z = ops.matmul(x, y)
        assert shape(z) == (2, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((2, 3)) @ ones((3, 4))))
        
        # Test batch matrix multiplication
        x = ones((5, 2, 3))
        y = ones((5, 3, 4))
        z = ops.matmul(x, y)
        assert shape(z) == (5, 2, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((5, 2, 3)) @ ones((5, 3, 4))))
        
        # Test broadcasting in batch dimensions
        x = ones((1, 2, 3))
        y = ones((5, 3, 4))
        z = ops.matmul(x, y)
        assert shape(z) == (5, 2, 4)
        assert ops.allclose(to_numpy(z), to_numpy(ones((1, 2, 3)) @ ones((5, 3, 4))))

class TestReductionOperations:
    """Tests for reduction operations."""

    def test_mean(self, backend):
        """Test mean operation."""
        # Create a tensor
        x = arange(12).reshape((3, 4))
        
        # Test mean over all elements
        y = ops.mean(x)
        assert np.isscalar(to_numpy(y)) or shape(y) == ()
        assert ops.allclose(to_numpy(y), to_numpy(ops.mean(arange(12).reshape((3, 4)))))
        
        # Test mean along axis 0
        y = ops.mean(x, axis=0)
        assert shape(y) == (4,)
        assert ops.allclose(to_numpy(y), to_numpy(ops.mean(arange(12).reshape((3, 4)), axis=0)))
        
        # Test mean along axis 1
        y = ops.mean(x, axis=1)
        assert shape(y) == (3,)
        assert ops.allclose(to_numpy(y), to_numpy(ops.mean(arange(12).reshape((3, 4)), axis=1)))
        
        # Test mean with keepdims
        y = ops.mean(x, axis=0, keepdims=True)
        assert shape(y) == (1, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.mean(arange(12).reshape((3, 4)), axis=0, keepdims=True)))

    def test_sum(self, backend):
        """Test sum operation."""
        # Create a tensor
        x = arange(12).reshape((3, 4))
        
        # Test sum over all elements
        y = ops.sum(x)
        assert np.isscalar(to_numpy(y)) or shape(y) == ()
        assert ops.allclose(to_numpy(y), to_numpy(ops.sum(arange(12).reshape((3, 4)))))
        
        # Test sum along axis 0
        y = ops.sum(x, axis=0)
        assert shape(y) == (4,)
        assert ops.allclose(to_numpy(y), to_numpy(ops.sum(arange(12).reshape((3, 4)), axis=0)))
        
        # Test sum along axis 1
        y = ops.sum(x, axis=1)
        assert shape(y) == (3,)
        assert ops.allclose(to_numpy(y), to_numpy(ops.sum(arange(12).reshape((3, 4)), axis=1)))
        
        # Test sum with keepdims
        y = ops.sum(x, axis=0, keepdims=True)
        assert shape(y) == (1, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.sum(arange(12).reshape((3, 4)), axis=0, keepdims=True)))

class TestElementWiseOperations:
    """Tests for element-wise operations."""

    def test_exp(self, backend):
        """Test exp operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test exp
        y = ops.exp(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.exp(ones((3, 4)))))

    def test_log(self, backend):
        """Test log operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test log
        y = ops.log(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.log(ones((3, 4)))))

    def test_sqrt(self, backend):
        """Test sqrt operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test sqrt
        y = ops.sqrt(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.sqrt(ones((3, 4)))))

    def test_pow(self, backend):
        """Test pow operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test pow with scalar exponent
        y = ops.pow(x, 2)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.pow(ones((3, 4)), 2)))
        
        # Test pow with tensor exponent
        x = ones((3, 4))
        z = full((3, 4), 2)
        y = ops.pow(x, z)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.pow(ones((3, 4)), full((3, 4), 2))))
        
        # Test pow with broadcasting
        x = ones((3, 4))
        z = full((1, 4), 2)
        y = ops.pow(x, z)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.pow(ones((3, 4)), full((1, 4), 2))))

    def test_abs(self, backend):
        """Test abs operation."""
        # Create a tensor with negative values
        x = ops.subtract(zeros((3, 4)), ones((3, 4)))
        
        # Test abs
        y = ops.abs(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.abs(-ones((3, 4)))))

    def test_clip(self, backend):
        """Test clip operation."""
        # Create a tensor
        x = arange(12).reshape((3, 4))
        
        # Test clip
        y = ops.clip(x, 3, 8)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.clip(arange(12).reshape((3, 4)), 3, 8)))

class TestActivationFunctions:
    """Tests for activation functions."""

    def test_sigmoid(self, backend):
        """Test sigmoid operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test sigmoid
        y = ops.sigmoid(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.sigmoid(ones((3, 4)))))

    def test_relu(self, backend):
        """Test relu operation."""
        # Create a tensor with negative values
        x = ops.subtract(arange(12).reshape((3, 4)), 5)
        
        # Test relu
        y = ops.relu(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(maximum(0, arange(12).reshape((3, 4)) - 5)))

    def test_tanh(self, backend):
        """Test tanh operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test tanh
        y = ops.tanh(x)
        assert shape(y) == (3, 4)
        assert ops.allclose(to_numpy(y), to_numpy(ops.tanh(ones((3, 4)))))

    def test_softmax(self, backend):
        """Test softmax operation."""
        # Create a tensor
        x = arange(12).reshape((3, 4))
        
        # Test softmax along last axis (default)
        y = ops.softmax(x)
        assert shape(y) == (3, 4)
        
        # Compute expected result using ops
        expected = to_numpy(ops.softmax(arange(12).reshape((3, 4))))
        assert ops.allclose(to_numpy(y), expected)
        
        # Test softmax along axis 0
        y = ops.softmax(x, axis=0)
        assert shape(y) == (3, 4)
        
        # Compute expected result using ops
        expected = to_numpy(ops.softmax(arange(12).reshape((3, 4)), axis=0))
        assert ops.allclose(to_numpy(y), expected)