"""
Unit tests for tensor operations across different backends.

This module contains pytest tests for the tensor operations in the nn.tensor module.
It tests each operation with different backends to ensure consistency.
"""

import pytest
from ember_ml.nn.tensor import (
    zeros, ones, zeros_like, ones_like, eye, reshape, transpose,
    concatenate, stack, shape, dtype, cast, copy,
    float32, int32, array, EmberTensor
)
from ember_ml.backend import get_backend, set_backend

# List of backends to test
BACKENDS = ['numpy']
try:
    # Check if torch is available without importing it directly
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        BACKENDS.append('torch')
except ImportError:
    pass

try:
    # Check if mlx is available without importing it directly
    import importlib.util
    if importlib.util.find_spec("mlx") is not None:
        BACKENDS.append('mlx')
except ImportError:
    pass

@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with different backends."""
    prev_backend = get_backend()
    set_backend(request.param)
    yield request.param
    # Make sure prev_backend is not None before setting it
    if prev_backend is not None:
        set_backend(prev_backend)
    else:
        # Default to numpy if prev_backend was None
        set_backend('numpy')

class TestTensorCreation:
    """Tests for tensor creation operations."""

    def test_zeros(self, backend):
        """Test zeros operation."""
        # Test with 1D shape
        tensor_shape = (5,)
        x = zeros(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Create a reference tensor of zeros for comparison
        ref_zeros = zeros(tensor_shape)
        assert shape(x) == shape(ref_zeros)
        
        # Verify all elements are 0
        for i in range(tensor_shape[0]):
            assert x[i].item() == 0
        
        # Test with 2D shape
        tensor_shape = (3, 4)
        x = zeros(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Verify all elements are 0
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                assert x[i, j].item() == 0

        # Test with dtype
        x = zeros(tensor_shape, dtype=float32)
        assert dtype(x) == float32

    def test_ones(self, backend):
        """Test ones operation."""
        # Test with 1D shape
        tensor_shape = (5,)
        x = ones(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Verify all elements are 1
        for i in range(tensor_shape[0]):
            assert x[i].item() == 1
        
        # Test with 2D shape
        tensor_shape = (3, 4)
        x = ones(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Verify all elements are 1
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                assert x[i, j].item() == 1

        # Test with dtype
        x = ones(tensor_shape, dtype=float32)
        assert dtype(x) == float32

    def test_zeros_like(self, backend):
        """Test zeros_like operation."""
        # Create a tensor to use as reference
        tensor_shape = (3, 4)
        x_ref = ones(tensor_shape)
        
        # Test zeros_like
        x = zeros_like(x_ref)
        assert shape(x) == tensor_shape
        
        # Verify all elements are 0
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                assert x[i, j].item() == 0
        
        # Test with different dtype
        x = zeros_like(x_ref, dtype=float32)
        assert dtype(x) == float32

    def test_ones_like(self, backend):
        """Test ones_like operation."""
        # Create a tensor to use as reference
        tensor_shape = (3, 4)
        x_ref = zeros(tensor_shape)
        
        # Test ones_like
        x = ones_like(x_ref)
        assert shape(x) == tensor_shape
        
        # Verify all elements are 1
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                assert x[i, j].item() == 1
        
        # Test with different dtype
        x = ones_like(x_ref, dtype=float32)
        assert dtype(x) == float32

    def test_eye(self, backend):
        """Test eye operation."""
        # Test square matrix
        n = 3
        x = eye(n)
        assert shape(x) == (n, n)
        
        # Verify it's an identity matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert x[i, j].item() == 1
                else:
                    assert x[i, j].item() == 0
        
        # Test rectangular matrix
        n, m = 3, 4
        x = eye(n, m)
        assert shape(x) == (n, m)
        
        # Verify it's an identity matrix (rectangular)
        for i in range(n):
            for j in range(m):
                if i == j:
                    assert x[i, j].item() == 1
                else:
                    assert x[i, j].item() == 0
        
        # Test with dtype
        x = eye(n, dtype=float32)
        assert dtype(x) == float32

class TestTensorManipulation:
    """Tests for tensor manipulation operations."""

    def test_reshape(self, backend):
        """Test reshape operation."""
        # Create a tensor
        original_shape = (2, 6)
        x = ones(original_shape)
        
        # Test reshape
        new_shape = (3, 4)
        y = reshape(x, new_shape)
        assert shape(y) == new_shape
        
        # Verify all elements are still 1
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                assert y[i, j].item() == 1
        
        # Test reshape with -1 dimension
        new_shape = (3, -1)
        y = reshape(x, new_shape)
        assert shape(y) == (3, 4)
        
        # Verify all elements are still 1
        for i in range(3):
            for j in range(4):
                assert y[i, j].item() == 1

    def test_transpose(self, backend):
        """Test transpose operation."""
        # Create a tensor with different values for each position
        tensor_shape = (2, 3)
        data = [[1, 2, 3], [4, 5, 6]]
        x = array(data)
        
        # Test default transpose
        y = transpose(x)
        
        # Verify the transpose
        assert shape(y) == (3, 2)
        assert y[0, 0].item() == 1
        assert y[0, 1].item() == 4
        assert y[1, 0].item() == 2
        assert y[1, 1].item() == 5
        assert y[2, 0].item() == 3
        assert y[2, 1].item() == 6

    def test_concatenate(self, backend):
        """Test concatenate operation."""
        # Create tensors
        shape1 = (2, 3)
        shape2 = (2, 3)
        x1 = ones(shape1)
        x2 = zeros(shape2)
        
        # Test concatenate along axis 0
        y = concatenate([x1, x2], axis=0)
        assert shape(y) == (4, 3)
        
        # Verify the concatenation
        for i in range(4):
            for j in range(3):
                if i < 2:
                    assert y[i, j].item() == 1  # First 2 rows should be 1
                else:
                    assert y[i, j].item() == 0  # Last 2 rows should be 0
        
        # Test concatenate along axis 1
        y = concatenate([x1, x2], axis=1)
        assert shape(y) == (2, 6)
        
        # Verify the concatenation
        for i in range(2):
            for j in range(6):
                if j < 3:
                    assert y[i, j].item() == 1  # First 3 columns should be 1
                else:
                    assert y[i, j].item() == 0  # Last 3 columns should be 0

    def test_stack(self, backend):
        """Test stack operation."""
        # Create tensors
        tensor_shape = (2, 3)
        x1 = ones(tensor_shape)
        x2 = zeros(tensor_shape)
        
        # Test stack along axis 0
        y = stack([x1, x2], axis=0)
        assert shape(y) == (2, 2, 3)
        
        # Verify the stacking
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    if i == 0:
                        assert y[i, j, k].item() == 1  # First tensor should be all 1s
                    else:
                        assert y[i, j, k].item() == 0  # Second tensor should be all 0s
        
        # Test stack along axis 1
        y = stack([x1, x2], axis=1)
        assert shape(y) == (2, 2, 3)
        
        # Verify the stacking
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    if j == 0:
                        assert y[i, j, k].item() == 1  # First position along axis 1 should be all 1s
                    else:
                        assert y[i, j, k].item() == 0  # Second position along axis 1 should be all 0s
        
        # Test stack along axis 2
        y = stack([x1, x2], axis=2)
        assert shape(y) == (2, 3, 2)
        
        # Verify the stacking
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    if k == 0:
                        assert y[i, j, k].item() == 1  # First position along axis 2 should be all 1s
                    else:
                        assert y[i, j, k].item() == 0  # Second position along axis 2 should be all 0s

class TestTensorInfo:
    """Tests for tensor information operations."""

    def test_shape(self, backend):
        """Test shape operation."""
        # Test with 1D shape
        tensor_shape = (5,)
        x = ones(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Test with 2D shape
        tensor_shape = (3, 4)
        x = ones(tensor_shape)
        assert shape(x) == tensor_shape
        
        # Test with 3D shape
        tensor_shape = (2, 3, 4)
        x = ones(tensor_shape)
        assert shape(x) == tensor_shape

    def test_dtype(self, backend):
        """Test dtype operation."""
        # Test with default dtype
        x = ones((3, 4))
        
        # The default dtype depends on the backend, but we can't check the exact type
        # since we're using the abstracted dtype objects now
        
        # Test with specified dtype
        x = ones((3, 4), dtype=float32)
        assert dtype(x) == float32

    def test_cast(self, backend):
        """Test cast operation."""
        # Create a tensor
        x = ones((3, 4))
        
        # Test cast to different dtype
        y = cast(x, int32)
        assert dtype(y) == int32
        
        # Verify all elements are still 1
        for i in range(3):
            for j in range(4):
                assert y[i, j].item() == 1

    def test_copy(self, backend):
        """Test copy operation."""
        # Create a tensor with specific values
        data = [[1, 2], [3, 4]]
        x = array(data)
        
        # Test copy
        y = copy(x)
        assert shape(y) == shape(x)
        assert dtype(y) == dtype(x)
        
        # Verify the values are the same
        for i in range(2):
            for j in range(2):
                assert y[i, j].item() == x[i, j].item()