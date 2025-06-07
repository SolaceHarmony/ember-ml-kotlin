"""Tests for MLX backend strong typing implementation.

This test suite validates the strong typing approach implemented in the MLX backend.
If successful, this approach can be applied to other backends as well.
"""

import pytest
import numpy as np
import mlx.core as mx
import torch

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.backend import get_backend, set_backend


class TestMLXBackendStrongTyping:
    """Test suite for MLX backend strong typing."""

    def setup_method(self):
        """Set up the test environment."""
        # Store the current backend
        self.original_backend = get_backend()
        # Default to 'numpy' if no backend is set
        if self.original_backend is None:
            self.original_backend = 'numpy'
        
    def teardown_method(self):
        """Tear down the test environment."""
        # Restore the original backend
        # Ensure we have a valid backend name (not None)
        backend_name = self.original_backend if self.original_backend is not None else 'numpy'
        set_backend(backend_name)
        ops.set_ops(backend_name)

    def test_mlx_tensor_creation(self):
        """Test MLX tensor creation with different input types."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Test with Python list
        data = [1, 2, 3, 4]
        x = tensor.convert_to_tensor(data)
        assert isinstance(x, mx.array)
        assert np.array_equal(tensor.to_numpy(x), np.array(data))
        
        # Test with NumPy array
        data = np.array([1, 2, 3, 4])
        x = tensor.convert_to_tensor(data)
        assert isinstance(x, mx.array)
        assert np.array_equal(tensor.to_numpy(x), data)
        
        # Test with MLX array
        data = mx.array([1, 2, 3, 4])
        x = tensor.convert_to_tensor(data)
        assert isinstance(x, mx.array)
        assert np.array_equal(tensor.to_numpy(x), tensor.to_numpy(data))
        
        # Test with PyTorch tensor - should raise ValueError due to strong typing
        data = torch.tensor([1, 2, 3, 4])
        with pytest.raises(ValueError):
            x = tensor.convert_to_tensor(data)

    def test_mlx_tensor_operations(self):
        """Test MLX tensor operations with strong typing."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Create MLX tensors
        x = tensor.ones((3, 4))
        y = tensor.ones((3, 4))
        
        # Test basic operations
        z = ops.add(x, y)
        assert isinstance(z, mx.array)
        assert np.array_equal(tensor.to_numpy(z), np.ones((3, 4)) * 2)
        
        # Test with Python scalar
        z = ops.multiply(x, 2)
        assert isinstance(z, mx.array)
        assert np.array_equal(tensor.to_numpy(z), np.ones((3, 4)) * 2)
        
        # Test with NumPy array - should work through conversion
        data = np.ones((3, 4))
        z = ops.add(x, data)
        assert isinstance(z, mx.array)
        assert np.array_equal(tensor.to_numpy(z), np.ones((3, 4)) * 2)
        
        # Test with PyTorch tensor - should raise ValueError due to strong typing
        data = torch.ones((3, 4))
        with pytest.raises(ValueError):
            z = ops.add(x, data)

    def test_mlx_to_numpy_conversion(self):
        """Test conversion from MLX tensor to NumPy array."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Create MLX tensor
        x = tensor.ones((3, 4))
        
        # Convert to NumPy
        x_np = tensor.to_numpy(x)
        assert isinstance(x_np, np.ndarray)
        assert np.array_equal(x_np, np.ones((3, 4)))

    def test_numpy_to_mlx_conversion(self):
        """Test conversion from NumPy array to MLX tensor."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Create NumPy array
        x_np = np.ones((3, 4))
        
        # Convert to MLX tensor
        x = tensor.convert_to_tensor(x_np)
        assert isinstance(x, mx.array)
        assert np.array_equal(tensor.to_numpy(x), x_np)

    def test_cross_backend_conversion_numpy_to_mlx(self):
        """Test conversion from NumPy backend to MLX backend."""
        # Set the backend to NumPy
        set_backend('numpy')
        ops.set_ops('numpy')
        
        # Create tensor in NumPy backend
        x_numpy = tensor.ones((3, 4))
        
        # Convert to NumPy array
        x_np = tensor.to_numpy(x_numpy)
        
        # Switch to MLX backend
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Convert NumPy array to MLX tensor
        x_mlx = tensor.convert_to_tensor(x_np)
        assert isinstance(x_mlx, mx.array)
        assert np.array_equal(tensor.to_numpy(x_mlx), x_np)

    def test_cross_backend_conversion_mlx_to_numpy(self):
        """Test conversion from MLX backend to NumPy backend."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Create tensor in MLX backend
        x_mlx = tensor.ones((3, 4))
        
        # Convert to NumPy array
        x_np = tensor.to_numpy(x_mlx)
        
        # Switch to NumPy backend
        set_backend('numpy')
        ops.set_ops('numpy')
        
        # Convert NumPy array to NumPy backend tensor
        x_numpy = tensor.convert_to_tensor(x_np)
        assert isinstance(x_numpy, np.ndarray)
        assert np.array_equal(tensor.to_numpy(x_numpy), x_np)

    def test_direct_backend_tensor_conversion_failure(self):
        """Test that direct conversion between backend tensors fails."""
        # Set the backend to MLX
        set_backend('mlx')
        ops.set_ops('mlx')
        
        # Create MLX tensor
        x_mlx = tensor.ones((3, 4))
        
        # Switch to PyTorch backend
        set_backend('torch')
        ops.set_ops('torch')
        
        # Attempt to convert MLX tensor to PyTorch tensor directly
        # This should fail with a ValueError due to strong typing
        with pytest.raises(ValueError):
            x_torch = tensor.convert_to_tensor(x_mlx)
        
        # The correct way is to go through NumPy
        x_np = tensor.to_numpy(x_mlx)
        x_torch = tensor.convert_to_tensor(x_np)
        assert isinstance(x_torch, torch.Tensor)
        assert np.array_equal(tensor.to_numpy(x_torch), x_np)