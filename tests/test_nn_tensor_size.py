"""
Test for tensor size calculations.

This module tests that the tensor size (number of elements) is calculated correctly.
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
def test_tensor_size(backend_name, original_backend):
    """Test that the tensor size (number of elements) is calculated correctly."""
    try:
        # Set the backend
        ops.set_backend(backend_name)
        
        # Helper function to calculate the number of elements in a tensor
        def calculate_size(t):
            shape = tensor.shape(t)
            if len(shape) == 0:
                return 1  # Scalar tensor
            size = 1
            for dim in shape:
                size *= dim
            return size
        
        # Test with 1D tensor
        t_1d = tensor.convert_to_tensor([1, 2, 3, 4, 5])
        assert calculate_size(t_1d) == 5
        
        # Test with 2D tensor
        t_2d = tensor.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
        assert calculate_size(t_2d) == 6
        
        # Test with 3D tensor
        t_3d = tensor.convert_to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert calculate_size(t_3d) == 8
        
        # Test with empty tensor
        t_empty = tensor.convert_to_tensor([])
        # Empty tensor has shape (0,), so product is 0
        assert calculate_size(t_empty) == 0
        
        # Test with scalar tensor
        t_scalar = tensor.convert_to_tensor(5)
        assert calculate_size(t_scalar) == 1
        
        # Test with zeros
        t_zeros = tensor.zeros((2, 3, 4))
        assert calculate_size(t_zeros) == 24
        
        # Test with ones
        t_ones = tensor.ones((3, 2, 1))
        assert calculate_size(t_ones) == 6
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")


if __name__ == "__main__":
    # Set backend to numpy for running as a standalone script
    ops.set_backend('numpy')
    test_tensor_size('numpy', 'numpy')
    print("All tests passed!")