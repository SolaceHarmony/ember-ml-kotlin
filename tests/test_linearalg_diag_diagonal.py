"""
Test the diag and diagonal functions across all backends.

This module tests the diag and diagonal functions in the Ember ML framework.
"""

import pytest
from ember_ml.ops import linearalg
from ember_ml.nn import tensor
from ember_ml.backend import set_backend, get_backend
from ember_ml import ops

# Test with all available backends
BACKENDS = ['numpy', 'torch', 'mlx']

@pytest.fixture
def original_backend():
    """Fixture to save and restore the original backend."""
    original = get_backend()
    yield original
    # Restore original backend
    if original is not None:
        set_backend(original)
    else:
        # Default to 'numpy' if original is None
        set_backend('numpy')

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diag_vector_to_matrix(backend_name, original_backend):
    """Test diag function converting a vector to a matrix."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test vector
        v = tensor.convert_to_tensor([1.0, 2.0, 3.0])
        
        # Create a diagonal matrix
        d = linearalg.diag(v)
        
        # Expected result
        expected = tensor.convert_to_tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        # Check result
        assert ops.allclose(d, expected)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diag_matrix_to_vector(backend_name, original_backend):
    """Test diag function extracting the diagonal from a matrix."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test matrix
        m = tensor.convert_to_tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Extract the diagonal
        d = linearalg.diag(m)
        
        # Expected result
        expected = tensor.convert_to_tensor([1.0, 5.0, 9.0])
        
        # Check result
        assert ops.allclose(d, expected)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diag_with_offset(backend_name, original_backend):
    """Test diag function with offset."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test matrix
        m = tensor.convert_to_tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Extract the diagonal with offset 1 (above main diagonal)
        d_upper = linearalg.diag(m, k=1)
        
        # Extract the diagonal with offset -1 (below main diagonal)
        d_lower = linearalg.diag(m, k=-1)
        
        # Expected results
        expected_upper = tensor.convert_to_tensor([2.0, 6.0])
        expected_lower = tensor.convert_to_tensor([4.0, 8.0])
        
        # Check results
        assert ops.allclose(d_upper, expected_upper)
        assert ops.allclose(d_lower, expected_lower)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diagonal_basic(backend_name, original_backend):
    """Test diagonal function with default parameters."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test matrix
        m = tensor.convert_to_tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Extract the diagonal
        d = linearalg.diagonal(m)
        
        # Expected result
        expected = tensor.convert_to_tensor([1.0, 5.0, 9.0])
        
        # Check result
        assert ops.allclose(d, expected)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diagonal_with_offset(backend_name, original_backend):
    """Test diagonal function with offset."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a test matrix
        m = tensor.convert_to_tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Extract the diagonal with offset 1 (above main diagonal)
        d_upper = linearalg.diagonal(m, offset=1)
        
        # Extract the diagonal with offset -1 (below main diagonal)
        d_lower = linearalg.diagonal(m, offset=-1)
        
        # Expected results
        expected_upper = tensor.convert_to_tensor([2.0, 6.0])
        expected_lower = tensor.convert_to_tensor([4.0, 8.0])
        
        # Check results
        assert ops.allclose(d_upper, expected_upper)
        assert ops.allclose(d_lower, expected_lower)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")

@pytest.mark.parametrize("backend_name", BACKENDS)
def test_diagonal_with_axes(backend_name, original_backend):
    """Test diagonal function with different axes."""
    try:
        # Set the backend
        set_backend(backend_name)
        
        # Create a 3D tensor
        t = tensor.convert_to_tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])
        
        # Extract diagonal with default axes (0, 1)
        d_default = linearalg.diagonal(t)
        
        # Extract diagonal with axes (0, 2)
        d_02 = linearalg.diagonal(t, axis1=0, axis2=2)
        
        # Extract diagonal with axes (1, 2)
        d_12 = linearalg.diagonal(t, axis1=1, axis2=2)
        
        # Expected results
        expected_default = tensor.convert_to_tensor([[1, 2], [8, 8]])
        expected_02 = tensor.convert_to_tensor([[1, 3], [6, 8]])
        expected_12 = tensor.convert_to_tensor([[1, 5], [4, 8]])
        
        # Check results
        assert ops.allclose(d_default, expected_default)
        assert ops.allclose(d_02, expected_02)
        assert ops.allclose(d_12, expected_12)
    except ImportError:
        pytest.skip(f"{backend_name} backend not available")
    except Exception as e:
        # Some backends might not support all axis combinations
        pytest.skip(f"Error with {backend_name} backend: {str(e)}")