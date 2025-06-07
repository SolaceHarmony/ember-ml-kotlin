"""
Test linear algebra operations.

This module tests the linear algebra operations in the Ember ML framework.
"""

import pytest
import numpy as np

from ember_ml.backend import set_backend

# Test backends
BACKENDS = ['numpy', 'torch', 'mlx']

@pytest.mark.parametrize('backend', BACKENDS)
def test_qr_decomposition(backend):
    """Test QR decomposition."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
    # Create a test matrix
    a = tensor.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Compute QR decomposition
    q, r = linearalg.qr(a)
    
    # Check that Q is orthogonal
    from ember_ml import ops
    q_t = ops.transpose(q)
    q_t_q = ops.matmul(q_t, q)
    identity = tensor.eye(q_t_q.shape[0])
    
    # Check that Q*R = A
    qr_product = ops.matmul(q, r)
    
    # Check results
    assert ops.allclose(q_t_q, identity, atol=1e-5)
    assert ops.allclose(qr_product, a, atol=1e-5)

@pytest.mark.parametrize('backend', BACKENDS)
def test_diag_vector_to_matrix(backend):
    """Test diagonal operation (vector to matrix)."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
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

@pytest.mark.parametrize('backend', BACKENDS)
def test_diag_matrix_to_vector(backend):
    """Test diagonal operation (matrix to vector)."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
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

@pytest.mark.parametrize('backend', BACKENDS)
def test_diag_with_offset(backend):
    """Test diagonal operation with offset."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
    # Create a test matrix
    m = tensor.convert_to_tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Extract the diagonal with offset 1
    d_upper = linearalg.diag(m, k=1)
    
    # Extract the diagonal with offset -1
    d_lower = linearalg.diag(m, k=-1)
    
    # Expected results
    expected_upper = tensor.convert_to_tensor([2.0, 6.0])
    expected_lower = tensor.convert_to_tensor([4.0, 8.0])
    
    # Check results
    assert ops.allclose(d_upper, expected_upper)
    assert ops.allclose(d_lower, expected_lower)

@pytest.mark.parametrize('backend', BACKENDS)
def test_diagonal(backend):
    """Test diagonal operation."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
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

@pytest.mark.parametrize('backend', BACKENDS)
def test_diagonal_with_offset(backend):
    """Test diagonal operation with offset."""
    set_backend(backend)
    import ember_ml.ops as ops
    from ember_ml.ops import linearalg
    from ember_ml.nn import tensor
    # Create a test matrix
    m = tensor.convert_to_tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Extract the diagonal with offset 1
    d_upper = linearalg.diagonal(m, offset=1)
    
    # Extract the diagonal with offset -1
    d_lower = linearalg.diagonal(m, offset=-1)
    
    # Expected results
    expected_upper = tensor.convert_to_tensor([2.0, 6.0])
    expected_lower = tensor.convert_to_tensor([4.0, 8.0])
    
    # Check results
    assert ops.allclose(d_upper, expected_upper)
    assert ops.allclose(d_lower, expected_lower)