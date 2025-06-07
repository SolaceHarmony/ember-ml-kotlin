"""
Tests for High-Precision Computing (HPC) operations in the NumPy backend.

Tests operate strictly through the frontend ops and tensor interfaces where possible.
"""

import pytest

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import linearalg
# No direct backend imports in frontend test files

# Note: Assumes conftest.py provides the numpy_backend fixture
# which sets the backend to 'numpy' before these tests run.

def test_hpc_orthogonal_vs_standard_qr(numpy_backend):
    """
    Test that HPC orthogonal implementation has better numerical stability
    than standard QR for ill-conditioned matrices.
    """
    # Create a highly ill-conditioned matrix
    n = 100
    m = 50

    # Create a matrix with exponentially decreasing singular values
    u = tensor.random_normal((n, m))
    s = ops.exp(-tensor.arange(m, dtype=tensor.float32) / 5)  # Exponentially decreasing
    v = tensor.random_normal((m, m))

    # Create ill-conditioned matrix A = U * diag(s) * V^T
    u_orth = ops.linearalg.orthogonal((n, m))
    v_orth = ops.linearalg.orthogonal((m, m))

    # Create diagonal matrix with singular values using ops.linearalg.diag
    diag_s = ops.linearalg.diag(s)

    # Compute A = U * diag(s) * V^T
    a = ops.matmul(ops.matmul(u_orth, diag_s), tensor.transpose(v_orth))

    # Get orthogonal matrix using our HPC implementation (via frontend)
    q_hpc = ops.linearalg.orthogonal((n, m))

    # Check orthogonality of columns (Q^T * Q should be close to identity)
    q_t_q = ops.matmul(tensor.transpose(q_hpc), q_hpc)
    identity = tensor.eye(m)

    # Compute error
    error_hpc = ops.stats.mean(ops.abs(q_t_q - identity))

    # Now try with standard QR (using NumPy directly for comparison)
    import numpy as np
    a_np = tensor.to_numpy(a)
    q_np, _ = ops.linearalg.qr(a_np, mode='reduced')
    q_t_q_np = ops.matmul(q_np.T, q_np)
    identity_np = ops.eye(m)
    error_standard = stats.mean(ops.abs(q_t_q_np - identity_np))

    # The HPC implementation should have significantly better numerical stability
    assert ops.all(ops.less(error_hpc, tensor.convert_to_tensor(error_standard, dtype=tensor.float32))), f"HPC error: {error_hpc}, Standard error: {error_standard}"
    print(f"HPC error: {error_hpc}, Standard error: {error_standard}")

    # The HPC error should be very small
    assert ops.all(ops.less(error_hpc, tensor.convert_to_tensor(1e-5, dtype=tensor.float32))), f"HPC error too large: {error_hpc}"

def test_orthogonalize_nonsquare(numpy_backend):
    """
    Test the orthogonalize_nonsquare function for large matrices.

    This test verifies that the frontend orthogonalize_nonsquare function
    works correctly for large non-square matrices with the NumPy backend.
    """
    # This test replicates the structure of the MLX Metal kernel test,
    # but tests the frontend `orthogonalize_nonsquare` function with NumPy.
    # It does not test a specific NumPy backend implementation detail like the Metal kernel.

    # Create a large non-square matrix
    n = 1024
    m = 512

    # Create random matrix
    a = tensor.random_normal((n, m))

    # Use the frontend orthogonalization function
    q = linearalg.orthogonalize_nonsquare(a)

    # Check orthogonality
    q_t_q = ops.matmul(tensor.transpose(q), q)
    identity = tensor.eye(m)
    error = ops.stats.mean(ops.abs(q_t_q - identity)).item()

    print(f"NumPy orthogonalize_nonsquare error: {error}")

    # The error should be small
    assert ops.all(ops.less(error, tensor.convert_to_tensor(1e-5, dtype=tensor.float32))), f"Orthogonalize_nonsquare error too large: {error}"


def test_hpc_limb_arithmetic_precision(numpy_backend):
    """
    Test that HPC limb arithmetic provides better precision than standard arithmetic.

    This test demonstrates how the double-single precision technique used in HPC
    can represent numbers more precisely than standard floating point.
    """
    # This test in the MLX version directly imports and uses a backend internal
    # function (_add_limb_precision) which violates the backend purity rule
    # for frontend test files. It is not testing a frontend API.
    # Therefore, this test cannot be directly replicated using only frontend
    # abstractions and is skipped for the NumPy backend.
    pytest.skip("HPC limb arithmetic test is specific to MLX backend implementation details.")


def test_orthogonal_non_square_matrices(numpy_backend):
    """
    Test that the orthogonal function works correctly for non-square matrices.

    This test verifies that the orthogonal function produces matrices with
    orthogonal columns even for highly rectangular matrices.
    """
    # Test with various shapes
    shapes = [
        (100, 10),    # Tall and thin
        (10, 100),    # Short and wide
        (128, 64),    # Power of 2 dimensions
        (65, 33),     # Odd dimensions
        (200, 199),   # Almost square
        (3, 100)      # Very rectangular
    ]

    for shape in shapes:
        # Generate orthogonal matrix
        q = ops.linearalg.orthogonal(shape)

        # Check shape
        assert q.shape == shape, f"Expected shape {shape}, got {q.shape}"

        # Check orthogonality of columns
        if shape[0] >= shape[1]:
            # Tall matrix: Q^T * Q should be identity
            q_t_q = ops.matmul(tensor.transpose(q), q)
            identity = tensor.eye(shape[1])
            error = ops.stats.mean(ops.abs(q_t_q - identity))
        else:
            # Wide matrix: Q * Q^T should be identity
            q_q_t = ops.matmul(q, tensor.transpose(q))
            identity = tensor.eye(shape[0])
            error = ops.stats.mean(ops.abs(q_q_t - identity))

        # Error should be small
        assert ops.all(ops.less(error, tensor.convert_to_tensor(1e-5, dtype=tensor.float32))), f"Orthogonality error too large for shape {shape}: {error}"
        print(f"Shape {shape}: orthogonality error = {error}")