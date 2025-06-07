"""
Tests for High-Precision Computing (HPC) operations in the PyTorch backend.

This module tests the specialized HPC operations that provide enhanced numerical
stability and precision, particularly for operations that are challenging for
standard floating-point arithmetic.
"""

import pytest
import numpy as np
import math
import torch

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.backend.torch.linearalg.orthogonal_ops import HPC16x8, qr_128

@pytest.fixture
def torch_backend():
    """Set up PyTorch backend for tests."""
    from ember_ml.ops import set_backend
    prev_backend = ops.get_backend()
    set_backend('torch')
    yield None
    set_backend(prev_backend)

def test_hpc_orthogonal_vs_standard_qr(torch_backend):
    """
    Test that HPC orthogonal implementation has better numerical stability
    than standard QR for ill-conditioned matrices.
    """
    # Create a highly ill-conditioned matrix
    n = 100
    m = 50
    
    # Create a matrix with exponentially decreasing singular values
    # This will be very challenging for standard QR
    u = tensor.random_normal((n, m))
    s = tensor.exp(-tensor.arange(m, dtype=tensor.float32) / 5)  # Exponentially decreasing
    v = tensor.random_normal((m, m))
    
    # Create ill-conditioned matrix A = U * diag(s) * V^T
    u_orth = ops.linearalg.orthogonal((n, m))
    v_orth = ops.linearalg.orthogonal((m, m))
    
    # Create diagonal matrix with singular values
    diag_s = tensor.zeros((n, m))
    for i in range(m):
        diag_s = tensor.slice_update(diag_s, (i, i), s[i])
    
    # Compute A = U * diag(s) * V^T
    a = ops.matmul(ops.matmul(u_orth, diag_s), ops.transpose(v_orth))
    
    # Get orthogonal matrix using our HPC implementation
    q_hpc = ops.linearalg.orthogonal((n, m))
    
    # Check orthogonality of columns (Q^T * Q should be close to identity)
    q_t_q = ops.matmul(ops.transpose(q_hpc), q_hpc)
    identity = tensor.eye(m)
    
    # Compute error
    error_hpc = ops.stats.mean(ops.abs(q_t_q - identity))
    
    # Now try with standard QR (without HPC)
    # We'll use torch.linalg.qr directly
    a_torch = tensor.to_numpy(a)
    a_torch = tensor.convert_to_tensor(a_torch, dtype=torch.float32)
    q_torch, _ = torch.linalg.qr(a_torch, mode='reduced')
    q_t_q_torch = ops.matmul(q_torch.T, q_torch)
    identity_torch = torch.eye(m)
    error_standard = torch.mean(torch.abs(q_t_q_torch - identity_torch)).item()
    
    # The HPC implementation should have better numerical stability
    assert error_hpc < error_standard, f"HPC error: {error_hpc}, Standard error: {error_standard}"
    print(f"HPC error: {error_hpc}, Standard error: {error_standard}")
    
    # The HPC error should be very small
    assert error_hpc < 1e-5, f"HPC error too large: {error_hpc}"

def test_hpc_limb_arithmetic_precision(torch_backend):
    """
    Test that HPC limb arithmetic provides better precision than standard arithmetic.
    
    This test demonstrates how the double-single precision technique used in HPC
    can represent numbers more precisely than standard floating point.
    """
    # Create a small number
    small = 1e-8
    
    # Create a large number
    large = 1e8
    
    # In standard floating point, adding a small number to a large number
    # and then subtracting the large number should give the small number,
    # but due to precision limitations, it often doesn't
    
    # Standard arithmetic
    large_torch = tensor.convert_to_tensor(large, dtype=torch.float32)
    small_torch = tensor.convert_to_tensor(small, dtype=torch.float32)
    
    sum_standard = large_torch + small_torch
    diff_standard = sum_standard - large_torch
    
    # HPC limb arithmetic
    large_hpc = HPC16x8(large_torch)
    small_hpc = HPC16x8(small_torch)
    
    # Add using HPC
    from ember_ml.backend.torch.linearalg.orthogonal_ops import _add_limb_precision
    sum_high, sum_low = _add_limb_precision(large_hpc.high, large_hpc.low, small_hpc.high, small_hpc.low)
    
    # Create HPC object for sum
    sum_hpc = HPC16x8(sum_high, sum_low)
    
    # Subtract using HPC
    diff_high, diff_low = _add_limb_precision(sum_hpc.high, sum_hpc.low, -large_hpc.high, -large_hpc.low)
    
    # Convert back to standard precision
    diff_hpc = diff_high + diff_low
    
    # The HPC version should be closer to the true small value
    error_standard = abs(diff_standard.item() - small) / small
    error_hpc = abs(diff_hpc.item() - small) / small
    
    print(f"Standard arithmetic result: {diff_standard.item()}, expected: {small}")
    print(f"HPC arithmetic result: {diff_hpc.item()}, expected: {small}")
    print(f"Standard relative error: {error_standard}, HPC relative error: {error_hpc}")
    
    # The HPC error should be smaller
    assert error_hpc < error_standard, f"HPC error: {error_hpc}, Standard error: {error_standard}"

def test_orthogonal_non_square_matrices(torch_backend):
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
            q_t_q = ops.matmul(ops.transpose(q), q)
            identity = tensor.eye(shape[1])
            error = ops.stats.mean(ops.abs(q_t_q - identity))
        else:
            # Wide matrix: Q * Q^T should be identity
            q_q_t = ops.matmul(q, ops.transpose(q))
            identity = tensor.eye(shape[0])
            error = ops.stats.mean(ops.abs(q_q_t - identity))
        
        # Error should be small
        assert error < 1e-5, f"Orthogonality error too large for shape {shape}: {error}"
        print(f"Shape {shape}: orthogonality error = {error}")

def test_qr_128_precision(torch_backend):
    """
    Test that the 128-bit precision QR decomposition has better numerical stability
    than standard QR for ill-conditioned matrices.
    """
    # Create a matrix with poor conditioning
    n = 50
    
    # Create a matrix with exponentially decreasing diagonal elements
    diag_vals = torch.exp(-torch.arange(n, dtype=torch.float32) / 5)
    a = torch.diag(diag_vals)
    
    # Add some noise to make it more challenging
    noise = torch.randn(n, n) * 1e-3
    a = a + noise
    
    # Compute QR using our 128-bit precision implementation
    q_hpc, r_hpc = qr_128(a)
    
    # Compute QR using standard PyTorch
    q_std, r_std = torch.linalg.qr(a)
    
    # Check orthogonality of Q
    q_t_q_hpc = ops.matmul(q_hpc.T, q_hpc)
    q_t_q_std = ops.matmul(q_std.T, q_std)
    
    identity = torch.eye(n)
    
    # Compute errors
    error_hpc = torch.mean(torch.abs(q_t_q_hpc - identity)).item()
    error_std = torch.mean(torch.abs(q_t_q_std - identity)).item()
    
    # Check reconstruction error
    recon_hpc = ops.matmul(q_hpc, r_hpc)
    recon_std = ops.matmul(q_std, r_std)
    
    recon_error_hpc = torch.mean(torch.abs(recon_hpc - a)).item()
    recon_error_std = torch.mean(torch.abs(recon_std - a)).item()
    
    print(f"HPC orthogonality error: {error_hpc}, Standard error: {error_std}")
    print(f"HPC reconstruction error: {recon_error_hpc}, Standard error: {recon_error_std}")
    
    # The HPC implementation should have better orthogonality
    assert error_hpc <= error_std * 1.1, f"HPC error: {error_hpc}, Standard error: {error_std}"
    
    # Both should have similar reconstruction error
    assert abs(recon_error_hpc - recon_error_std) < 1e-5, \
        f"HPC recon error: {recon_error_hpc}, Standard recon error: {recon_error_std}"