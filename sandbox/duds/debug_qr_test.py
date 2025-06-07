"""
Debug script for the QR decomposition test.

This script adds detailed print statements to troubleshoot the QR orthogonality issue.
"""

import sys
import os
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.ops import stats
from ember_ml.ops import linearalg

def debug_qr_test():
    """Debug version of the QR numerical stability test."""
    print("\n=== QR Numerical Stability Debug Test ===\n")
    
    # Set the backend to MLX
    set_backend("mlx")
    print(f"Backend: {ops.get_backend()}")
    
    # Create a test matrix
    n = 10
    m = 5
    print(f"Matrix dimensions: {n}x{m}")
    
    # Create a random matrix with a fixed seed for reproducibility
    tensor.set_seed(42)
    a = tensor.random_normal((n, m))
    print(f"Input matrix shape: {a.shape}")
    
    # Print the first few rows of the input matrix
    print("\nInput matrix (first 3 rows):")
    for i in range(min(3, n)):
        print(f"Row {i}: {a[i]}")
    
    # Perform QR decomposition
    print("\nPerforming QR decomposition...")
    start_time = time.time()
    q, r = linearalg.qr(a)
    end_time = time.time()
    print(f"QR decomposition completed in {end_time - start_time:.4f} seconds")
    
    # Print shapes
    print(f"Q shape: {q.shape}")
    print(f"R shape: {r.shape}")
    
    # Print the first few rows of Q and R
    print("\nQ matrix (first 3 rows):")
    for i in range(min(3, q.shape[0])):
        print(f"Row {i}: {q[i]}")
    
    print("\nR matrix (first 3 rows):")
    for i in range(min(3, r.shape[0])):
        print(f"Row {i}: {r[i]}")
    
    # Check if R is upper triangular
    print("\nChecking if R is upper triangular...")
    is_upper_triangular = True
    for i in range(1, min(r.shape[0], r.shape[1])):
        for j in range(i):
            if abs(r[i, j]) > 1e-10:
                is_upper_triangular = False
                print(f"R[{i},{j}] = {r[i, j]} (should be close to 0)")
    
    if is_upper_triangular:
        print("R is upper triangular (all elements below diagonal are close to 0)")
    else:
        print("R is NOT upper triangular")
    
    # Check orthogonality of columns (Q^T * Q should be identity)
    print("\nChecking orthogonality of Q...")
    q_t = tensor.transpose(q)
    print(f"Q^T shape: {q_t.shape}")
    
    q_t_q = ops.matmul(q_t, q)
    print(f"Q^T * Q shape: {q_t_q.shape}")
    
    # Create identity matrix
    identity = tensor.eye(q_t_q.shape[0])
    print(f"Identity shape: {identity.shape}")
    
    # Print Q^T * Q and identity matrices
    print("\nQ^T * Q matrix (first 3x3 elements):")
    for i in range(min(3, q_t_q.shape[0])):
        row_str = ""
        for j in range(min(3, q_t_q.shape[1])):
            row_str += f"{q_t_q[i, j]:.6f} "
        print(row_str)
    
    print("\nIdentity matrix (first 3x3 elements):")
    for i in range(min(3, identity.shape[0])):
        row_str = ""
        for j in range(min(3, identity.shape[1])):
            row_str += f"{identity[i, j]:.6f} "
        print(row_str)
    
    # Compute error matrix and maximum absolute error
    diff = ops.subtract(q_t_q, identity)
    print(f"Difference matrix shape: {diff.shape}")
    
    # Print the difference matrix
    print("\nDifference matrix (first 3x3 elements):")
    for i in range(min(3, diff.shape[0])):
        row_str = ""
        for j in range(min(3, diff.shape[1])):
            row_str += f"{diff[i, j]:.6f} "
        print(row_str)
    
    # Compute error statistics
    abs_diff = ops.abs(diff)
    max_error = ops.stats.max(abs_diff)
    mean_error = stats.mean(abs_diff)
    
    print(f"\nMaximum absolute error: {max_error}")
    print(f"Mean absolute error: {mean_error}")
    
    # Check diagonal and off-diagonal elements separately
    diag_errors = []
    offdiag_errors = []
    
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            error = abs(diff[i, j])
            if i == j:
                diag_errors.append(error)
            else:
                offdiag_errors.append(error)
    
    max_diag_error = max(diag_errors) if diag_errors else 0
    max_offdiag_error = max(offdiag_errors) if offdiag_errors else 0
    
    print(f"Maximum error on diagonal: {max_diag_error}")
    print(f"Maximum error off diagonal: {max_offdiag_error}")
    
    # Check reconstruction
    print("\nChecking reconstruction (Q * R ≈ A)...")
    recon = ops.matmul(q, r)
    print(f"Reconstruction shape: {recon.shape}")
    
    # Compute reconstruction error
    recon_diff = ops.subtract(a, recon)
    recon_abs_diff = ops.abs(recon_diff)
    recon_max_error = ops.stats.max(recon_abs_diff)
    recon_mean_error = stats.mean(recon_abs_diff)
    
    print(f"Maximum reconstruction error: {recon_max_error}")
    print(f"Mean reconstruction error: {recon_mean_error}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    orthogonality_ok = max_error < 1e-10
    reconstruction_ok = recon_max_error < 1e-10
    
    print(f"Orthogonality check: {'PASS' if orthogonality_ok else 'FAIL'}")
    print(f"Reconstruction check: {'PASS' if reconstruction_ok else 'FAIL'}")
    
    if not orthogonality_ok:
        print("\nOrthogonality issue detected. Possible causes:")
        print("1. Numerical precision issues in the QR implementation")
        print("2. Issues with the Metal kernel implementation")
        print("3. Incorrect handling of the Q matrix in the QR algorithm")
    
    return orthogonality_ok and reconstruction_ok

if __name__ == "__main__":
    debug_qr_test()