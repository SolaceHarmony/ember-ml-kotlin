"""
Debug script for the QR decomposition Metal kernel.

This script tests the QR decomposition with a small matrix and prints
detailed debug information from the Metal kernel.
"""

import sys
import os
import time
import mlx.core as mx

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Ember ML modules
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import set_backend
from ember_ml.ops import linearalg

# Import the QR function directly from the MLX backend
from ember_ml.backend.mlx.linearalg.qr_ops import qr as mlx_qr

def debug_qr_kernel():
    """Debug the QR decomposition Metal kernel with a small matrix."""
    print("\n=== QR Metal Kernel Debug Test ===\n")
    
    # Set the backend to MLX
    set_backend("mlx")
    print(f"Backend: {ops.get_backend()}")
    
    # Create a small test matrix
    n, m = 3, 2
    print(f"Matrix dimensions: {n}x{m}")
    
    # Create a matrix with known values
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = tensor.convert_to_tensor(a_values)
    print(f"Input matrix shape: {a.shape}")
    
    # Print the input matrix
    print("\nInput matrix:")
    for i in range(n):
        print(f"Row {i}: {a[i]}")
    
    # Perform QR decomposition with debug=True to get debug info and print detailed debug information
    print("\nPerforming QR decomposition with debug=True...")
    start_time = time.time()
    q, r, dbg = mlx_qr(a, debug=True)
    end_time = time.time()
    print(f"QR decomposition completed in {end_time - start_time:.4f} seconds")
    
    # Print shapes
    print(f"Q shape: {q.shape}")
    print(f"R shape: {r.shape}")
    print(f"Debug info shape: {dbg.shape}")
    
    # Print debug info
    print("\nDebug info array:")
    print(dbg)
    
    # Print Q and R matrices
    print("\nQ matrix:")
    for i in range(q.shape[0]):
        print(f"Row {i}: {q[i]}")
    
    print("\nR matrix:")
    for i in range(r.shape[0]):
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
    
    # Check orthogonality of Q
    print("\nChecking orthogonality of Q...")
    q_t = tensor.transpose(q)
    q_t_q = ops.matmul(q_t, q)
    identity = tensor.eye(q_t_q.shape[0])
    
    print("\nQ^T * Q matrix:")
    for i in range(q_t_q.shape[0]):
        print(f"Row {i}: {q_t_q[i]}")
    
    # Compute error matrix
    diff = ops.subtract(q_t_q, identity)
    abs_diff = ops.abs(diff)
    max_error = ops.stats.max(abs_diff)
    mean_error = ops.stats.mean(abs_diff)
    
    print(f"\nMaximum orthogonality error: {max_error}")
    print(f"Mean orthogonality error: {mean_error}")
    
    # Check reconstruction
    print("\nChecking reconstruction (Q * R ≈ A)...")
    recon = ops.matmul(q, r)
    
    print("\nReconstruction matrix:")
    for i in range(recon.shape[0]):
        print(f"Row {i}: {recon[i]}")
    
    # Compute reconstruction error
    recon_diff = ops.subtract(a, recon)
    recon_abs_diff = ops.abs(recon_diff)
    recon_max_error = ops.stats.max(recon_abs_diff)
    recon_mean_error = ops.stats.mean(recon_abs_diff)
    
    print(f"\nMaximum reconstruction error: {recon_max_error}")
    print(f"Mean reconstruction error: {recon_mean_error}")
    
    # Try using MLX's built-in QR function for comparison
    print("\n=== Comparing with MLX's built-in QR function ===")
    
    # Convert to MLX array
    a_mx = mx.array(a_values)
    
    # Use MLX's built-in QR function
    q_mx, r_mx = mx.linalg.qr(a_mx, stream=mx.cpu)
    
    print("\nMLX built-in QR function results:")
    print(f"Q shape: {q_mx.shape}")
    print(f"R shape: {r_mx.shape}")
    
    print("\nQ matrix (MLX built-in):")
    for i in range(q_mx.shape[0]):
        print(f"Row {i}: {q_mx[i]}")
    
    print("\nR matrix (MLX built-in):")
    for i in range(r_mx.shape[0]):
        print(f"Row {i}: {r_mx[i]}")
    
    # Check orthogonality of Q from MLX's built-in function
    q_t_mx = mx.transpose(q_mx)
    q_t_q_mx = mx.matmul(q_t_mx, q_mx)
    identity_mx = mx.eye(q_t_q_mx.shape[0])
    
    print("\nQ^T * Q matrix (MLX built-in):")
    for i in range(q_t_q_mx.shape[0]):
        print(f"Row {i}: {q_t_q_mx[i]}")
    
    # Compute error matrix
    diff_mx = q_t_q_mx - identity_mx
    abs_diff_mx = mx.abs(diff_mx)
    max_error_mx = mx.max(abs_diff_mx)
    mean_error_mx = mx.mean(abs_diff_mx)
    
    print(f"\nMaximum orthogonality error (MLX built-in): {max_error_mx}")
    print(f"Mean orthogonality error (MLX built-in): {mean_error_mx}")
    
    # Check reconstruction with MLX's built-in function
    recon_mx = mx.matmul(q_mx, r_mx)
    
    print("\nReconstruction matrix (MLX built-in):")
    for i in range(recon_mx.shape[0]):
        print(f"Row {i}: {recon_mx[i]}")
    
    # Compute reconstruction error
    recon_diff_mx = a_mx - recon_mx
    recon_abs_diff_mx = mx.abs(recon_diff_mx)
    recon_max_error_mx = mx.max(recon_abs_diff_mx)
    recon_mean_error_mx = mx.mean(recon_abs_diff_mx)
    
    print(f"\nMaximum reconstruction error (MLX built-in): {recon_max_error_mx}")
    print(f"Mean reconstruction error (MLX built-in): {recon_mean_error_mx}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print("Custom Metal kernel QR:")
    print(f"  Orthogonality error: {max_error}")
    print(f"  Reconstruction error: {recon_max_error}")
    print("MLX built-in QR:")
    print(f"  Orthogonality error: {max_error_mx}")
    print(f"  Reconstruction error: {recon_max_error_mx}")
    
    # Check if the debug info has any non-zero values
    if ops.stats.max(ops.abs(dbg)) > 0:
        print("\nDebug info has non-zero values:")
        for i in range(dbg.shape[0]):
            if abs(dbg[i]) > 0:
                print(f"  dbg[{i}] = {dbg[i]}")
    else:
        print("\nAll debug info values are zero, which suggests the kernel might not be executing properly.")
    
    print("\nPossible issues:")
    print("1. The Metal kernel might not be executing correctly")
    print("2. There might be an issue with the kernel parameters or grid/threadgroup configuration")
    print("3. The kernel might be failing to write results back to the output arrays")

if __name__ == "__main__":
    debug_qr_kernel()