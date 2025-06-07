"""
Simple test for the QR decomposition with a small matrix.
"""

import sys
import os
import time
import mlx.core as mx

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the QR function directly from the MLX backend
from ember_ml.backend.mlx.linearalg.qr_ops import qr as mlx_qr

def test_simple_qr():
    """Test the QR decomposition with a small matrix."""
    print("\n=== Simple QR Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Perform QR decomposition with debug=True
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
    print("\nDebug info:")
    print(dbg)
    
    # Check if debug info contains non-zero values
    dbg_nonzero = mx.any(mx.abs(dbg) > 0).item()
    print(f"Debug info contains non-zero values: {dbg_nonzero}")
    
    if dbg_nonzero:
        for i in range(dbg.shape[0]):
            if abs(dbg[i]) > 0:
                print(f"  dbg[{i}] = {dbg[i]}")
    
    # Print Q and R matrices
    print("\nQ matrix:")
    print(q)
    
    print("\nR matrix:")
    print(r)
    
    # Check orthogonality of Q
    q_t = mx.transpose(q)
    q_t_q = mx.matmul(q_t, q)
    identity = mx.eye(q_t_q.shape[0])
    
    print("\nQ^T * Q matrix:")
    print(q_t_q)
    
    # Compute error matrix
    diff = q_t_q - identity
    abs_diff = mx.abs(diff)
    max_error = mx.max(abs_diff)
    mean_error = mx.mean(abs_diff)
    
    print(f"\nMaximum orthogonality error: {max_error}")
    print(f"Mean orthogonality error: {mean_error}")
    
    # Check reconstruction
    recon = mx.matmul(q, r)
    
    print("\nReconstruction matrix:")
    print(recon)
    
    # Compute reconstruction error
    recon_diff = a - recon
    recon_abs_diff = mx.abs(recon_diff)
    recon_max_error = mx.max(recon_abs_diff)
    recon_mean_error = mx.mean(recon_abs_diff)
    
    print(f"\nMaximum reconstruction error: {recon_max_error}")
    print(f"Mean reconstruction error: {recon_mean_error}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Orthogonality error: {max_error}")
    print(f"Reconstruction error: {recon_max_error}")
    
    # Compare with MLX's built-in QR function
    print("\n=== Comparing with MLX's built-in QR function ===")
    
    try:
        # Use MLX's built-in QR function
        q_mx, r_mx = mx.linalg.qr(a, stream=mx.cpu)
        
        print("\nMLX built-in QR function results:")
        print(f"Q shape: {q_mx.shape}")
        print(f"R shape: {r_mx.shape}")
        
        print("\nQ matrix (MLX built-in):")
        print(q_mx)
        
        print("\nR matrix (MLX built-in):")
        print(r_mx)
        
        # Check orthogonality of Q from MLX's built-in function
        q_t_mx = mx.transpose(q_mx)
        q_t_q_mx = mx.matmul(q_t_mx, q_mx)
        identity_mx = mx.eye(q_t_q_mx.shape[0])
        
        print("\nQ^T * Q matrix (MLX built-in):")
        print(q_t_q_mx)
        
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
        print(recon_mx)
        
        # Compute reconstruction error
        recon_diff_mx = a - recon_mx
        recon_abs_diff_mx = mx.abs(recon_diff_mx)
        recon_max_error_mx = mx.max(recon_abs_diff_mx)
        recon_mean_error_mx = mx.mean(recon_abs_diff_mx)
        
        print(f"\nMaximum reconstruction error (MLX built-in): {recon_max_error_mx}")
        print(f"Mean reconstruction error (MLX built-in): {recon_mean_error_mx}")
        
        print("\n=== Comparison Results ===")
        print("Custom Metal kernel QR:")
        print(f"  Orthogonality error: {max_error}")
        print(f"  Reconstruction error: {recon_max_error}")
        print("MLX built-in QR:")
        print(f"  Orthogonality error: {max_error_mx}")
        print(f"  Reconstruction error: {recon_max_error_mx}")
    except Exception as e:
        print(f"Error comparing with MLX's built-in QR function: {e}")

if __name__ == "__main__":
    test_simple_qr()