#!/usr/bin/env python3
"""
Simplified QR test to examine the actual values produced by the Metal kernel.
Uses MLX exclusively without any NumPy dependencies.
"""
import os
import sys
import time
import mlx.core as mx

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the working implementation directly
from mlxtests.hpc_16x8_method.enhanced_qr_decomp import enhanced_tiled_qr

def simple_qr_test():
    """Simple test to check QR values directly using MLX only."""
    print("=" * 80)
    print("SIMPLE QR TEST (MLX-ONLY)")
    print("=" * 80)
    
    # Set Metal environment
    os.environ["MLX_USE_METAL"] = "1"
    
    # Create a small test matrix using MLX
    A = mx.array([
        [4.0, 1.0, -2.0, 2.0],
        [1.0, 2.0, 0.0, 1.0],
        [-2.0, 0.0, 3.0, -2.0],
        [2.0, 1.0, -2.0, -1.0]
    ], dtype=mx.float32)
    
    # Test QR decomposition
    print("Running QR decomposition...")
    t0 = time.time()
    Q, R, dbg = enhanced_tiled_qr(A, debug=True)
    dt = time.time() - t0
    
    print(f"Execution time: {dt:.6f} seconds")
    
    # Check if matrices contain all zeros
    q_all_zeros = mx.all(Q == 0.0).item()
    r_all_zeros = mx.all(R == 0.0).item()
    
    print(f"Q contains all zeros: {q_all_zeros}")
    print(f"R contains all zeros: {r_all_zeros}")
    
    # Print first few values of Q and R
    print("\nFirst few values of Q:")
    print(Q[:2, :2])
    
    print("\nFirst few values of R:")
    print(R[:2, :2])
    
    # Check orthogonality
    QTQ = mx.matmul(Q.T, Q)
    ortho_error = mx.mean(mx.abs(QTQ - mx.eye(Q.shape[0]))).item()
    print(f"\nOrthogonality error (‖QᵀQ−I‖₁): {ortho_error:.8f}")
    
    # Check reconstruction
    QR = mx.matmul(Q, R)
    recon_error = mx.mean(mx.abs(QR - A)).item()
    print(f"Reconstruction error (‖QR−A‖₁): {recon_error:.8f}")
    
    # Calculate mean absolute values of Q and R
    q_mean = mx.mean(mx.abs(Q)).item()
    r_mean = mx.mean(mx.abs(R)).item()
    print(f"\nMean absolute value in Q: {q_mean:.8f}")
    print(f"Mean absolute value in R: {r_mean:.8f}")
    
    # Print debug values
    print("\nDebug values:")
    for i, v in enumerate(dbg):
        print(f"dbg[{i}]: {v}")
    
    # Example values from self-test for comparison
    print("\nSelf-test example:")
    # Create random matrices of different sizes
    mx.random.seed(42)
    for (m, n) in [(10, 10), (100, 50)]:
        A_test = mx.random.normal((m, n))
        Q_test, R_test = enhanced_tiled_qr(A_test)
        
        ortho_test = mx.mean(mx.abs(mx.matmul(Q_test.T, Q_test) - mx.eye(m))).item()
        recon_test = mx.mean(mx.abs(mx.matmul(Q_test, R_test) - A_test)).item()
        print(f"{m:4d}×{n:<4d}  ‖QᵀQ−I‖₁={ortho_test:9.2e}   ‖QR−A‖₁={recon_test:9.2e}")
        
    # Test with your specific example
    print("\nTest with specific example matrix:")
    A_example = mx.array([
        [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
        [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
        [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
        [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
        [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
    ], dtype=mx.float32)
    
    Q_example, R_example, dbg_example = enhanced_tiled_qr(A_example, debug=True)
    
    # Check and display results
    q_zeros = mx.all(Q_example == 0.0).item()
    r_zeros = mx.all(R_example == 0.0).item()
    print(f"Q contains all zeros: {q_zeros}")
    print(f"R contains all zeros: {r_zeros}")
    
    # Only analyze non-zero results
    if not q_zeros or not r_zeros:
        # Check if R is upper triangular
        m, n = A_example.shape
        is_upper = True
        for i in range(1, min(m, n)):
            if mx.any(mx.abs(R_example[i, :i]) > 1e-5).item():
                is_upper = False
                break
        print(f"R is upper triangular: {is_upper}")
        
        # Check Q is orthogonal
        qTq = mx.matmul(Q_example.T, Q_example)
        eye = mx.eye(Q_example.shape[0])
        ortho_error = mx.mean(mx.abs(qTq - eye)).item()
        print(f"Q orthogonality error: {ortho_error}")
        
        # Check A = QR
        qr_product = mx.matmul(Q_example, R_example)
        recon_error = mx.mean(mx.abs(qr_product - A_example)).item()
        print(f"Reconstruction error: {recon_error}")

if __name__ == "__main__":
    simple_qr_test()