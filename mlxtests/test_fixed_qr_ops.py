#!/usr/bin/env python3
"""
Test script for the fixed QR decomposition implementation.
"""
import os
import sys
import time
import mlx.core as mx

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the fixed QR implementation
from ember_ml.backend.mlx.linearalg.qr_ops import qr

def test_fixed_qr():
    """Test the fixed QR implementation with various matrices."""
    print("=" * 80)
    print("FIXED QR DECOMPOSITION TEST")
    print("=" * 80)
    
    # Test matrices
    test_matrices = [
        # Small well-conditioned matrix
        {
            "name": "Small well-conditioned matrix (4x4)",
            "matrix": mx.array([
                [4.0, 1.0, -2.0, 2.0],
                [1.0, 2.0, 0.0, 1.0],
                [-2.0, 0.0, 3.0, -2.0],
                [2.0, 1.0, -2.0, -1.0]
            ], dtype=mx.float32)
        },
        # Example matrix that failed
        {
            "name": "Example matrix that failed (5x5)",
            "matrix": mx.array([
                [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
                [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
                [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
                [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
                [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
            ], dtype=mx.float32)
        },
        # Random matrix
        {
            "name": "Random matrix (10x10)",
            "matrix": mx.random.normal((10, 10))
        }
    ]
    
    # Run tests for each matrix
    for test_case in test_matrices:
        print("\n" + "=" * 80)
        print(f"TESTING: {test_case['name']}")
        print("=" * 80)
        
        A = test_case["matrix"]
        
        # Run fixed implementation
        print("\nRunning fixed implementation...")
        t0 = time.time()
        Q, R, dbg = qr(A, debug=True)
        dt = time.time() - t0
        print(f"Fixed implementation time: {dt:.6f} seconds")
        
        # Check if matrices contain all zeros
        q_zeros = mx.all(Q == 0.0).item()
        r_zeros = mx.all(R == 0.0).item()
        print(f"Q contains all zeros: {q_zeros}")
        print(f"R contains all zeros: {r_zeros}")
        
        if not q_zeros:
            # Verify QR property: A ≈ Q·R
            reconstruct = mx.matmul(Q, R)
            error = mx.mean(mx.abs(reconstruct - A)).item()
            print(f"Reconstruction error: {error}")
            
            # Verify orthogonality of Q
            identity = mx.eye(Q.shape[0])
            ortho_error = mx.mean(mx.abs(mx.matmul(Q.T, Q) - identity)).item()
            print(f"Orthogonality error: {ortho_error}")
            
            # Print first few values of Q and R
            print("\nFirst few values of Q:")
            print(Q[:2, :2])
            
            print("\nFirst few values of R:")
            print(R[:2, :2])
        
        # Print debug values
        print("\nDebug values:")
        for i, v in enumerate(dbg):
            print(f"dbg[{i}]: {v.item()}")

if __name__ == "__main__":
    # Set environment variables to control Metal execution
    os.environ["MLX_USE_METAL"] = "1"
    print("Running fixed QR decomposition tests...")
    test_fixed_qr()