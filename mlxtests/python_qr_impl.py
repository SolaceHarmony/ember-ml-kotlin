#!/usr/bin/env python3
"""
Pure Python implementation of QR decomposition for debugging.
This implementation avoids Metal kernel issues by using a pure Python/MLX approach.
"""
import os
import sys
import time
import mlx.core as mx
import numpy as np

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def python_qr(A, debug=False):
    """
    Pure Python implementation of QR decomposition.
    
    Args:
        A: Input matrix (MLX array)
        debug: Whether to return debug information
        
    Returns:
        (Q, R[, dbg])
    """
    # Convert to numpy for easier processing
    if isinstance(A, mx.array):
        A_np = A.tolist()
        A_np = np.array(A_np, dtype=np.float32)
    else:
        A_np = np.array(A, dtype=np.float32)
    
    # Constants
    EPSILON = 1e-10
    
    # Debug array
    dbg = np.zeros(16, dtype=np.float32)
    
    # Get dimensions
    m, n = A_np.shape
    min_dim = min(m, n)
    
    # Initialize Q as identity and R as copy of A
    Q = np.eye(m, dtype=np.float32)
    R = A_np.copy()
    
    if debug:
        print("Initial Q:")
        print(Q[:4, :4])  # Print a subset for large matrices
        print("Initial R:")
        print(R[:4, :4])  # Print a subset for large matrices
    
    # Main QR algorithm
    for k in range(min_dim):
        if debug:
            print(f"\n--- Column {k} ---")
        
        # Column scaling (improves robustness)
        cmax = np.max(np.abs(R[k:m, k]))
        scale = 1.0 / cmax if cmax > EPSILON else 1.0
        R[k:m, k] *= scale
        
        if k == 0:
            dbg[10] = scale
            dbg[13] = cmax
        
        if debug:
            print(f"Column max: {cmax}, Scale: {scale}")
        
        # Build Householder vector
        sigma = np.sum(R[k:m, k] ** 2)
        if k == 0:
            dbg[4] = sigma
        
        norm = np.sqrt(sigma)
        if k == 0:
            dbg[5] = norm
        
        if debug:
            print(f"Sigma: {sigma}, Norm: {norm}")
        
        # Skip if column is effectively zero
        if norm < EPSILON:
            R[k, k] /= scale
            if debug:
                print("Zero column detected, skipping")
            continue
        
        # Update first element with Householder reflection
        sign = 1.0 if R[k, k] >= 0.0 else -1.0
        R[k, k] += sign * norm
        
        if k == 0:
            dbg[14] = R[k, k]
        
        if debug:
            print(f"Sign: {sign}, Updated R[{k},{k}]: {R[k, k]}")
        
        # Calculate v^T v
        vtv = np.sum(R[k:m, k] ** 2)
        if k == 0:
            dbg[6] = vtv
        
        # Calculate inverse of v^T v
        inv_vtv = 1.0 / vtv if vtv > EPSILON else 0.0
        if k == 0:
            dbg[7] = inv_vtv
        
        if debug:
            print(f"v^T v: {vtv}, inv_vtv: {inv_vtv}")
        
        # Skip reflection if inv_vtv is zero
        if inv_vtv == 0.0:
            R[k:m, k] /= scale
            continue
        
        # Reflect R
        for j in range(k, n):
            dot = np.sum(R[k:m, k] * R[k:m, j])
            if j == k and k == 0:
                dbg[8] = dot
            
            beta = 2.0 * dot * inv_vtv
            R[k:m, j] -= beta * R[k:m, k]
            
            if debug and j == k:
                print(f"R reflection - dot: {dot}, beta: {beta}")
        
        # Reflect Q
        for j in range(m):
            dot = np.sum(R[k:m, k] * Q[k:m, j])
            if j == 0 and k == 0:
                dbg[9] = dot
            
            beta = 2.0 * dot * inv_vtv
            Q[k:m, j] -= beta * R[k:m, k]
            
            if debug and j == 0:
                print(f"Q reflection - dot: {dot}, beta: {beta}")
        
        # Un-scale column k
        R[k:m, k] /= scale
        
        if debug:
            print(f"Unscaled R column {k}")
    
    # Force R upper-triangular
    for r in range(1, m):
        for c in range(min(r, n)):
            R[r, c] = 0.0
    
    # Set success flag
    dbg[15] = 1.0
    
    # Convert back to MLX arrays
    Q_mx = mx.array(Q, dtype=mx.float32)
    R_mx = mx.array(R, dtype=mx.float32)
    dbg_mx = mx.array(dbg, dtype=mx.float32)
    
    return (Q_mx, R_mx, dbg_mx) if debug else (Q_mx, R_mx)

def test_python_qr():
    """Test the Python QR implementation with various matrices."""
    print("=" * 80)
    print("PYTHON QR DECOMPOSITION TEST")
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
        
        # Run Python implementation
        print("\nRunning Python implementation...")
        t0 = time.time()
        Q, R, dbg = python_qr(A, debug=True)
        dt = time.time() - t0
        print(f"Python implementation time: {dt:.6f} seconds")
        
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
    print("Running Python QR decomposition tests...")
    test_python_qr()