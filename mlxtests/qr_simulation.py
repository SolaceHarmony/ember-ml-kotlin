#!/usr/bin/env python3
"""
Python simulation of the Metal QR decomposition kernel to debug zero matrices issue.
This simulates the exact behavior of the Metal kernel to identify where zeros might be introduced.
"""
import os
import sys
import time
import numpy as np
import mlx.core as mx

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original implementation for comparison
from mlxtests.hpc_16x8_method.enhanced_qr_decomp import enhanced_tiled_qr

def simulate_metal_qr(A, debug=False):
    """
    Python simulation of the Metal kernel for QR decomposition.
    This follows the exact same algorithm as the Metal kernel.
    """
    # Constants from the Metal kernel
    EPSILON = 1e-10
    NUM_LIMBS = 8
    LIMB_RADIX = 65536.0  # 2^16
    
    # Convert to numpy for easier debugging
    if isinstance(A, mx.array):
        A_np = A.tolist()
        A_np = np.array(A_np, dtype=np.float32)
    else:
        A_np = np.array(A, dtype=np.float32)
    
    m, n = A_np.shape
    min_dim = min(m, n)
    
    # Debug array
    dbg = np.zeros(16, dtype=np.float32)
    
    # Initialize Q as identity and R as copy of A
    Q = np.eye(m, dtype=np.float32)
    R = A_np.copy()
    
    # Print initial values for debugging
    if debug:
        print("Initial Q:")
        print(Q[:4, :4])  # Print a subset for large matrices
        print("Initial R:")
        print(R[:4, :4])  # Print a subset for large matrices
    
    # Main QR algorithm (single-thread numerical core)
    for k in range(min_dim):
        if debug:
            print(f"\n--- Column {k} ---")
        
        # Column scaling (improves robustness)
        cmax = np.max(np.abs(R[k:m, k]))
        scale = 1.0 / cmax if cmax > EPSILON else 1.0
        R[k:m, k] *= scale
        dbg[10] = scale
        
        if debug:
            print(f"Column max: {cmax}, Scale: {scale}")
            print(f"Scaled column {k}:")
            print(R[k:min(k+4, m), k])
        
        # Build Householder vector
        sigma = np.sum(R[k:m, k] ** 2)
        dbg[4] = sigma
        norm = np.sqrt(sigma)
        dbg[5] = norm
        
        if debug:
            print(f"Sigma: {sigma}, Norm: {norm}")
        
        if norm < EPSILON:  # Zero column
            R[k, k] /= scale
            if debug:
                print(f"Zero column detected, skipping (norm < EPSILON)")
            continue
        
        sign = 1.0 if R[k, k] >= 0.0 else -1.0
        R[k, k] += sign * norm  # v₀ update
        
        if debug:
            print(f"Sign: {sign}, Updated R[{k},{k}]: {R[k, k]}")
        
        # Simulate limb-precision vᵀv calculation
        # This is a high-precision dot product implementation
        # For simulation, we'll use numpy's high precision
        vtv = np.sum(R[k:m, k] ** 2)
        dbg[6] = vtv
        inv_vtv = 0.0 if vtv < EPSILON else 1.0 / vtv
        dbg[7] = inv_vtv
        
        if debug:
            print(f"vᵀv: {vtv}, inv_vtv: {inv_vtv}")
        
        # Reflect R (k ... n-1)
        for j in range(k, n):
            dot = np.sum(R[k:m, k] * R[k:m, j])
            if j == k:
                dbg[8] = dot
            beta = 2.0 * dot * inv_vtv
            R[k:m, j] -= beta * R[k:m, k]
            
            if debug and j == k:
                print(f"R reflection - dot: {dot}, beta: {beta}")
                print(f"Updated R column {k} after reflection:")
                print(R[k:min(k+4, m), k])
        
        # Reflect Q (0 ... m-1)
        for j in range(m):
            dot = np.sum(R[k:m, k] * Q[k:m, j])
            if j == 0:
                dbg[9] = dot
            beta = 2.0 * dot * inv_vtv
            Q[k:m, j] -= beta * R[k:m, k]
            
            if debug and j == 0:
                print(f"Q reflection - dot: {dot}, beta: {beta}")
                print(f"Updated Q column 0 after reflection:")
                print(Q[k:min(k+4, m), 0])
        
        # Un-scale column k
        R[k:m, k] /= scale
        
        if debug:
            print(f"Unscaled R column {k}:")
            print(R[k:min(k+4, m), k])
    
    # Force R upper-triangular
    for r in range(1, m):
        for c in range(min(r, n)):
            R[r, c] = 0.0
    
    dbg[15] = 0.0  # success flag
    
    # Convert back to MLX arrays for consistency
    Q_mx = mx.array(Q, dtype=mx.float32)
    R_mx = mx.array(R, dtype=mx.float32)
    dbg_mx = mx.array(dbg, dtype=mx.float32)
    
    return (Q_mx, R_mx, dbg_mx) if debug else (Q_mx, R_mx)

def compare_implementations(A, debug=True):
    """Compare the Python simulation with the Metal kernel implementation."""
    print("=" * 80)
    print("COMPARING PYTHON SIMULATION VS METAL KERNEL")
    print("=" * 80)
    
    # Convert to MLX array if needed
    if not isinstance(A, mx.array):
        A_mx = mx.array(A, dtype=mx.float32)
    else:
        A_mx = A
    
    # Run Python simulation
    print("\nRunning Python simulation...")
    t0 = time.time()
    Q_py, R_py, dbg_py = simulate_metal_qr(A_mx, debug=debug)
    dt_py = time.time() - t0
    print(f"Python simulation time: {dt_py:.6f} seconds")
    
    # Run Metal kernel
    print("\nRunning Metal kernel...")
    t0 = time.time()
    Q_metal, R_metal, dbg_metal = enhanced_tiled_qr(A_mx, debug=True)
    dt_metal = time.time() - t0
    print(f"Metal kernel time: {dt_metal:.6f} seconds")
    
    # Check if matrices contain all zeros
    q_py_zeros = mx.all(Q_py == 0.0).item()
    r_py_zeros = mx.all(R_py == 0.0).item()
    q_metal_zeros = mx.all(Q_metal == 0.0).item()
    r_metal_zeros = mx.all(R_metal == 0.0).item()
    
    print("\nZero matrices check:")
    print(f"Python Q contains all zeros: {q_py_zeros}")
    print(f"Python R contains all zeros: {r_py_zeros}")
    print(f"Metal Q contains all zeros: {q_metal_zeros}")
    print(f"Metal R contains all zeros: {r_metal_zeros}")
    
    # Compare debug values
    print("\nDebug values comparison:")
    for i in range(16):
        py_val = dbg_py[i].item()
        metal_val = dbg_metal[i].item()
        print(f"dbg[{i}]: Python={py_val:.6e}, Metal={metal_val:.6e}, Diff={abs(py_val-metal_val):.6e}")
    
    # Compare first few values of Q and R
    print("\nFirst few values of Q (Python):")
    print(Q_py[:2, :2])
    print("\nFirst few values of Q (Metal):")
    print(Q_metal[:2, :2])
    
    print("\nFirst few values of R (Python):")
    print(R_py[:2, :2])
    print("\nFirst few values of R (Metal):")
    print(R_metal[:2, :2])
    
    # Check orthogonality and reconstruction for Python simulation
    if not q_py_zeros:
        QTQ_py = mx.matmul(Q_py.T, Q_py)
        ortho_error_py = mx.mean(mx.abs(QTQ_py - mx.eye(Q_py.shape[0]))).item()
        QR_py = mx.matmul(Q_py, R_py)
        recon_error_py = mx.mean(mx.abs(QR_py - A_mx)).item()
        print(f"\nPython - Orthogonality error: {ortho_error_py:.8f}")
        print(f"Python - Reconstruction error: {recon_error_py:.8f}")
    
    # Check orthogonality and reconstruction for Metal kernel
    if not q_metal_zeros:
        QTQ_metal = mx.matmul(Q_metal.T, Q_metal)
        ortho_error_metal = mx.mean(mx.abs(QTQ_metal - mx.eye(Q_metal.shape[0]))).item()
        QR_metal = mx.matmul(Q_metal, R_metal)
        recon_error_metal = mx.mean(mx.abs(QR_metal - A_mx)).item()
        print(f"\nMetal - Orthogonality error: {ortho_error_metal:.8f}")
        print(f"Metal - Reconstruction error: {recon_error_metal:.8f}")
    
    return Q_py, R_py, Q_metal, R_metal

def run_test_cases():
    """Run test cases to identify when zeros occur."""
    print("=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)
    
    # Test case 1: Small well-conditioned matrix
    print("\nTest Case 1: Small well-conditioned matrix (4x4)")
    A1 = mx.array([
        [4.0, 1.0, -2.0, 2.0],
        [1.0, 2.0, 0.0, 1.0],
        [-2.0, 0.0, 3.0, -2.0],
        [2.0, 1.0, -2.0, -1.0]
    ], dtype=mx.float32)
    
    compare_implementations(A1, debug=True)
    
    # Test case 2: Matrix with exact values from the example
    print("\nTest Case 2: Sample from the failed case")
    A2 = mx.array([
        [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
        [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
        [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
        [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
        [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
    ], dtype=mx.float32)
    
    compare_implementations(A2, debug=True)

if __name__ == "__main__":
    # Set environment variables to control Metal execution
    os.environ["MLX_USE_METAL"] = "1"
    print("Running QR decomposition simulation...")
    run_test_cases()