#!/usr/bin/env python3
"""
Diagnostic script for QR decomposition in MLX backend.
Tests different matrix sizes and thread configurations to identify issues.
Uses MLX exclusively without any NumPy dependencies.
"""
import os
import sys
import time
import mlx.core as mx

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the working implementation directly from hpc_16x8_method
from mlxtests.hpc_16x8_method.enhanced_qr_decomp import enhanced_tiled_qr, _ENHANCED_QR_KERNEL

def test_qr_decomposition():
    """Test QR decomposition with various matrix configurations."""
    print("=" * 80)
    print("QR DECOMPOSITION DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Test case 1: Small well-conditioned matrix
    print("\nTest Case 1: Small well-conditioned matrix (4x4)")
    A1 = mx.array([
        [4.0, 1.0, -2.0, 2.0],
        [1.0, 2.0, 0.0, 1.0],
        [-2.0, 0.0, 3.0, -2.0],
        [2.0, 1.0, -2.0, -1.0]
    ], dtype=mx.float32)
    
    # Test with enhanced_tiled_qr
    print("Running with enhanced_tiled_qr from hpc_16x8_method...")
    q1, r1, dbg1 = enhanced_tiled_qr(A1, debug=True)
    
    # Interpret debug values
    print("\nThread Synchronization Debug Values:")
    print(f"Thread ID: {dbg1[0]}")
    print(f"Threadgroup size: {dbg1[1]}")
    print(f"Grid size: {dbg1[2]}")
    print(f"Total threads: {dbg1[3]}")
    print(f"Sigma: {dbg1[4]}")
    print(f"Norm: {dbg1[5]}")
    print(f"Scale: {dbg1[10]}")
    print(f"Success flag: {dbg1[15]} (1.0 = success)")
    
    print(f"Q contains zeros only: {mx.all(q1 == 0.0).item()}")
    print(f"R contains zeros only: {mx.all(r1 == 0.0).item()}")
    
    # Only verify QR properties if non-zero results
    if not mx.all(q1 == 0.0).item():
        # Verify QR property: A ≈ Q·R
        reconstruct = mx.matmul(q1, r1)
        error = mx.mean(mx.abs(reconstruct - A1)).item()
        print(f"Reconstruction error: {error}")
        
        # Verify orthogonality of Q
        ortho_error = mx.mean(mx.abs(mx.matmul(q1.T, q1) - mx.eye(q1.shape[0]))).item()
        print(f"Orthogonality error: {ortho_error}")
    
    # Test case 2: Medium-sized matrix
    print("\nTest Case 2: Medium-sized random matrix (32x32)")
    mx.random.seed(42)
    A2 = mx.random.normal((32, 32))
    
    # Test various thread and grid configurations
    configurations = [
        ((256, 1, 1), (256, 1, 1), "Default (256x256)"),
        ((512, 1, 1), (512, 1, 1), "Large (512x512)"),
        ((128, 1, 1), (128, 1, 1), "Small (128x128)"),
        ((1024, 1, 1), (256, 1, 1), "Wide Grid (1024x256)")
    ]
    
    for grid_size, thread_size, config_name in configurations:
        print(f"\nTesting {config_name} configuration...")
        print(f"Grid: {grid_size}, Thread: {thread_size}")
        
        # Custom kernel call with specific configuration
        m, n = A2.shape
        shape = mx.array([m, n], dtype=mx.uint32)
        dbg = mx.zeros((16,), dtype=mx.float32)
        
        t0 = time.time()
        Q, R, dbg = _ENHANCED_QR_KERNEL(
            inputs = [A2, shape],
            output_shapes = [(m, m), (m, n), (16,)],
            output_dtypes = [mx.float32, mx.float32, mx.float32],
            grid = grid_size,
            threadgroup = thread_size
        )
        dt = time.time() - t0
        
        print(f"Execution time: {dt:.3f} seconds")
        
        # Interpret debug values
        print("\nThread Synchronization Debug:")
        print(f"Thread ID: {dbg[0]}")
        print(f"Sigma: {dbg[4]}")
        print(f"Norm: {dbg[5]}")
        print(f"Scale: {dbg[10]}")
        print(f"Success flag: {dbg[15]} (1.0 = success)")
        
        print(f"Q contains zeros only: {mx.all(Q == 0.0).item()}")
        print(f"R contains zeros only: {mx.all(R == 0.0).item()}")
        
        if not mx.all(Q == 0.0).item():
            # Verify QR property: A ≈ Q·R
            reconstruct = mx.matmul(Q, R)
            error = mx.mean(mx.abs(reconstruct - A2)).item()
            print(f"Reconstruction error: {error}")
            
            # Verify orthogonality of Q
            ortho_error = mx.mean(mx.abs(mx.matmul(Q.T, Q) - mx.eye(Q.shape[0]))).item()
            print(f"Orthogonality error: {ortho_error}")
    
    # Test case 3: Matrices with problematic numerical properties
    print("\nTest Case 3: Matrices with problematic numerical properties")
    
    # Test 3a: Nearly singular matrix
    print("\nTest 3a: Nearly singular matrix")
    A3a = mx.eye(10, dtype=mx.float32)
    A3a = mx.array_update(A3a, mx.array([1e-7], dtype=mx.float32), (0, 0))
    
    q3a, r3a, dbg3a = enhanced_tiled_qr(A3a, debug=True)
    print(f"Debug values: {dbg3a}")
    print(f"Q contains zeros only: {mx.all(q3a == 0.0).item()}")
    print(f"R contains zeros only: {mx.all(r3a == 0.0).item()}")
    
    # Test 3b: Ill-conditioned matrix
    print("\nTest 3b: Ill-conditioned matrix")
    # Create Hilbert matrix (known to be ill-conditioned)
    n = 10
    A3b = mx.zeros((n, n), dtype=mx.float32)
    for i in range(n):
        for j in range(n):
            A3b = mx.array_update(A3b, mx.array([1.0 / (i + j + 1)], dtype=mx.float32), (i, j))
    
    q3b, r3b, dbg3b = enhanced_tiled_qr(A3b, debug=True)
    print(f"Debug values: {dbg3b}")
    print(f"Q contains zeros only: {mx.all(q3b == 0.0).item()}")
    print(f"R contains zeros only: {mx.all(r3b == 0.0).item()}")
    
    # Test 3c: Matrix with exact values from your example
    print("\nTest 3c: Sample from your failed case")
    # Create a sample with values in the range from your example
    A3c = mx.array([
        [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
        [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
        [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
        [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
        [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
    ], dtype=mx.float32)
    
    q3c, r3c, dbg3c = enhanced_tiled_qr(A3c, debug=True)
    print(f"Debug values: {dbg3c}")
    print(f"Q contains zeros only: {mx.all(q3c == 0.0).item()}")
    print(f"R contains zeros only: {mx.all(r3c == 0.0).item()}")
    
    # Test with your specific example again but with detailed verification
    if not mx.all(q3c == 0.0).item():
        print("\nDetailed analysis of your specific example:")
        # Check R is upper triangular
        is_upper = True
        for i in range(1, min(A3c.shape)):
            for j in range(i):
                if mx.abs(r3c[i, j]).item() > 1e-5:
                    is_upper = False
                    break
        print(f"R is upper triangular: {is_upper}")
        
        # Check Q is orthogonal
        qTq = mx.matmul(q3c.T, q3c)
        eye = mx.eye(q3c.shape[0])
        ortho_error = mx.mean(mx.abs(qTq - eye)).item()
        print(f"Q orthogonality error: {ortho_error}")
        
        # Check A = QR
        qr_product = mx.matmul(q3c, r3c)
        recon_error = mx.mean(mx.abs(qr_product - A3c)).item()
        print(f"Reconstruction error: {recon_error}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Set environment variables to control Metal execution
    os.environ["MLX_USE_METAL"] = "1"
    print("Running QR decomposition diagnostic tests...")
    test_qr_decomposition()