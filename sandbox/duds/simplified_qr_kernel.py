"""
Simplified QR kernel based on the official MLX documentation.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def simplified_qr(A):
    """
    Simplified QR decomposition that just initializes Q to identity and R to A.
    Uses proper Metal kernel syntax based on the documentation.
    """
    # Convert input to MLX array
    A = mx.array(A, dtype=mx.float32)
    
    # Get dimensions
    m, n = A.shape
    
    # Source code for the Metal kernel - only the body of the function
    source = """
        // Get thread ID
        uint tid = thread_position_in_grid.x;
        uint total_threads = grid_size.x * threads_per_threadgroup.x;
        
        // Get matrix dimensions
        const uint m = shape[0];
        const uint n = shape[1];
        
        // Set debug values - only thread 0 should set these
        if (tid == 0) {
            dbg[0] = 1.0f;  // Kernel executed
            dbg[1] = float(m);  // Number of rows
            dbg[2] = float(n);  // Number of columns
            dbg[3] = float(total_threads);  // Total number of threads
            dbg[15] = 1.0f;  // Success flag
        }
        
        // Initialize Q to identity matrix
        for (uint idx = tid; idx < m * m; idx += total_threads) {
            uint row = idx / m;
            uint col = idx % m;
            Q_out[idx] = (row == col) ? 1.0f : 0.0f;
        }
        
        // Copy A to R
        for (uint idx = tid; idx < m * n; idx += total_threads) {
            R_out[idx] = A[idx];
        }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="simplified_qr",
        input_names=["A", "shape"],
        output_names=["Q_out", "R_out", "dbg"],
        source=source,
    )
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    print(f"Input shape: {A.shape}")
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling simplified QR kernel...")
    start_time = time.time()
    outputs = kernel(
        inputs=[A, shape],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        verbose=True  # Print generated code for debugging
    )
    end_time = time.time()
    print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get outputs
    Q, R, dbg = outputs
    
    return Q, R, dbg

def test_simplified_qr():
    """Test the simplified QR decomposition."""
    print("\n=== Simplified QR Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Perform simplified QR decomposition
    print("\nPerforming simplified QR decomposition...")
    q, r, dbg = simplified_qr(a)
    
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
    
    # Check if Q is identity
    identity = mx.eye(q.shape[0])
    q_is_identity = mx.all(mx.abs(q - identity) < 1e-6).item()
    print(f"Q is identity: {q_is_identity}")
    
    # Check if R is equal to A
    r_equals_a = mx.all(mx.abs(r - a) < 1e-6).item()
    print(f"R equals A: {r_equals_a}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if dbg_nonzero else 'FAILURE'}")
    print(f"Q is identity: {'SUCCESS' if q_is_identity else 'FAILURE'}")
    print(f"R equals A: {'SUCCESS' if r_equals_a else 'FAILURE'}")

if __name__ == "__main__":
    test_simplified_qr()