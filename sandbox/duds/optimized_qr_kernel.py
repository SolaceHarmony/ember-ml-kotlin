"""
Optimized QR kernel based on the debugging report recommendations.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def optimized_qr(A):
    """
    Simplified QR decomposition with optimized Metal kernel implementation.
    Incorporates best practices for Metal kernel execution in MLX.
    """
    # Convert input to MLX array and ensure contiguity
    A = mx.array(A, dtype=mx.float32).reshape(*A.shape)
    
    # Get dimensions
    m, n = A.shape
    
    # Source code for the Metal kernel
    source = """
        // Get thread ID and dimensions
        uint tid = thread_position_in_grid.x;
        uint3 tpg = threads_per_threadgroup;
        uint3 gpg = grid_size;
        uint total_threads = gpg.x * tpg.x;
        
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
        
        // Add memory barrier to ensure visibility
        threadgroup_barrier(mem_flags::mem_device);
        
        // Initialize Q to identity matrix with proper indexing
        for (uint idx = tid; idx < m * m; idx += total_threads) {
            uint row = idx / m;
            uint col = idx % m;
            // Only set diagonal elements to 1, rest are already 0 from initialization
            if (row == col) {
                Q_out[idx] = 1.0f;
            }
        }
        
        // Copy A to R with bounds checking
        for (uint idx = tid; idx < m * n; idx += total_threads) {
            if (idx < m * n) {
                R_out[idx] = A[idx];
            }
        }
        
        // Final barrier to ensure all writes are complete
        threadgroup_barrier(mem_flags::mem_device);
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="optimized_qr",
        input_names=["A", "shape"],
        output_names=["Q_out", "R_out", "dbg"],
        source=source,
    )
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    
    # Calculate optimal grid and threadgroup sizes
    elements = max(m*m, m*n)
    threads_per_group = min(256, elements)  # Device maximum is typically 1024
    grid_x = (elements + threads_per_group - 1) // threads_per_group
    grid = (max(1, grid_x), 1, 1)
    threadgroup = (threads_per_group, 1, 1)
    
    print(f"Input shape: {A.shape}")
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    print(f"Total threads: {grid[0] * threadgroup[0]}")
    
    # Prepare outputs
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling optimized QR kernel...")
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

def test_optimized_qr():
    """Test the optimized QR decomposition."""
    print("\n=== Optimized QR Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Perform optimized QR decomposition
    print("\nPerforming optimized QR decomposition...")
    q, r, dbg = optimized_qr(a)
    
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
    
    # Compare with MLX's built-in QR function
    try:
        print("\n=== Comparing with MLX's built-in QR function ===")
        q_mx, r_mx = mx.linalg.qr(a, stream=mx.cpu)
        
        print(f"Q shape (MLX built-in): {q_mx.shape}")
        print(f"R shape (MLX built-in): {r_mx.shape}")
        
        print("\nQ matrix (MLX built-in):")
        print(q_mx)
        
        print("\nR matrix (MLX built-in):")
        print(r_mx)
        
        # Check reconstruction
        recon = mx.matmul(q_mx, r_mx)
        recon_error = mx.max(mx.abs(recon - a)).item()
        print(f"\nReconstruction error (MLX built-in): {recon_error}")
    except Exception as e:
        print(f"Error comparing with MLX's built-in QR function: {e}")

if __name__ == "__main__":
    # Set environment variables for Metal debugging
    os.environ["MTL_DEBUG_LAYER"] = "1"
    os.environ["MTL_SHADER_VALIDATION"] = "1"
    
    test_optimized_qr()