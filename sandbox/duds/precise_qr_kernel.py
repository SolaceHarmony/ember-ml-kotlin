"""
Precisely configured QR kernel with optimal thread and grid settings.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def precise_qr(A):
    """
    Simplified QR decomposition with precisely configured Metal kernel.
    Focuses on optimal thread utilization matching the actual workload.
    """
    # Convert input to MLX array and ensure contiguity
    A = mx.array(A, dtype=mx.float32).reshape(*A.shape)
    
    # Get dimensions
    m, n = A.shape
    
    # Source code for the Metal kernel
    source = """
        // Get thread ID
        uint tid = thread_position_in_grid.x;
        
        // Get matrix dimensions from input shapes
        const uint m = shape[0];
        const uint n = shape[1];
        
        // Calculate total elements to process
        const uint q_elements = Q_shape[0] * Q_shape[1];
        const uint r_elements = R_shape[0] * R_shape[1];
        const uint total_elements = q_elements + r_elements;
        
        // Set debug values - only thread 0 should set these
        if (tid == 0) {
            dbg[0] = 1.0f;  // Kernel executed
            dbg[1] = float(m);  // Number of rows (from shape)
            dbg[2] = float(n);  // Number of columns (from shape)
            dbg[3] = float(total_elements);  // Total elements to process
            dbg[4] = float(thread_position_in_grid.x);  // Thread ID
            dbg[5] = float(threads_per_threadgroup.x);  // Threads per threadgroup
            dbg[6] = float(grid_size.x);  // Grid size (total threads)
            dbg[15] = 1.0f;  // Success flag
        }
        
        // Process elements using thread ID as linear index
        if (tid < total_elements) {
            // First process Q matrix elements (identity matrix)
            if (tid < q_elements) {
                uint row = tid / Q_shape[1];
                uint col = tid % Q_shape[1];
                Q_out[tid] = (row == col) ? 1.0f : 0.0f;
            }
            // Then process R matrix elements (copy from A)
            else {
                uint r_idx = tid - q_elements;
                // Calculate row and column in A for linear index r_idx
                uint a_row = r_idx / shape[1];
                uint a_col = r_idx % shape[1];
                // Calculate linear index in A
                uint a_linear_idx = a_row * shape[1] + a_col;
                
                R_out[r_idx] = A[a_linear_idx];
            }
        }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="precise_qr",
        input_names=["A", "shape", "Q_shape", "R_shape"],
        output_names=["Q_out", "R_out", "dbg"],
        source=source,
    )
    
    # Prepare inputs
    shape = mx.array([m, n], dtype=mx.uint32)
    q_shape = mx.array([m, m], dtype=mx.uint32)
    r_shape = mx.array([m, n], dtype=mx.uint32)
    
    # Calculate optimal grid and threadgroup sizes based on actual workload
    total_elements = m*m + m*n  # Total elements to process
    
    # Use a reasonable threadgroup size (multiple of 32 for SIMD efficiency)
    threads_per_group = 32  # Start with minimal SIMD width
    
    # Calculate grid size to cover all elements with exactly one thread per element
    grid_x = (total_elements + threads_per_group - 1) // threads_per_group
    
    # Ensure we have at least one grid
    grid_x = max(1, grid_x)
    
    # Ensure we don't exceed device limits
    # Set grid size to the total number of elements to process
    grid = (total_elements, 1, 1)
    # Use a reasonable threadgroup size (multiple of 32 for SIMD efficiency)
    # Ensure threadgroup size does not exceed grid size in any dimension
    threadgroup_x = min(threads_per_group, total_elements)
    threadgroup = (threadgroup_x, 1, 1)
    
    # Prepare total_threads_input (not needed anymore as grid size is total elements)
    # total_threads_input = mx.array([grid[0]], dtype=mx.uint32)
    
    print(f"Input matrix: {m}x{n}")
    print(f"Total elements to process: {total_elements}")
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    print(f"Total threads launched: {grid[0]}")
    
    # Prepare outputs
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling precisely configured QR kernel...")
    start_time = time.time()
    outputs = kernel(
        inputs=[A, shape, q_shape, r_shape], # Remove total_threads_input
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

def test_precise_qr():
    """Test the precisely configured QR decomposition."""
    print("\n=== Precisely Configured QR Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Perform precisely configured QR decomposition
    print("\nPerforming precisely configured QR decomposition...")
    q, r, dbg = precise_qr(a)
    
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
    test_precise_qr()