"""
Grid-stride loop kernel that initializes a matrix.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def grid_stride_kernel():
    """
    Create a kernel that uses a grid-stride loop to initialize a matrix.
    """
    # Source code for the Metal kernel
    source = """
        // Get thread ID and dimensions
        uint tid = thread_position_in_grid.x;
        uint total_threads = grid_size.x * threads_per_threadgroup.x;
        
        // Set debug values - only thread 0 should set these
        if (tid == 0) {
            dbg[0] = 1.0f;  // Success flag
            dbg[1] = float(thread_position_in_grid.x);  // Thread ID
            dbg[2] = float(threads_per_threadgroup.x);  // Threads per threadgroup
            dbg[3] = float(grid_size.x);  // Grid size
            dbg[4] = float(total_threads);  // Total threads
        }
        
        // Initialize matrix using grid-stride loop
        // Each thread processes elements at positions tid, tid+total_threads, tid+2*total_threads, etc.
        for (uint i = tid; i < 9; i += total_threads) {
            uint row = i / 3;
            uint col = i % 3;
            
            // Set diagonal elements to 1, others to 0
            matrix[i] = (row == col) ? 1.0f : 0.0f;
        }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="grid_stride",
        input_names=[],
        output_names=["matrix", "dbg"],
        source=source,
    )
    
    # Configure kernel execution - use a small number of threads
    grid = (1, 1, 1)
    threadgroup = (4, 1, 1)  # Use 4 threads (less than the 9 elements)
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    print(f"Total threads: {grid[0] * threadgroup[0]}")
    
    # Prepare outputs
    output_shapes = [(3, 3), (5,)]
    output_dtypes = [mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling grid-stride kernel...")
    start_time = time.time()
    outputs = kernel(
        inputs=[],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        verbose=True  # Print generated code for debugging
    )
    end_time = time.time()
    print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get outputs
    matrix, dbg = outputs
    
    # Print outputs
    print(f"Debug info: {dbg}")
    print(f"Matrix:\n{matrix}")
    
    # Check if matrix is identity
    identity = mx.eye(3)
    matrix_is_identity = mx.all(mx.abs(matrix - identity) < 1e-6).item()
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if dbg[0] == 1.0 else 'FAILURE'}")
    print(f"Matrix is identity: {'SUCCESS' if matrix_is_identity else 'FAILURE'}")
    
    return matrix_is_identity

if __name__ == "__main__":
    grid_stride_kernel()