"""
Simple matrix kernel that initializes a small matrix.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def simple_matrix_kernel():
    """
    Create a simple matrix kernel that initializes a small matrix.
    """
    # Source code for the Metal kernel
    source = """
        // Get thread ID
        uint tid = thread_position_in_grid.x;
        
        // Set debug values - only thread 0 should set these
        if (tid == 0) {
            dbg[0] = 1.0f;  // Success flag
            dbg[1] = float(thread_position_in_grid.x);  // Thread ID
            dbg[2] = float(threads_per_threadgroup.x);  // Threads per threadgroup
            dbg[3] = float(grid_size.x);  // Grid size
        }
        
        // Initialize matrix - each thread handles one element
        if (tid < 9) {  // 3x3 matrix has 9 elements
            uint row = tid / 3;
            uint col = tid % 3;
            
            // Set diagonal elements to 1, others to 0
            matrix[tid] = (row == col) ? 1.0f : 0.0f;
        }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="simple_matrix",
        input_names=[],
        output_names=["matrix", "dbg"],
        source=source,
    )
    
    # Configure kernel execution - use enough threads for a 3x3 matrix
    grid = (1, 1, 1)
    threadgroup = (9, 1, 1)  # Exactly 9 threads for 9 elements
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(3, 3), (4,)]
    output_dtypes = [mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling simple matrix kernel...")
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
    simple_matrix_kernel()