"""
Minimal working kernel that just sets a few values.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def minimal_working_kernel():
    """
    Create a minimal working kernel that just sets a few values.
    """
    # Source code for the Metal kernel
    source = """
        // Get thread ID
        uint tid = thread_position_in_grid.x;
        
        // Set output values - only thread 0 should set these
        if (tid == 0) {
            out[0] = 1.0f;  // Success flag
            out[1] = float(thread_position_in_grid.x);  // Thread ID
            out[2] = float(threads_per_threadgroup.x);  // Threads per threadgroup
            out[3] = float(grid_size.x);  // Grid size
        }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="minimal_working",
        input_names=[],
        output_names=["out"],
        source=source,
    )
    
    # Configure kernel execution - use minimal configuration
    grid = (1, 1, 1)
    threadgroup = (1, 1, 1)
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(4,)]
    output_dtypes = [mx.float32]
    
    # Call the kernel
    print("Calling minimal working kernel...")
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
    
    # Get output
    out = outputs[0]
    
    # Print output
    print(f"Output: {out}")
    
    # Check if output contains expected values
    success = out[0] == 1.0
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if success else 'FAILURE'}")
    
    return success

if __name__ == "__main__":
    minimal_working_kernel()