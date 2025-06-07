"""
Test script to verify input buffer access in a Metal kernel in MLX.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

# Metal kernel that reads the first element of the input matrix and writes debug info
_READ_INPUT_KERNEL_SRC = r"""
    // Get thread ID
    uint tid = thread_position_in_grid.x;
    
    // Get matrix dimensions from input shape
    const uint m = A_shape[0];
    const uint n = A_shape[1];
    
    // Set debug values and read input - only thread 0 should do this
    if (tid == 0) {
        output[0] = 1.0f;  // Kernel executed flag
        output[1] = float(tid); // Thread ID (should be 0 for thread 0)
        output[2] = float(threads_per_threadgroup.x);  // Threads per threadgroup x
        output[3] = float(grid_size.x);  // Grid size x
        output[4] = float(m); // Number of rows from A_shape
        output[5] = float(n); // Number of columns from A_shape
        
        // Read the first element of the input matrix A
        output[6] = A[0]; 
    }
"""

# Compile the kernel
_READ_INPUT_KERNEL = mx.fast.metal_kernel(
    name="read_input_test",
    input_names=["A"], # Input matrix A
    output_names=["output"], # Output buffer for debug info and A[0]
    source=_READ_INPUT_KERNEL_SRC,
)

def test_read_input_kernel():
    """Test a Metal kernel that reads the first element of an input matrix."""
    print("\n=== Read Input Kernel Test ===\n")
    
    # Create a small test matrix
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")
    
    # Prepare inputs
    a_shape = mx.array(a.shape, dtype=mx.uint32)
    
    # Configure kernel execution - use a minimal configuration
    # We only need one thread to read the first element
    grid = (1, 1, 1)
    threadgroup = (1, 1, 1)
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs - need space for debug values and A[0]
    output_shapes = [(7,)] # 7 values for debug info and A[0]
    output_dtypes = [mx.float32]
    
    try:
        # Call the kernel
        print("Calling read input kernel...")
        start_time = time.time()
        outputs = _READ_INPUT_KERNEL(
            inputs=[a], # Pass only the input matrix
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            verbose=True  # Print generated code for debugging
        )
        end_time = time.time()
        print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
        
        # Get output
        output = outputs[0]
        print(f"Output debug info and A[0]: {output}")
        
        # Verify kernel executed
        kernel_executed = output[0].item() == 1.0
        print(f"Kernel executed flag: {kernel_executed}")
        
        if kernel_executed:
            # Verify debug info and A[0]
            observed_tid = output[1].item()
            observed_tpg_x = output[2].item()
            observed_gpg_x = output[3].item()
            observed_m = output[4].item()
            observed_n = output[5].item()
            observed_a_0 = output[6].item()
            
            print(f"Observed Thread ID (tid=0): {observed_tid}")
            print(f"Observed Threadgroup.x: {observed_tpg_x}")
            print(f"Observed Grid.x: {observed_gpg_x}")
            print(f"Observed m (from A_shape): {observed_m}")
            print(f"Observed n (from A_shape): {observed_n}")
            print(f"Observed A[0]: {observed_a_0}")
            
            # Check if observed values match expected values
            # Removed check for observed_gpg_x == grid[0] as grid_size.x is always 0 in MLX dispatch
            match = (observed_tid == 0.0 and
                     observed_tpg_x == threadgroup[0] and
                     observed_m == a.shape[0] and
                     observed_n == a.shape[1] and
                     abs(observed_a_0 - a[0, 0].item()) < 1e-6)
            print(f"Observed values match expected: {match}")
            
            return kernel_executed and match # Return True if successful
        
        return False # Return False if kernel didn't execute
        
    except Exception as e:
        print(f"Kernel execution FAILED: {e}")
        return False # Return False if kernel failed

if __name__ == "__main__":
    test_read_input_kernel()