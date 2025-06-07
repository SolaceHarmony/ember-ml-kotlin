"""
Test script to explore Metal kernel thread and grid limits in MLX.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

# Simple Metal kernel that writes thread ID and debug info
_THREAD_GRID_KERNEL_SRC = r"""
    // Get thread ID and dimensions
    uint tid = thread_position_in_grid.x;
    uint3 tpg = threads_per_threadgroup;
    uint3 gpg = grid_size;
    
    // Calculate total threads
    uint total_threads = gpg.x * tpg.x;
    
    // Write thread ID to output buffer
    if (tid < output_size[0]) {
        output[tid] = float(tid);
    }
    
    // Set debug values - only thread 0 should set these
    if (tid == 0) {
        dbg[0] = 1.0f;  // Kernel executed flag
        dbg[1] = float(tpg.x);  // Threads per threadgroup x
        dbg[2] = float(gpg.x);  // Grid size x
        dbg[3] = float(total_threads); // Total threads
    }
"""

# Compile the kernel
_THREAD_GRID_KERNEL = mx.fast.metal_kernel(
    name="thread_grid_test",
    input_names=["output_size"],
    output_names=["output", "dbg"],
    source=_THREAD_GRID_KERNEL_SRC,
)

def test_thread_grid_config(grid_x: int, threadgroup_x: int):
    """Tests a specific grid and threadgroup configuration."""
    print(f"\nTesting Grid: ({grid_x}, 1, 1), Threadgroup: ({threadgroup_x}, 1, 1)")
    
    grid = (grid_x, 1, 1)
    threadgroup = (threadgroup_x, 1, 1)
    
    total_threads = grid_x * threadgroup_x
    output_size = total_threads # Output size matches total threads
    
    # Prepare inputs
    output_size_arr = mx.array([output_size], dtype=mx.uint32)
    
    # Prepare outputs
    output_shapes = [(output_size,), (4,)]
    output_dtypes = [mx.float32, mx.float32]
    
    try:
        # Call the kernel
        start_time = time.time()
        outputs = _THREAD_GRID_KERNEL(
            inputs=[output_size_arr],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            # verbose=True # Uncomment to see generated code
        )
        end_time = time.time()
        print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
        
        # Get outputs
        output, dbg = outputs
        
        # Check if kernel executed
        kernel_executed = dbg[0].item() == 1.0
        print(f"Kernel executed: {kernel_executed}")
        
        if kernel_executed:
            # Verify debug info
            observed_tpg_x = dbg[1].item()
            observed_gpg_x = dbg[2].item()
            observed_total_threads = dbg[3].item()
            
            print(f"Observed Threadgroup.x: {observed_tpg_x}")
            print(f"Observed Grid.x: {observed_gpg_x}")
            print(f"Observed Total Threads: {observed_total_threads}")
            
            debug_info_match = (observed_tpg_x == threadgroup_x and
                                observed_gpg_x == grid_x and
                                observed_total_threads == total_threads)
            print(f"Debug info matches config: {debug_info_match}")
            
            # Verify output content (thread IDs)
            expected_output = mx.arange(total_threads, dtype=mx.float32)
            output_matches = mx.allclose(output, expected_output, atol=1e-6).item()
            print(f"Output matches expected thread IDs: {output_matches}")
            
            if not output_matches:
                 # Print first few elements if mismatch
                 print("Output mismatch. First few elements:")
                 print(f"  Expected: {expected_output[:min(10, total_threads)]}")
                 print(f"  Actual:   {output[:min(10, total_threads)]}")

        return kernel_executed # Return True if kernel executed
        
    except Exception as e:
        print(f"Kernel execution FAILED: {e}")
        return False # Return False if kernel failed

if __name__ == "__main__":
    # Set environment variables for Metal debugging (optional)
    # os.environ["MTL_DEBUG_LAYER"] = "1"
    # os.environ["MTL_SHADER_VALIDATION"] = "1"
    
    # Define configurations to test (grid_x, threadgroup_x)
    # Start with small, safe values and gradually increase
    configs_to_test = [
        (1, 1),      # Minimal
        (1, 32),     # Single threadgroup, SIMD size
        (1, 256),    # Single threadgroup, common max size
        (2, 32),     # Multiple threadgroups
        (10, 32),    # More threadgroups
        (32, 32),    # Equal grid and threadgroup size
        (100, 32),   # Larger grid
        (1024, 32),  # Max grid_x with small threadgroup
        (32, 256),   # Larger threadgroup
        (100, 256),  # Larger grid and threadgroup
        (1024, 256), # Max grid_x and larger threadgroup
        # Add larger values cautiously if previous tests pass
        # (1024, 512),
        # (1024, 1024), # Max grid_x and threadgroup_x
        # (2048, 32),
    ]
    
    print("=== Exploring Thread and Grid Limits ===")
    
    for grid_x, threadgroup_x in configs_to_test:
        test_thread_grid_config(grid_x, threadgroup_x)
        # Add a small delay between tests to avoid overwhelming the GPU
        time.sleep(0.1)
        
    print("\n=== Exploration Complete ===")