"""
Test script to verify access to Metal kernel built-in variables in MLX.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

# Metal kernel that writes built-in variable values to output
_BUILTIN_VAR_KERNEL_SRC = r"""
    // Get thread ID and dimensions
    uint tid = thread_position_in_grid.x;
    uint3 tpg = threads_per_threadgroup;
    uint3 gpg = grid_size;
    
    // Write built-in variable values to output buffer - only thread 0
    if (tid == 0) {
        output[0] = 1.0f;  // Kernel executed flag
        output[1] = float(tid); // Thread ID (should be 0 for thread 0)
        output[2] = float(tpg.x);  // Threads per threadgroup x
        output[3] = float(gpg.x);  // Grid size x
        output[4] = float(tpg.y);  // Threads per threadgroup y
        output[5] = float(tpg.z);  // Threads per threadgroup z
        output[6] = float(gpg.y);  // Grid size y
        output[7] = float(gpg.z);  // Grid size z
    }
"""

# Compile the kernel
_BUILTIN_VAR_KERNEL = mx.fast.metal_kernel(
    name="builtin_var_test",
    input_names=[],
    output_names=["output"],
    source=_BUILTIN_VAR_KERNEL_SRC,
)

def test_builtin_variables(total_threads: int, threadgroup_x: int):
    """Tests if Metal built-in variables are accessible and correct."""
    print(f"\nTesting Total Threads: {total_threads}, Threadgroup.x: {threadgroup_x}")
    
    # In MLX, the 'grid' parameter specifies the total number of threads
    grid = (total_threads, 1, 1)
    # The 'threadgroup' parameter specifies the threads per threadgroup
    threadgroup = (threadgroup_x, 1, 1)
    
    # Prepare outputs - need space for debug values
    output_shapes = [(8,)] # 8 values for debug info
    output_dtypes = [mx.float32]
    
    try:
        # Call the kernel
        start_time = time.time()
        outputs = _BUILTIN_VAR_KERNEL(
            inputs=[],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            verbose=True # Print generated code
        )
        end_time = time.time()
        print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
        
        # Get output
        output = outputs[0]
        print(f"Output debug info: {output}")
        
        # Verify kernel executed
        kernel_executed = output[0].item() == 1.0
        print(f"Kernel executed flag: {kernel_executed}")
        
        if kernel_executed:
            # Verify built-in variable values
            observed_tid = output[1].item()
            observed_tpg_x = output[2].item()
            observed_gpg_x = output[3].item()
            observed_tpg_y = output[4].item()
            observed_tpg_z = output[5].item()
            observed_gpg_y = output[6].item()
            observed_gpg_z = output[7].item()
            
            print(f"Observed Thread ID (tid=0): {observed_tid}")
            print(f"Observed Threadgroup.x: {observed_tpg_x}")
            print(f"Observed Grid.x: {observed_gpg_x}")
            print(f"Observed Threadgroup.y: {observed_tpg_y}")
            print(f"Observed Threadgroup.z: {observed_tpg_z}")
            print(f"Observed Grid.y: {observed_gpg_y}")
            print(f"Observed Grid.z: {observed_gpg_z}")
            
            # Check if observed values match configured values (for x dimension)
            # Note: grid_size.x in the kernel should be total_threads / threads_per_threadgroup.x
            expected_grid_x_in_kernel = total_threads // threadgroup_x if threadgroup_x > 0 else 0
            
            match_x = (observed_tpg_x == threadgroup_x and observed_gpg_x == expected_grid_x_in_kernel)
            print(f"Observed x-dimensions match config: {match_x}")
            
            # Check if y and z dimensions are 1 (as configured)
            match_yz = (observed_tpg_y == 1.0 and observed_tpg_z == 1.0 and
                        observed_gpg_y == 1.0 and observed_gpg_z == 1.0)
            print(f"Observed y/z-dimensions are 1: {match_yz}")
            
            return kernel_executed and match_x and match_yz # Return True if successful
        
        return False # Return False if kernel didn't execute
        
    except Exception as e:
        print(f"Kernel execution FAILED: {e}")
        return False # Return False if kernel failed

if __name__ == "__main__":
    # Set environment variables for Metal debugging (optional)
    # os.environ["MTL_DEBUG_LAYER"] = "1"
    # os.environ["MTL_SHADER_VALIDATION"] = "1"
    
    # Define configurations to test (total_threads, threadgroup_x)
    # total_threads must be a multiple of threadgroup_x for simplicity in this test
    configs_to_test = [
        (1, 1),      # Minimal
        (32, 1),     # Total threads = SIMD size, threadgroup = 1
        (32, 32),    # Total threads = SIMD size, threadgroup = SIMD size
        (256, 32),   # Total threads = common max threadgroup, threadgroup = SIMD size
        (256, 256),  # Total threads = common max threadgroup, threadgroup = common max threadgroup
        (1024, 32),  # Total threads = typical max threadgroup, threadgroup = SIMD size
        (1024, 256), # Total threads = typical max threadgroup, threadgroup = common max threadgroup
        (1024, 1024),# Total threads = typical max threadgroup, threadgroup = typical max threadgroup
        (2048, 32),  # Total threads > typical max threadgroup, threadgroup = SIMD size
        (2048, 256), # Total threads > typical max threadgroup, threadgroup = common max threadgroup
        (4096, 32),
        (4096, 256),
        (4096, 1024),
        (8192, 32),
        (8192, 256),
        (8192, 1024),
    ]
    
    print("=== Exploring Metal Built-in Variable Access with Correct Dispatch Model ===")
    
    for total_threads, threadgroup_x in configs_to_test:
        # Ensure total_threads is a multiple of threadgroup_x for this test
        if total_threads % threadgroup_x != 0:
            print(f"\nSkipping config ({total_threads}, {threadgroup_x}): total_threads must be a multiple of threadgroup_x")
            continue
            
        test_builtin_variables(total_threads, threadgroup_x)
        # Add a small delay between tests to avoid overwhelming the GPU
        time.sleep(0.1)
        
    print("\n=== Exploration Complete ===")