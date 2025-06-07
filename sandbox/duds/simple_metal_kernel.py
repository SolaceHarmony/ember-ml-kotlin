"""
Simple Metal kernel test that copies input to output.
"""

import sys
import os
import time
import mlx.core as mx

# Simple Metal kernel that copies input to output
_SIMPLE_KERNEL_SRC = r"""
    // Thread ID and dimensions - using Metal built-in variables
    uint tid = thread_position_in_grid.x;
    
    // Get matrix dimensions
    const uint size = shape[0];
    
    // Set debug values - only thread 0 should set these
    if (tid == 0) {
        debug[0] = 1.0f;  // Kernel executed
        debug[1] = float(size);  // Size of input/output
        debug[2] = float(threads_per_threadgroup.x);  // Threads per threadgroup
        debug[3] = float(grid_size.x);  // Grid size
    }
    
    // Copy input to output - only process if within bounds
    if (tid < size) {
        output[tid] = input[tid];
    }
"""

# Compile the kernel
_SIMPLE_KERNEL = mx.fast.metal_kernel(
    name="simple_kernel",
    source=_SIMPLE_KERNEL_SRC,
    input_names=["input", "shape"],
    output_names=["output", "debug"],
    ensure_row_contiguous=True
)

def test_simple_kernel():
    """Test a simple Metal kernel that copies input to output."""
    print("\n=== Simple Metal Kernel Test ===\n")
    
    # Create input data
    size = 10
    input_data = mx.arange(size, dtype=mx.float32)
    shape = mx.array([size], dtype=mx.uint32)
    
    print(f"Input data: {input_data}")
    print(f"Shape: {shape}")
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(size,), (4,)]
    output_dtypes = [mx.float32, mx.float32]
    
    # Call the kernel
    print("Calling Metal kernel...")
    start_time = time.time()
    outputs = _SIMPLE_KERNEL(
        inputs=[input_data, shape],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup
    )
    end_time = time.time()
    print(f"Metal kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get outputs
    output, debug = outputs
    
    # Print outputs
    print(f"Output data: {output}")
    print(f"Debug info: {debug}")
    
    # Check if debug info contains non-zero values
    debug_nonzero = mx.any(mx.abs(debug) > 0).item()
    print(f"Debug info contains non-zero values: {debug_nonzero}")
    
    if debug_nonzero:
        for i in range(debug.shape[0]):
            if abs(debug[i]) > 0:
                print(f"  debug[{i}] = {debug[i]}")
    
    # Check if output matches input
    match = mx.all(mx.abs(output - input_data) < 1e-6).item()
    print(f"Output matches input: {match}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if debug_nonzero else 'FAILURE'}")
    print(f"Input/output match: {'SUCCESS' if match else 'FAILURE'}")

if __name__ == "__main__":
    test_simple_kernel()