"""
Simple Metal kernel example based on the official MLX documentation.
"""

import sys
import os
import time
import mlx.core as mx
import numpy as np

def simple_metal_kernel(a):
    """Simple Metal kernel that adds 1 to each element."""
    source = """
        uint elem = thread_position_in_grid.x;
        if (elem < inp_shape[0]) {
            out[elem] = inp[elem] + 1.0f;
        }
    """

    kernel = mx.fast.metal_kernel(
        name="simple_add_one",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )
    
    print(f"Input shape: {a.shape}")
    print(f"Input dtype: {a.dtype}")
    
    # Configure kernel execution
    grid = (a.size, 1, 1)
    threadgroup = (min(256, a.size), 1, 1)
    
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Call the kernel
    print("Calling Metal kernel...")
    start_time = time.time()
    outputs = kernel(
        inputs=[a],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        verbose=True  # Print generated code for debugging
    )
    end_time = time.time()
    print(f"Kernel execution completed in {end_time - start_time:.4f} seconds")
    
    return outputs[0]

def test_simple_metal_kernel():
    """Test the simple Metal kernel."""
    print("\n=== Simple Metal Kernel Test ===\n")
    
    # Create input data
    size = 10
    a = mx.arange(size, dtype=mx.float32)
    print(f"Input data: {a}")
    
    # Run the kernel
    b = simple_metal_kernel(a)
    print(f"Output data: {b}")
    
    # Verify the result
    expected = a + 1
    match = mx.all(mx.abs(b - expected) < 1e-6).item()
    print(f"Output matches expected: {match}")
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if match else 'FAILURE'}")

if __name__ == "__main__":
    test_simple_metal_kernel()