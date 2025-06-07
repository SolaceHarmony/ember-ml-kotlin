"""
Minimal Metal kernel test to verify basic kernel execution.
"""

import sys
import os
import time
import mlx.core as mx

# Minimal Metal kernel that just sets a single value
_MINIMAL_KERNEL_SRC = r"""
    // Set a single value in the output
    output[0] = 42.0f;
"""

# Compile the kernel
_MINIMAL_KERNEL = mx.fast.metal_kernel(
    name="minimal_kernel",
    source=_MINIMAL_KERNEL_SRC,
    input_names=[],
    output_names=["output"],
    ensure_row_contiguous=True
)

def test_minimal_kernel():
    """Test a minimal Metal kernel to verify basic kernel execution."""
    print("\n=== Minimal Metal Kernel Test ===\n")
    
    # Configure kernel execution
    grid = (1, 1, 1)
    threadgroup = (1, 1, 1)
    print(f"Grid size: {grid}")
    print(f"Threadgroup size: {threadgroup}")
    
    # Prepare outputs
    output_shapes = [(1,)]
    output_dtypes = [mx.float32]
    
    # Call the kernel
    print("Calling Metal kernel...")
    start_time = time.time()
    outputs = _MINIMAL_KERNEL(
        inputs=[],
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup
    )
    end_time = time.time()
    print(f"Metal kernel execution completed in {end_time - start_time:.4f} seconds")
    
    # Get output
    output = outputs[0]
    
    # Print output
    print(f"Output: {output}")
    
    # Check if output contains the expected value
    expected_value = 42.0
    success = abs(output.item() - expected_value) < 1e-6
    
    # Print conclusion
    print("\n=== Test Results ===")
    print(f"Kernel execution: {'SUCCESS' if success else 'FAILURE'}")
    print(f"Expected value: {expected_value}")
    print(f"Actual value: {output.item()}")

if __name__ == "__main__":
    test_minimal_kernel()