import mlx.core as mx

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "dimensions"],
        output_names=["out"],
        ensure_row_contiguous=True
    )

# Properly structured kernel with correct use of dimensions and thread positions
PROPER_KERNEL_SRC = """
    // Get dimensions from input buffer
    const uint grid_width = dimensions[0];
    const uint grid_height = dimensions[1];
    const uint out_size = dimensions[2];
    
    // Get thread position
    const uint tid_x = thread_position_in_grid.x;
    const uint tid_y = thread_position_in_grid.y;
    
    // Calculate 1D index using proper grid width from buffer
    const uint out_idx = tid_y * grid_width + tid_x;
    
    // Only write if within bounds
    if (out_idx < out_size) {
        // Convert index to float and store as uint32 bit pattern
        float val = float(out_idx + 1);
        out[out_idx] = as_type<uint>(val);
        
        // Special marker from thread (0,0) - avoid redundant write
        if (tid_x == 0 && tid_y == 0) {
            // Use atomic operation to make sure it's written
            out[0] = 0xCAFEBABE;
        }
    }
"""

# Compile kernel
properK = _compile(PROPER_KERNEL_SRC, "proper_kernel")

def test_proper_kernel():
    print("\n=== Testing Properly Structured Metal Kernel ===")
    
    # Create input data
    A = mx.ones((1,), dtype=mx.float32)  # Just a dummy input
    
    # Set parameters for grid dimensions
    grid_width = 4
    grid_height = 4
    out_size = grid_width * grid_height
    dimensions = mx.array([grid_width, grid_height, out_size], dtype=mx.uint32)
    
    # Create output buffer
    out = mx.zeros((out_size,), dtype=mx.uint32)
    
    # Print input parameters
    print(f"Grid dimensions: {grid_width}x{grid_height}")
    print(f"Output buffer size: {out_size}")
    
    # Run the kernel
    grid = (grid_width, grid_height, 1)
    threadgroup = (4, 4, 1)  # Using 4x4 thread groups
    
    print(f"Launching kernel with grid={grid}, threadgroup={threadgroup}")
    
    properK(inputs=[A, dimensions],
            output_shapes=[out.shape],
            output_dtypes=[mx.uint32],
            grid=grid, threadgroup=threadgroup, verbose=True)
    
    # Check output
    print("\nOutput buffer (first few elements):")
    print(out[:16])
    
    # Check for magic number
    if out[0] == 0xCAFEBABE:
        print("\nSUCCESS: Magic number found in output!")
    else:
        print(f"\nFAILURE: Magic number not found. First value: {hex(out[0].item() if hasattr(out[0], 'item') else out[0])}")
    
    # Check if other positions were written
    non_zero = (out != 0).sum()
    if non_zero > 0:
        print(f"SUCCESS: {non_zero} output positions were written to.")
    else:
        print("FAILURE: No output positions were written to.")
    
    # Display as a grid for visualization
    try:
        grid_view = out.reshape(grid_height, grid_width)
        print("\nOutput buffer as a grid:")
        print(grid_view)
        
        # Also display as floating-point values
        float_view = out.view(dtype=mx.float32).reshape(grid_height, grid_width)
        print("\nOutput as floating-point values:")
        print(float_view)
    except:
        print("Could not reshape to grid view.")

if __name__ == "__main__":
    test_proper_kernel()