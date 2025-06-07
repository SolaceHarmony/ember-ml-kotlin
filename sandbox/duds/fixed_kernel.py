import mlx.core as mx
import numpy as np

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "grid_dims"], # Added "grid_dims"
        output_names=["out"],
        ensure_row_contiguous=True
    )

# Fixed simple kernel with corrections
SIMPLE_SRC = """
    const device uint* grid_dims [[buffer(1)]];
    
    // Get thread IDs
    const uint tid_x = thread_position_in_grid.x;
    const uint tid_y = thread_position_in_grid.y;
    
    // Read grid dimensions from the input buffer
    const uint grid_width = grid_dims[0];  // Fixed: was missing [0]
    const uint grid_height = grid_dims[1];
    const uint out_size = grid_width * grid_height;
    
    // Calculate 1D index using proper grid width from buffer
    const uint out_idx = tid_y * grid_width + tid_x;
    
    // Only write if within bounds
    if (out_idx < out_size) {
        // Convert index to float and store as uint32 bit pattern
        float val = float(out_idx + 1);
        out[out_idx] = as_type<uint>(val);
        
        // Special marker from thread (0,0)
        if (tid_x == 0 && tid_y == 0) {
            out[0] = 0xF00D0001;  // Fixed: was missing [0]
        }
    }
"""

# Compile the simple kernel
simpleK = _compile(SIMPLE_SRC, "minimal_kernel_test")

def test_minimal_kernel():
    print("\n=== Testing Fixed Minimal Kernel ===")
    
    # Create dummy input
    A = mx.ones((4, 4), dtype=mx.float32)
    
    # Create output buffer
    out = mx.zeros((16,), dtype=mx.uint32)
    
    # Define grid and threadgroup sizes for dispatch
    grid = (4, 4, 1)
    threadgroup = (4, 4, 1)
    
    # Create MLX array to pass grid dimensions to the kernel
    grid_dims = mx.array(grid, dtype=mx.uint32)
    
    print(f"Launching kernel with grid={grid}, threadgroup={threadgroup}")
    print(f"Passing grid dimensions array: {grid_dims.tolist()}")
    
    # Run the kernel
    simpleK(inputs=[A, grid_dims],
            output_shapes=[out.shape],
            output_dtypes=[mx.uint32],
            grid=grid,
            threadgroup=threadgroup,
            verbose=True)
    
    # Display raw output buffer
    print("\nRaw output buffer (uint32):")
    print(out)
    
    # Check for magic number
    if out[0] == 0xF00D0001:
        print(f"\nSUCCESS: Magic number 0xF00D0001 found at position 0")
    else:
        print(f"\nFAILURE: Magic number not found. First value: {hex(out[0].item())}")
    
    # Display as grid for visualization
    try:
        grid_view = out.reshape(4, 4)
        print("\nOutput buffer as grid:")
        print(grid_view)
        
        # Convert to float32 to see the actual values
        float_view = out.view(dtype=mx.float32).reshape(4, 4)
        print("\nOutput as float values:")
        print(float_view)
    except:
        print("Could not reshape to grid view.")

if __name__ == "__main__":
    test_minimal_kernel()