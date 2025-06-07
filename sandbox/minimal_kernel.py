import mlx.core as mx
import numpy as np

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A"], # Added "grid_dims"
        output_names=["out"],
        ensure_row_contiguous=True # Ensures A is row-contiguous, simplifies indexing if used [1]
    )

# Extremely simple kernel that just copies values
SIMPLE_SRC = """
    const uint3 threadgroup_size = threads_per_threadgroup;
    const uint3 tid = thread_position_in_threadgroup;
    const uint3 gid = thread_position_in_grid;
    const uint3 grid_size = threadgroup_size * grid_dimensions;
    const uint index = gid.x + gid.y * grid_size.x + gid.z * grid_size.x * grid_size.y;
    out[index] = gid.x + 1;  // Write thread ID to output
"""

# Compile the simple kernel
simpleK = _compile(SIMPLE_SRC, "minimal_kernel_test")

def test_minimal_kernel():
    print("\n=== Testing Minimal Kernel ===")

    # Create dummy input (used by MLX to determine kernel input buffer type/size,
    # though not explicitly used in this specific kernel logic beyond providing context)
    A = mx.ones((4, 4), dtype=mx.uint32) # MLX adds 'const device float*' buffer arg for 'A' [1]

    # Create output buffer
    out = mx.zeros((16,), dtype=mx.uint32) # MLX adds 'device uint*' buffer arg for 'out' [1]

    # Define grid and threadgroup sizes for dispatch
    grid = (4, 4, 1)
    threadgroup = (4, 4, 1)

    # Create MLX array to pass grid dimensions to the kernel
    # grid is width, grid[1] is height for a 2D grid dispatch
    grid_dims = mx.array(grid, dtype=mx.uint32) # MLX adds 'const device uint*' buffer arg for 'grid_dims' [1]

    print(f"Launching kernel with grid={grid}, threadgroup={threadgroup}")
    print(f"Passing grid dimensions array: {grid_dims.tolist()}")
    

    # Run the kernel
    # Inputs must match the order in input_names=["A", "grid_dims"]
    simpleK(inputs=[A],
            output_shapes=[out.shape], # Shape of the output buffer [1]
            output_dtypes=[mx.uint32], # Dtype of the output buffer [1]
            grid=grid,
            threadgroup=threadgroup,
            verbose=True) # verbose=True prints the generated Metal source

    # Display raw output buffer
    print("\nRaw output buffer (uint32):")
    print(out)


if __name__ == "__main__":
    test_minimal_kernel()
    # Uncomment to run other tests
    # test_kernels()
    # test_proper_kernel()
    
