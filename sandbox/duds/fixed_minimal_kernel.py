import mlx.core as mx

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A"],
        output_names=["out"],
        ensure_row_contiguous=True
    )

# Even simpler kernel - just copies a constant value to the output buffer
ULTRA_SIMPLE_SRC = """
    // Just write a constant value to the first element - no fancy logic
    out[0] = 0xCAFEBABE;  // Easy to recognize magic number
"""

# Minimal kernel that uses thread positions properly
SIMPLE_SRC = """

    out[thread_position_in_grid.x,thread_position_in_grid.y,thread_position_in_grid.z] = as_type<uint>(float(tid_x + 1));  // Write thread ID to output

"""

# Compile both kernels
ultraSimpleK = _compile(ULTRA_SIMPLE_SRC, "ultra_simple_kernel")
simpleK = _compile(SIMPLE_SRC, "simple_kernel")

def test_kernels():
    print("\n=== Testing Ultra Simple Kernel ===")
    
    # Create dummy input and output
    A = mx.ones((1,), dtype=mx.float32)
    out = mx.zeros((16,1,1), dtype=mx.uint32)
    
    # Run the ultra simple kernel
    grid = (16,1,1 )
    threadgroup = (16,1,1)
    print(f"Launching ultra simple kernel with grid={grid}, threadgroup={threadgroup}")
    
    ultraSimpleK(inputs=[A],
                 output_shapes=[out.shape],
                 output_dtypes=[mx.uint32],
                 grid=grid, threadgroup=threadgroup, verbose=True)
    
    # Check results
    print("\nOutput buffer (first 4 elements):")
    print(out[:4])
    
    # Check for magic number
    if out[0] == 0xCAFEBABE:
        print("SUCCESS: Ultra simple kernel executed properly!")
    else:
        print(f"FAILURE: Expected 0xCAFEBABE, got {hex(out[0].item() if hasattr(out[0], 'item') else out[0])}")
    
    print("\n=== Testing Simple Kernel with Thread Positions ===")
    
    # Reset output buffer
    out = mx.zeros((16,), dtype=mx.uint32)
    
    # Run the simple kernel
    grid = (4, 4, 1)
    threadgroup = (4, 4, 1)
    print(f"Launching simple kernel with grid={grid}, threadgroup={threadgroup}")
    
    simpleK(inputs=[A],
            output_shapes=[out.shape],
            output_dtypes=[mx.uint32],
            grid=grid, threadgroup=threadgroup, verbose=True)
    
    # Display output as grid for visualization
    print("\nOutput buffer as grid:")
    try:
        grid_view = out.reshape(4, 4)
        print(grid_view)
    except:
        print("Could not reshape to grid view.")
        print("Raw output:", out)
    
    # Create a version with float interpretation
    float_view = out.view(dtype=mx.float32)
    print("\nOutput interpreted as floats:")
    try:
        float_grid = float_view.reshape(4, 4)
        print(float_grid)
    except:
        print("Could not reshape float view.")
        print("Raw float view:", float_view)
    
    # Check for expected patterns
    if out[0] == 0xCAFEBABE:
        print("\nSUCCESS: Magic number found in output!")
    else:
        print(f"\nFAILURE: Magic number not found. First value: {hex(out[0].item() if hasattr(out[0], 'item') else out[0])}")
    
    # Check if threads incremented output as expected
    if (out != 0).sum() > 0:
        print(f"SUCCESS: {(out != 0).sum()} output positions were written to.")
    else:
        print("FAILURE: No output positions were written to.")
    
    # Try one more thing - create output in a different way
    print("\n=== Testing Direct Creation of Output Buffer ===")
    out2 = mx.array([0] * 16, dtype=mx.uint32)
    
    ultraSimpleK(inputs=[A],
                 output_shapes=[out2.shape],
                 output_dtypes=[mx.uint32],
                 grid=(1, 1, 1), threadgroup=(1, 1, 1))
    
    print("Direct creation output:", out2[0])
    if out2[0] == 0xCAFEBABE:
        print("SUCCESS: Direct creation of output buffer works!")
    else:
        print("FAILURE: Direct creation of output buffer failed.")

if __name__ == "__main__":
    test_kernels()