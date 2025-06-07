import mlx.core as mx

def test_kernel_communication():
    """Test basic buffer read/write in Metal"""
    src = """
    uint tid=thread_position_in_grid.x;
    // Iterate over the threadgroup
    out[tid] = in[tid] * 2.0f;

"""
    
    # Compile kernel
    kernel = mx.fast.metal_kernel(
        name="test_kernel",
        source=src,
        input_names=["in"],
        output_names=["out"],
        ensure_row_contiguous=True
    )
    
    # Test data
    data = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    out_shape = data.shape
    out_dtype = data.dtype
    
    # Execute kernel
    outputs = kernel(
        inputs=[data],
        output_shapes=[out_shape],
        output_dtypes=[out_dtype],
        grid=(data.size,1,1),
        threadgroup=(4,1,1)
    )
    out = outputs[0]
    
    print("Input:", data)
    print("Output:", out)
    print("Expected:", data * 2)
    print("Test", "PASSED" if mx.allclose(out, data*2) else "FAILED")

if __name__ == "__main__":
    test_kernel_communication()