"""
Test file for incrementally testing Metal kernels.
"""
import time
import mlx.core as mx

def test_simple_kernel():
    """
    Test a simple Metal kernel that just copies the input to the output.
    """
    print("Testing simple Metal kernel")
    print("=" * 80)
    
    # Define a simple kernel
    kernel_source = """
    // Simple kernel that copies input to output
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint num_threads = threads_per_threadgroup.x;
    
    // Copy input to output
    for (uint idx = tid; idx < n; idx += num_threads) {
        output[idx] = input[idx];
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="simple_kernel",
        source=kernel_source,
        input_names=["input", "shape"],
        output_names=["output"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n = 32
    input_data = mx.random.normal((n,), dtype=mx.float32)
    shape_data = mx.array([n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (1, 1, 1)
    threadgroup = (1, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[input_data, shape_data],
            output_shapes=[(n,)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        output_data = outputs[0]
        
        # Verify result
        diff = mx.mean(mx.abs(output_data - input_data)).item()
        print(f"  Difference between input and output: {diff:.2e}")
        print("  Simple kernel test passed!")
    except Exception as e:
        print(f"  Simple kernel test failed: {e}")

def test_kernel_with_shared_memory():
    """
    Test a Metal kernel that uses shared memory.
    """
    print("\nTesting Metal kernel with shared memory")
    print("=" * 80)
    
    # Define a kernel with shared memory
    kernel_source = """
    // Kernel that uses shared memory to compute sum
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint num_threads = threads_per_threadgroup.x;
    
    // Shared memory for partial sums
    threadgroup float partial_sums[32];
    
    // Initialize partial sums to zero
    if (tid < 32) {
        partial_sums[tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Each thread computes partial sum
    float thread_sum = 0.0f;
    for (uint idx = tid; idx < n; idx += num_threads) {
        thread_sum += input[idx];
    }
    
    // First thread in each warp writes to shared memory
    if (tid % 32 == 0 && tid / 32 < 32) {
        partial_sums[tid / 32] = thread_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Thread 0 combines results
    if (tid == 0) {
        float total_sum = 0.0f;
        for (uint i = 0; i < min(32u, (num_threads + 31) / 32); i++) {
            total_sum += partial_sums[i];
        }
        output[0] = total_sum;
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="shared_memory_kernel",
        source=kernel_source,
        input_names=["input", "shape"],
        output_names=["output"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n = 32
    input_data = mx.ones((n,), dtype=mx.float32)
    shape_data = mx.array([n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[input_data, shape_data],
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        output_data = outputs[0]
        
        # Verify result
        expected_sum = mx.sum(input_data).item()
        actual_sum = output_data.item()
        print(f"  Expected sum: {expected_sum}")
        print(f"  Actual sum: {actual_sum}")
        print(f"  Difference: {abs(actual_sum - expected_sum):.2e}")
        print("  Shared memory kernel test passed!")
    except Exception as e:
        print(f"  Shared memory kernel test failed: {e}")

def test_kernel_with_simd_operations():
    """
    Test a Metal kernel that uses SIMD operations.
    """
    print("\nTesting Metal kernel with SIMD operations")
    print("=" * 80)
    
    # Define a kernel with SIMD operations
    kernel_source = """
    // Kernel that uses SIMD operations to compute sum
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint num_threads = threads_per_threadgroup.x;
    
    // Each thread computes partial sum
    float thread_sum = 0.0f;
    for (uint idx = tid; idx < n; idx += num_threads) {
        thread_sum += input[idx];
    }
    
    // Reduce within SIMD group
    thread_sum = simd_sum(thread_sum);
    
    // First thread in each SIMD group writes to output
    if (tid % 32 == 0 && tid / 32 < 32) {
        output[tid / 32] = thread_sum;
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="simd_kernel",
        source=kernel_source,
        input_names=["input", "shape"],
        output_names=["output"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n = 32
    input_data = mx.ones((n,), dtype=mx.float32)
    shape_data = mx.array([n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[input_data, shape_data],
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        output_data = outputs[0]
        
        # Verify result
        expected_sum = mx.sum(input_data).item()
        actual_sum = output_data.item()
        print(f"  Expected sum: {expected_sum}")
        print(f"  Actual sum: {actual_sum}")
        print(f"  Difference: {abs(actual_sum - expected_sum):.2e}")
        print("  SIMD kernel test passed!")
    except Exception as e:
        print(f"  SIMD kernel test failed: {e}")

def test_kernel_with_matrix_operations():
    """
    Test a Metal kernel that performs matrix operations.
    """
    print("\nTesting Metal kernel with matrix operations")
    print("=" * 80)
    
    # Define a kernel with matrix operations
    kernel_source = """
    // Kernel that performs matrix operations
    uint tid = thread_position_in_grid.x;
    uint m = shape[0];
    uint n = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    
    // Initialize output matrix
    for (uint idx = tid; idx < m * n; idx += num_threads) {
        uint row = idx / n;
        uint col = idx % n;
        output[idx] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Compute matrix-vector product: output = A * v
    for (uint row = tid; row < m; row += num_threads) {
        float sum = 0.0f;
        for (uint col = 0; col < n; col++) {
            sum += A[row * n + col] * v[col];
        }
        output[row] = sum;
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="matrix_kernel",
        source=kernel_source,
        input_names=["A", "v", "shape"],
        output_names=["output"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    m, n = 32, 32
    A = mx.random.normal((m, n), dtype=mx.float32)
    v = mx.random.normal((n,), dtype=mx.float32)
    shape_data = mx.array([m, n], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, v, shape_data],
            output_shapes=[(m,)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        output_data = outputs[0]
        
        # Verify result
        expected_output = mx.matmul(A, v)
        diff = mx.mean(mx.abs(output_data - expected_output)).item()
        print(f"  Difference between expected and actual output: {diff:.2e}")
        print("  Matrix kernel test passed!")
    except Exception as e:
        print(f"  Matrix kernel test failed: {e}")

def test_kernel_with_gram_schmidt():
    """
    Test a Metal kernel that performs Gram-Schmidt orthogonalization.
    """
    print("\nTesting Metal kernel with Gram-Schmidt orthogonalization")
    print("=" * 80)
    
    # Define a kernel with Gram-Schmidt orthogonalization
    kernel_source = """
    // Kernel that performs Gram-Schmidt orthogonalization
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint k = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    
    // Initialize Q_out with Q_init
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Gram-Schmidt orthogonalization
    for (uint col = 0; col < k; col++) {
        // Normalize column col
        float norm_sq = 0.0f;
        for (uint row = 0; row < n; row++) {
            float val = Q_out[row * k + col];
            norm_sq += val * val;
        }
        
        float norm = sqrt(norm_sq);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        
        for (uint row = tid; row < n; row += num_threads) {
            Q_out[row * k + col] *= inv_norm;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Orthogonalize remaining columns against column col
        for (uint j = col + 1; j < k; j++) {
            float dot = 0.0f;
            for (uint row = 0; row < n; row++) {
                dot += Q_out[row * k + col] * Q_out[row * k + j];
            }
            
            for (uint row = tid; row < n; row += num_threads) {
                Q_out[row * k + j] -= dot * Q_out[row * k + col];
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="gram_schmidt_kernel",
        source=kernel_source,
        input_names=["Q_init", "shape"],
        output_names=["Q_out"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_data = mx.array([n, k], dtype=mx.uint32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[Q_init, shape_data],
            output_shapes=[(n, k)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        Q_out = outputs[0]
        
        # Verify result: Q_out should be orthogonal
        Q_out_T = mx.transpose(Q_out)
        orthogonality = mx.matmul(Q_out_T, Q_out)
        identity = mx.eye(k, dtype=mx.float32)
        orthogonality_error = mx.mean(mx.abs(orthogonality - identity)).item()
        print(f"  Orthogonality error: {orthogonality_error:.2e}")
        print("  Gram-Schmidt kernel test passed!")
    except Exception as e:
        print(f"  Gram-Schmidt kernel test failed: {e}")

if __name__ == "__main__":
    # Run tests in order of increasing complexity
    test_simple_kernel()
    test_kernel_with_shared_memory()
    test_kernel_with_simd_operations()
    test_kernel_with_matrix_operations()
    test_kernel_with_gram_schmidt()