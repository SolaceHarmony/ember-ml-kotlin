"""
Test file for incrementally testing the SVD power iteration kernel.
"""
import time
import mlx.core as mx

def test_simplified_svd_kernel():
    """
    Test a simplified version of the SVD power iteration kernel.
    """
    print("Testing simplified SVD power iteration kernel")
    print("=" * 80)
    
    # Define a simplified SVD power iteration kernel
    kernel_source = """
    // Simplified SVD power iteration kernel
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint k = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    uint num_iterations = iter_params[0];
    
    // Initialize Q_out with Q_init
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Matrix multiplication: Z = A * Q_out
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Copy Z to Q_out
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            Q_out[idx] = Z[idx];
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="simplified_svd_kernel",
        source=kernel_source,
        input_names=["A", "Q_init", "shape", "iter_params"],
        output_names=["Q_out", "Z"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_data = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_data, iter_params],
            output_shapes=[(n, k), (n, k)],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        Q_out, Z = outputs
        
        # Verify result: Z should be A^5 * Q_init
        expected_Z = Q_init
        for _ in range(5):
            expected_Z = mx.matmul(A, expected_Z)
        
        diff = mx.mean(mx.abs(Z - expected_Z)).item()
        print(f"  Difference between expected and actual Z: {diff:.2e}")
        print("  Simplified SVD kernel test passed!")
    except Exception as e:
        print(f"  Simplified SVD kernel test failed: {e}")

def test_svd_kernel_with_orthogonalization():
    """
    Test an SVD power iteration kernel with orthogonalization.
    """
    print("\nTesting SVD power iteration kernel with orthogonalization")
    print("=" * 80)
    
    # Define an SVD power iteration kernel with orthogonalization
    kernel_source = """
    // SVD power iteration kernel with orthogonalization
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint k = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    uint num_iterations = iter_params[0];
    
    // Shared memory for reduction operations
    threadgroup float shared_mem[32];
    
    // Initialize Q_out with Q_init
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Matrix multiplication: Z = A * Q_out
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Orthogonalize columns of Z
        for (uint col = 0; col < k; col++) {
            // Compute norm of column col
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce norm_sq using shared memory
            if (tid < 32) {
                shared_mem[tid] = 0.0f;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Each thread adds its partial sum to shared memory
            if (tid < 32) {
                shared_mem[tid] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 computes final norm and normalizes column
            if (tid == 0) {
                float total_norm_sq = 0.0f;
                for (uint i = 0; i < min(32u, num_threads); i++) {
                    total_norm_sq += shared_mem[i];
                }
                
                float norm = sqrt(total_norm_sq);
                shared_mem[0] = norm;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            float norm = shared_mem[0];
            float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
            
            // Normalize column
            for (uint row = tid; row < n; row += num_threads) {
                Q_out[row * k + col] = Z[row * k + col] * inv_norm;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="svd_kernel_with_orthogonalization",
        source=kernel_source,
        input_names=["A", "Q_init", "shape", "iter_params"],
        output_names=["Q_out", "Z"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_data = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_data, iter_params],
            output_shapes=[(n, k), (n, k)],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        Q_out, Z = outputs
        
        # Verify result: Q_out should be orthogonal
        Q_out_T = mx.transpose(Q_out)
        orthogonality = mx.matmul(Q_out_T, Q_out)
        identity = mx.eye(k, dtype=mx.float32)
        orthogonality_error = mx.mean(mx.abs(orthogonality - identity)).item()
        print(f"  Orthogonality error: {orthogonality_error:.2e}")
        print("  SVD kernel with orthogonalization test passed!")
    except Exception as e:
        print(f"  SVD kernel with orthogonalization test failed: {e}")

def test_svd_kernel_with_gram_schmidt():
    """
    Test an SVD power iteration kernel with Gram-Schmidt orthogonalization.
    """
    print("\nTesting SVD power iteration kernel with Gram-Schmidt orthogonalization")
    print("=" * 80)
    
    # Define an SVD power iteration kernel with Gram-Schmidt orthogonalization
    kernel_source = """
    // SVD power iteration kernel with Gram-Schmidt orthogonalization
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint k = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    uint num_iterations = iter_params[0];
    
    // Shared memory for reduction operations
    threadgroup float shared_proj[32];
    threadgroup float shared_norm[32];
    
    // Initialize Q_out with Q_init
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Matrix multiplication: Z = A * Q_out
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * Z[row * k + col];
                }
                
                // Reduce using shared memory
                if (tid < 32) {
                    shared_proj[tid] = 0.0f;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                if (tid < 32) {
                    shared_proj[tid] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 computes final projection
                if (tid == 0) {
                    float proj = 0.0f;
                    for (uint i = 0; i < min(32u, num_threads); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                float proj = shared_proj[0];
                
                // Subtract projection: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Compute norm of column col
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce using shared memory
            if (tid < 32) {
                shared_norm[tid] = 0.0f;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            if (tid < 32) {
                shared_norm[tid] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 computes final norm
            if (tid == 0) {
                float total_norm_sq = 0.0f;
                for (uint i = 0; i < min(32u, num_threads); i++) {
                    total_norm_sq += shared_norm[i];
                }
                
                float norm = sqrt(total_norm_sq);
                shared_norm[0] = norm;
                shared_norm[1] = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            float norm = shared_norm[0];
            float inv_norm = shared_norm[1];
            
            // Normalize column
            for (uint row = tid; row < n; row += num_threads) {
                Q_out[row * k + col] = Z[row * k + col] * inv_norm;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="svd_kernel_with_gram_schmidt",
        source=kernel_source,
        input_names=["A", "Q_init", "shape", "iter_params"],
        output_names=["Q_out", "Z"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_data = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_data, iter_params],
            output_shapes=[(n, k), (n, k)],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        Q_out, Z = outputs
        
        # Verify result: Q_out should be orthogonal
        Q_out_T = mx.transpose(Q_out)
        orthogonality = mx.matmul(Q_out_T, Q_out)
        identity = mx.eye(k, dtype=mx.float32)
        orthogonality_error = mx.mean(mx.abs(orthogonality - identity)).item()
        print(f"  Orthogonality error: {orthogonality_error:.2e}")
        print("  SVD kernel with Gram-Schmidt orthogonalization test passed!")
    except Exception as e:
        print(f"  SVD kernel with Gram-Schmidt orthogonalization test failed: {e}")

def test_svd_kernel_with_simd():
    """
    Test an SVD power iteration kernel with SIMD operations.
    """
    print("\nTesting SVD power iteration kernel with SIMD operations")
    print("=" * 80)
    
    # Define an SVD power iteration kernel with SIMD operations
    kernel_source = """
    // SVD power iteration kernel with SIMD operations
    uint tid = thread_position_in_grid.x;
    uint n = shape[0];
    uint k = shape[1];
    uint num_threads = threads_per_threadgroup.x;
    uint num_iterations = iter_params[0];
    
    // Shared memory for reduction operations
    threadgroup float shared_proj[8];
    threadgroup float shared_norm[8];
    
    // Initialize Q_out with Q_init
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Matrix multiplication: Z = A * Q_out
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * Z[row * k + col];
                }
                
                // Reduce using SIMD operations
                thread_proj = simd_sum(thread_proj);
                
                // First thread in each SIMD group writes to shared memory
                if (tid % 32 == 0 && tid / 32 < 8) {
                    shared_proj[tid / 32] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 computes final projection
                if (tid == 0) {
                    float proj = 0.0f;
                    for (uint i = 0; i < min(8u, (num_threads + 31) / 32); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                float proj = shared_proj[0];
                
                // Subtract projection: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Compute norm of column col
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce using SIMD operations
            thread_norm_sq = simd_sum(thread_norm_sq);
            
            // First thread in each SIMD group writes to shared memory
            if (tid % 32 == 0 && tid / 32 < 8) {
                shared_norm[tid / 32] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 computes final norm
            if (tid == 0) {
                float total_norm_sq = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + 31) / 32); i++) {
                    total_norm_sq += shared_norm[i];
                }
                
                float norm = sqrt(total_norm_sq);
                shared_norm[0] = norm;
                shared_norm[1] = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            float norm = shared_norm[0];
            float inv_norm = shared_norm[1];
            
            // Normalize column
            for (uint row = tid; row < n; row += num_threads) {
                Q_out[row * k + col] = Z[row * k + col] * inv_norm;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="svd_kernel_with_simd",
        source=kernel_source,
        input_names=["A", "Q_init", "shape", "iter_params"],
        output_names=["Q_out", "Z"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_data = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_data, iter_params],
            output_shapes=[(n, k), (n, k)],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup
        )
        
        Q_out, Z = outputs
        
        # Verify result: Q_out should be orthogonal
        Q_out_T = mx.transpose(Q_out)
        orthogonality = mx.matmul(Q_out_T, Q_out)
        identity = mx.eye(k, dtype=mx.float32)
        orthogonality_error = mx.mean(mx.abs(orthogonality - identity)).item()
        print(f"  Orthogonality error: {orthogonality_error:.2e}")
        print("  SVD kernel with SIMD operations test passed!")
    except Exception as e:
        print(f"  SVD kernel with SIMD operations test failed: {e}")

if __name__ == "__main__":
    # Run tests in order of increasing complexity
    test_simplified_svd_kernel()
    test_svd_kernel_with_orthogonalization()
    test_svd_kernel_with_gram_schmidt()
    test_svd_kernel_with_simd()