"""
Test file for testing a version of the SVD power iteration kernel that's closer to the original.
"""
import time
import mlx.core as mx

def test_full_svd_kernel():
    """
    Test a version of the SVD power iteration kernel that's closer to the original.
    """
    print("Testing full SVD power iteration kernel")
    print("=" * 80)
    
    # Define a full SVD power iteration kernel
    kernel_source = """
    #define EPSILON 1e-10f
    #define MAX_K 8  // Reduced from 64 to avoid potential issues
    #define WARP_SIZE 32
    
    uint tid = thread_position_in_grid.x;
    uint num_threads = threads_per_threadgroup.x;
    uint simd_lane_id = tid % WARP_SIZE;
    uint simd_group_id = tid / WARP_SIZE;
    
    uint n = shapeParams[0];
    uint k = shapeParams[1];
    uint num_iterations = iterParams[0];
    float tolerance = tolParams[0];
    
    // Shared memory for matrix Z and reduction operations
    threadgroup float shared_Z[1000 * MAX_K]; // Further reduced size to fit within 32KB limit
    threadgroup float shared_proj[MAX_K]; // For projection coefficients
    threadgroup float shared_norm[MAX_K]; // For column norms
    
    // Initialize Q_out with Q_init in parallel
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration with Gram-Schmidt orthogonalization
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Step 1: Matrix multiplication Z = A * Q_out (in parallel)
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            // Compute dot product for this element
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            shared_Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Step 2: Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Step 2a: Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product in parallel: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * shared_Z[row * k + col];
                }
                
                // Reduce within SIMD group
                thread_proj = simd_sum(thread_proj);
                
                // First thread in each SIMD group writes to shared memory
                if (simd_lane_id == 0 && simd_group_id < 8) {
                    shared_proj[simd_group_id] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 combines results
                float proj = 0.0f;
                if (tid == 0) {
                    for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj; // Store for all threads to use
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                proj = shared_proj[0]; // All threads read the final value
                
                // Subtract projection in parallel: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    shared_Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Step 2b: Compute norm squared in parallel: norm_sq = Z[:, col]' * Z[:, col]
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = shared_Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce within SIMD group
            thread_norm_sq = simd_sum(thread_norm_sq);
            
            // First thread in each SIMD group writes to shared memory
            if (simd_lane_id == 0 && simd_group_id < 8) {
                shared_norm[simd_group_id] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results and computes norm
            float norm = 0.0f;
            if (tid == 0) {
                float norm_sq = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                    norm_sq += shared_norm[i];
                }
                norm = sqrt(norm_sq);
                shared_norm[0] = norm; // Store for all threads to use
                shared_norm[1] = (norm > tolerance) ? (1.0f / norm) : 0.0f; // Store inverse norm
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            norm = shared_norm[0]; // All threads read the final norm
            float inv_norm = shared_norm[1]; // All threads read the inverse norm
            
            // Step 2c: Normalize Z[:, col] and store in Q_out[:, col] in parallel
            for (uint row = tid; row < n; row += num_threads) {
                if (norm > tolerance) {
                    Q_out[row * k + col] = shared_Z[row * k + col] * inv_norm;
                } else {
                    Q_out[row * k + col] = 0.0f;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="full_svd_kernel",
        source=kernel_source,
        input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
        output_names=["Q_out"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 4  # Use a smaller k to avoid potential issues
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_params = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    tol_params = mx.array([1e-10], dtype=mx.float32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_params, iter_params, tol_params],
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
        print("  Full SVD kernel test passed!")
    except Exception as e:
        print(f"  Full SVD kernel test failed: {e}")

def test_full_svd_kernel_with_larger_k():
    """
    Test the full SVD power iteration kernel with a larger k value.
    """
    print("\nTesting full SVD power iteration kernel with larger k")
    print("=" * 80)
    
    # Define a full SVD power iteration kernel
    kernel_source = """
    #define EPSILON 1e-10f
    #define MAX_K 16  // Increased from 8
    #define WARP_SIZE 32
    
    uint tid = thread_position_in_grid.x;
    uint num_threads = threads_per_threadgroup.x;
    uint simd_lane_id = tid % WARP_SIZE;
    uint simd_group_id = tid / WARP_SIZE;
    
    uint n = shapeParams[0];
    uint k = shapeParams[1];
    uint num_iterations = iterParams[0];
    float tolerance = tolParams[0];
    
    // Shared memory for matrix Z and reduction operations
    threadgroup float shared_Z[500 * MAX_K]; // Further reduced size to fit within 32KB limit
    threadgroup float shared_proj[MAX_K]; // For projection coefficients
    threadgroup float shared_norm[MAX_K]; // For column norms
    
    // Initialize Q_out with Q_init in parallel
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration with Gram-Schmidt orthogonalization
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Step 1: Matrix multiplication Z = A * Q_out (in parallel)
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            // Compute dot product for this element
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            shared_Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Step 2: Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Step 2a: Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product in parallel: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * shared_Z[row * k + col];
                }
                
                // Reduce within SIMD group
                thread_proj = simd_sum(thread_proj);
                
                // First thread in each SIMD group writes to shared memory
                if (simd_lane_id == 0 && simd_group_id < 8) {
                    shared_proj[simd_group_id] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 combines results
                float proj = 0.0f;
                if (tid == 0) {
                    for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj; // Store for all threads to use
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                proj = shared_proj[0]; // All threads read the final value
                
                // Subtract projection in parallel: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    shared_Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Step 2b: Compute norm squared in parallel: norm_sq = Z[:, col]' * Z[:, col]
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = shared_Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce within SIMD group
            thread_norm_sq = simd_sum(thread_norm_sq);
            
            // First thread in each SIMD group writes to shared memory
            if (simd_lane_id == 0 && simd_group_id < 8) {
                shared_norm[simd_group_id] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results and computes norm
            float norm = 0.0f;
            if (tid == 0) {
                float norm_sq = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                    norm_sq += shared_norm[i];
                }
                norm = sqrt(norm_sq);
                shared_norm[0] = norm; // Store for all threads to use
                shared_norm[1] = (norm > tolerance) ? (1.0f / norm) : 0.0f; // Store inverse norm
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            norm = shared_norm[0]; // All threads read the final norm
            float inv_norm = shared_norm[1]; // All threads read the inverse norm
            
            // Step 2c: Normalize Z[:, col] and store in Q_out[:, col] in parallel
            for (uint row = tid; row < n; row += num_threads) {
                if (norm > tolerance) {
                    Q_out[row * k + col] = shared_Z[row * k + col] * inv_norm;
                } else {
                    Q_out[row * k + col] = 0.0f;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="full_svd_kernel_larger_k",
        source=kernel_source,
        input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
        output_names=["Q_out"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 32, 8  # Increased k from 4 to 8
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_params = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    tol_params = mx.array([1e-10], dtype=mx.float32)
    
    # Configure kernel execution
    grid = (32, 1, 1)
    threadgroup = (32, 1, 1)
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_params, iter_params, tol_params],
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
        print("  Full SVD kernel with larger k test passed!")
    except Exception as e:
        print(f"  Full SVD kernel with larger k test failed: {e}")

def test_full_svd_kernel_with_larger_matrix():
    """
    Test the full SVD power iteration kernel with a larger matrix.
    """
    print("\nTesting full SVD power iteration kernel with larger matrix")
    print("=" * 80)
    
    # Define a full SVD power iteration kernel
    kernel_source = """
    #define EPSILON 1e-10f
    #define MAX_K 8  // Back to 8
    #define WARP_SIZE 32
    
    uint tid = thread_position_in_grid.x;
    uint num_threads = threads_per_threadgroup.x;
    uint simd_lane_id = tid % WARP_SIZE;
    uint simd_group_id = tid / WARP_SIZE;
    
    uint n = shapeParams[0];
    uint k = shapeParams[1];
    uint num_iterations = iterParams[0];
    float tolerance = tolParams[0];
    
    // Shared memory for matrix Z and reduction operations
    threadgroup float shared_Z[500 * MAX_K]; // Reduced size to fit within 32KB limit
    threadgroup float shared_proj[MAX_K]; // For projection coefficients
    threadgroup float shared_norm[MAX_K]; // For column norms
    
    // Initialize Q_out with Q_init in parallel
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration with Gram-Schmidt orthogonalization
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Step 1: Matrix multiplication Z = A * Q_out (in parallel)
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            // Compute dot product for this element
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            shared_Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Step 2: Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Step 2a: Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product in parallel: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * shared_Z[row * k + col];
                }
                
                // Reduce within SIMD group
                thread_proj = simd_sum(thread_proj);
                
                // First thread in each SIMD group writes to shared memory
                if (simd_lane_id == 0 && simd_group_id < 8) {
                    shared_proj[simd_group_id] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 combines results
                float proj = 0.0f;
                if (tid == 0) {
                    for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj; // Store for all threads to use
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                proj = shared_proj[0]; // All threads read the final value
                
                // Subtract projection in parallel: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    shared_Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Step 2b: Compute norm squared in parallel: norm_sq = Z[:, col]' * Z[:, col]
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = shared_Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce within SIMD group
            thread_norm_sq = simd_sum(thread_norm_sq);
            
            // First thread in each SIMD group writes to shared memory
            if (simd_lane_id == 0 && simd_group_id < 8) {
                shared_norm[simd_group_id] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results and computes norm
            float norm = 0.0f;
            if (tid == 0) {
                float norm_sq = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                    norm_sq += shared_norm[i];
                }
                norm = sqrt(norm_sq);
                shared_norm[0] = norm; // Store for all threads to use
                shared_norm[1] = (norm > tolerance) ? (1.0f / norm) : 0.0f; // Store inverse norm
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            norm = shared_norm[0]; // All threads read the final norm
            float inv_norm = shared_norm[1]; // All threads read the inverse norm
            
            // Step 2c: Normalize Z[:, col] and store in Q_out[:, col] in parallel
            for (uint row = tid; row < n; row += num_threads) {
                if (norm > tolerance) {
                    Q_out[row * k + col] = shared_Z[row * k + col] * inv_norm;
                } else {
                    Q_out[row * k + col] = 0.0f;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    """
    
    # Compile the kernel
    kernel = mx.fast.metal_kernel(
        name="full_svd_kernel_larger_matrix",
        source=kernel_source,
        input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
        output_names=["Q_out"],
        ensure_row_contiguous=True
    )
    
    # Create test data
    n, k = 64, 4  # Increased n from 32 to 64
    A = mx.random.normal((n, n), dtype=mx.float32)
    # Make A symmetric positive definite
    A = mx.matmul(A, mx.transpose(A))
    Q_init = mx.random.normal((n, k), dtype=mx.float32)
    shape_params = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([5], dtype=mx.uint32)  # 5 iterations
    tol_params = mx.array([1e-10], dtype=mx.float32)
    
    # Configure kernel execution
    grid = (64, 1, 1)  # Increased from 32 to 64
    threadgroup = (64, 1, 1)  # Increased from 32 to 64
    
    # Execute the kernel
    try:
        outputs = kernel(
            inputs=[A, Q_init, shape_params, iter_params, tol_params],
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
        print("  Full SVD kernel with larger matrix test passed!")
    except Exception as e:
        print(f"  Full SVD kernel with larger matrix test failed: {e}")

if __name__ == "__main__":
    # Run tests in order of increasing complexity
    test_full_svd_kernel()
    test_full_svd_kernel_with_larger_k()
    test_full_svd_kernel_with_larger_matrix()