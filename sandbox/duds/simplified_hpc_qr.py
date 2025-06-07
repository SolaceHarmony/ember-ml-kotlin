"""
Simplified HPC QR decomposition using Modified Gram-Schmidt.

This implementation focuses on numerical stability with a simpler approach
to high-precision computing.
"""

import mlx.core as mx
import time
import numpy as np

def simplified_hpc_qr(A: mx.array) -> tuple[mx.array, mx.array]:
    """
    Simplified HPC QR decomposition using Modified Gram-Schmidt.
    
    This implementation uses a Metal kernel with a focus on numerical stability
    rather than complex limb-based precision handling.
    
    Args:
        A: Input matrix (M x N) as float32.
        
    Returns:
        Tuple of (Q, R) matrices as float32.
    """
    m, n = A.shape
    k = min(m, n)
    
    # Define Metal kernel source string for simplified HPC QR decomposition
    metal_kernel_source = """
/*****************************************************************************
 *  simplified_hpc_qr_kernel – QR decomposition with improved numerical stability
 *  
 *  This implementation uses Modified Gram-Schmidt with careful handling of
 *  numerical stability issues.
 *  
 *  outputs:  Q_out  (m×k),  R_out (k×n),  debug[0…15]
 *****************************************************************************/
#define EPSILON      1e-10f

/* ------------------------------------------------------------------------- */
{
    const uint m = shapeParams[0];
    const uint n = shapeParams[1];
    const uint k = shapeParams[2];
    
    /* Initialize debug array */
    if (thread_position_in_grid.x == 0) {
        for (uint i = 0; i < 16; ++i)
            debug[i] = 0.0f;
    }
    
    /* ============= 0. Initialize Q and R ==================================== */
    // Q is initialized to zeros (m×k)
    for (uint row = thread_position_in_grid.x;
         row < m;
         row += threads_per_threadgroup.x)
    {
        for (uint col = 0; col < k; ++col)
            Q_out[row*k + col] = 0.0f;
    }
    
    // R is initialized to zeros (k×n)
    for (uint row = thread_position_in_grid.x;
         row < k;
         row += threads_per_threadgroup.x)
    {
        for (uint col = 0; col < n; ++col)
            R_out[row*n + col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    /* ============= 1. Modified Gram-Schmidt ================================= */
    for (uint j = 0; j < k; ++j)
    {
        /* ---- 1a. Get column j of A ---------------------------------------- */
        // Each thread loads its assigned elements of column j
        for (uint row = thread_position_in_grid.x;
             row < m;
             row += threads_per_threadgroup.x)
        {
            // Store column j in Q_out[:, j] temporarily
            Q_out[row*k + j] = A[row*n + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        /* ---- 1b. Orthogonalize against previous columns ------------------- */
        for (uint i = 0; i < j; ++i)
        {
            // Compute dot product <Q[:, i], Q[:, j]>
            threadgroup float tg_dot;
            
            if (thread_position_in_threadgroup.x == 0) {
                tg_dot = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Each thread computes part of the dot product
            thread float local_dot = 0.0f;
            
            for (uint row = thread_position_in_grid.x;
                 row < m;
                 row += threads_per_threadgroup.x)
            {
                float qi = Q_out[row*k + i];  // Q[row, i]
                float qj = Q_out[row*k + j];  // Q[row, j]
                local_dot += qi * qj;
            }
            
            // Add local results to shared memory using atomic operations
            threadgroup atomic_float dot_atomic;
            if (thread_position_in_threadgroup.x == 0) {
                atomic_store_explicit(&dot_atomic, 0.0f, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            atomic_fetch_add_explicit(&dot_atomic, local_dot, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (thread_position_in_threadgroup.x == 0) {
                tg_dot = atomic_load_explicit(&dot_atomic, memory_order_relaxed);
                
                // Store in R[i, j]
                R_out[i*n + j] = tg_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Get dot product value
            const float dot = tg_dot;
            
            // Update Q[:, j] by subtracting projection
            for (uint row = thread_position_in_grid.x;
                 row < m;
                 row += threads_per_threadgroup.x)
            {
                float qi = Q_out[row*k + i];  // Q[row, i]
                float proj = qi * dot;
                Q_out[row*k + j] -= proj;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        /* ---- 1c. Compute column norm ------------------------------------- */
        threadgroup float tg_norm;
        
        if (thread_position_in_threadgroup.x == 0) {
            tg_norm = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Each thread computes part of the norm
        thread float local_norm_sq = 0.0f;
        
        for (uint row = thread_position_in_grid.x;
             row < m;
             row += threads_per_threadgroup.x)
        {
            float qj = Q_out[row*k + j];  // Q[row, j]
            local_norm_sq += qj * qj;
        }
        
        // Add local results to shared memory using atomic operations
        threadgroup atomic_float norm_atomic;
        if (thread_position_in_threadgroup.x == 0) {
            atomic_store_explicit(&norm_atomic, 0.0f, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        atomic_fetch_add_explicit(&norm_atomic, local_norm_sq, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (thread_position_in_threadgroup.x == 0) {
            float norm_sq = atomic_load_explicit(&norm_atomic, memory_order_relaxed);
            
            // Handle numerical stability
            if (norm_sq <= EPSILON) {
                norm_sq = EPSILON;
                debug[0] = 1.0f;  // Flag for zero norm
            }
            
            float norm = sqrt(norm_sq);
            tg_norm = norm;
            
            // Store in R[j, j]
            R_out[j*n + j] = norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        const float norm = tg_norm;
        
        /* ---- 1d. Normalize column j --------------------------------------- */
        for (uint row = thread_position_in_grid.x;
             row < m;
             row += threads_per_threadgroup.x)
        {
            float qj = Q_out[row*k + j];  // Q[row, j]
            Q_out[row*k + j] = qj / norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        /* ---- 1e. Compute remaining R entries ------------------------------ */
        if (j < n - 1) {
            for (uint col = j + 1; col < n; ++col) {
                // Compute dot product <Q[:, j], A[:, col]>
                threadgroup float tg_dot;
                
                if (thread_position_in_threadgroup.x == 0) {
                    tg_dot = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Each thread computes part of the dot product
                thread float local_dot = 0.0f;
                
                for (uint row = thread_position_in_grid.x;
                     row < m;
                     row += threads_per_threadgroup.x)
                {
                    float qj = Q_out[row*k + j];      // Q[row, j]
                    float a_col = A[row*n + col];     // A[row, col]
                    local_dot += qj * a_col;
                }
                
                // Add local results to shared memory using atomic operations
                threadgroup atomic_float dot_atomic;
                if (thread_position_in_threadgroup.x == 0) {
                    atomic_store_explicit(&dot_atomic, 0.0f, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                atomic_fetch_add_explicit(&dot_atomic, local_dot, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                if (thread_position_in_threadgroup.x == 0) {
                    float dot = atomic_load_explicit(&dot_atomic, memory_order_relaxed);
                    
                    // Store in R[j, col]
                    R_out[j*n + col] = dot;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
} /* end kernel body */
    """
    
    try:
        # Compile the Metal kernel
        compiled_kernel = mx.fast.metal_kernel(
            name="simplified_hpc_qr_kernel",
            source=metal_kernel_source,
            input_names=["A", "shapeParams"],
            output_names=["Q_out", "R_out", "debug"],
            ensure_row_contiguous=True
        )
    except Exception as e:
        print(f"Failed to compile Metal kernel: {e}")
        print("Please check the Metal kernel source for errors.")
        return mx.eye(m), mx.array(A)
    
    # Determine grid and threadgroup sizes based on matrix dimensions
    threads_per_group = min(256, max(32, m // 16))  # Larger thread groups for better occupancy
    num_groups = (m + threads_per_group - 1) // threads_per_group
    grid_size = (num_groups, 1, 1)
    tg_size = (threads_per_group, 1, 1)
    shape_params = mx.array([m, n, k], dtype=mx.uint32)
    
    dbg = mx.zeros((16,), dtype=mx.float32)
    
    try:
        # Execute the kernel
        Q, R, dbg = compiled_kernel(
            inputs=        [A, shape_params],
            output_shapes= [(m, k), (k, n), (16,)],
            output_dtypes= [A.dtype, A.dtype, mx.float32],
            grid=          grid_size,
            threadgroup=   tg_size
        )
        print("Debug flags:", dbg)
        
        # Convert to full Q matrix if needed (for compatibility with standard QR)
        if m > k:
            Q_full = mx.zeros((m, m), dtype=A.dtype)
            for i in range(m):
                for j in range(k):
                    Q_full = Q_full.at[i, j].set(Q[i, j])
                if i >= k:
                    Q_full = Q_full.at[i, i].set(1.0)
            return Q_full, R
        return Q, R
    except Exception as e:
        print(f"Metal kernel execution failed: {e}")
        return mx.eye(m), mx.array(A)

def benchmark_simplified_qr(sizes=[(100, 100), (500, 500), (1000, 1000)], 
                           repeat=3, 
                           compare_with_native=True):
    """
    Benchmark the simplified QR implementation against different matrix sizes
    and optionally compare with native MLX QR.
    
    Args:
        sizes: List of (m, n) tuples for matrix sizes to test
        repeat: Number of times to repeat each test for averaging
        compare_with_native: Whether to compare with native MLX QR
    """
    results = []
    
    for size in sizes:
        m, n = size
        print(f"\nTesting with matrix size {m}x{n}...")
        
        # Generate random matrix
        A = mx.random.normal((m, n), dtype=mx.float32)
        
        # Test simplified implementation
        simplified_times = []
        for i in range(repeat):
            start_time = time.time()
            Q_simplified, R_simplified = simplified_hpc_qr(A)
            end_time = time.time()
            simplified_times.append(end_time - start_time)
            
        avg_simplified_time = sum(simplified_times) / repeat
        print(f"Simplified QR completed in {avg_simplified_time:.4f} seconds (avg of {repeat} runs)")
        
        # Check orthogonality and reconstruction
        ortho_error = mx.mean(mx.abs(mx.matmul(Q_simplified.T, Q_simplified) - mx.eye(Q_simplified.shape[0]))).item()
        recon_error = mx.mean(mx.abs(mx.matmul(Q_simplified, R_simplified) - A)).item()
        print(f"Orthogonality error: {ortho_error:.6e}")
        print(f"Reconstruction error: {recon_error:.6e}")
        
        result = {
            "size": size,
            "simplified_time": avg_simplified_time,
            "ortho_error": ortho_error,
            "recon_error": recon_error
        }
        
        # Compare with native MLX QR if requested
        if compare_with_native:
            native_times = []
            for i in range(repeat):
                start_time = time.time()
                try:
                    Q_native, R_native = mx.linalg.qr(A, stream=mx.cpu)
                    end_time = time.time()
                    native_times.append(end_time - start_time)
                except Exception as e:
                    print(f"Native MLX QR failed: {e}")
                    native_times.append(float('nan'))
            
            if not all(np.isnan(native_times)):
                avg_native_time = sum(t for t in native_times if not np.isnan(t)) / sum(1 for t in native_times if not np.isnan(t))
                print(f"Native MLX QR completed in {avg_native_time:.4f} seconds (avg of {repeat} runs)")
                
                # Check differences if native QR succeeded
                try:
                    q_diff = mx.mean(mx.abs(Q_simplified - Q_native)).item()
                    r_diff = mx.mean(mx.abs(R_simplified - R_native)).item()
                    print(f"Difference in Q: {q_diff:.6e}")
                    print(f"Difference in R: {r_diff:.6e}")
                    
                    result["native_time"] = avg_native_time
                    result["q_diff"] = q_diff
                    result["r_diff"] = r_diff
                    result["speedup"] = avg_native_time / avg_simplified_time
                except Exception as e:
                    print(f"Could not compare with native QR: {e}")
        
        results.append(result)
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("Simplified HPC QR Decomposition for MLX")
    print("=" * 80)
    
    # Test with a smaller matrix first
    print("\nTesting with small matrix (10x10)...")
    A_small = mx.random.normal((10, 10), dtype=mx.float32)
    start_time = time.time()
    Q_small, R_small = simplified_hpc_qr(A_small)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_small.T, Q_small) - mx.eye(Q_small.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_small, R_small) - A_small)).item())
    print("-" * 40)
    
    # Run benchmarks with different matrix sizes
    print("\nRunning benchmarks...")
    results = benchmark_simplified_qr(sizes=[(50, 50), (100, 100), (200, 200)], repeat=2)
    
    # Print summary of benchmark results
    print("\nBenchmark Summary:")
    print("-" * 40)
    print(f"{'Size':>10} | {'Time (s)':>10} | {'Ortho Err':>10} | {'Recon Err':>10} | {'Speedup':>10}")
    print("-" * 60)
    for result in results:
        size_str = f"{result['size'][0]}x{result['size'][1]}"
        speedup = result.get('speedup', 'N/A')
        if speedup != 'N/A':
            speedup = f"{speedup:.2f}x"
        print(f"{size_str:>10} | {result['simplified_time']:>10.4f} | {result['ortho_error']:>10.2e} | {result['recon_error']:>10.2e} | {speedup:>10}")
    print("-" * 60)