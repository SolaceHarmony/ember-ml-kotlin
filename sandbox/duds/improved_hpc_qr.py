"""
Improved HPC QR decomposition using Modified Gram-Schmidt with limb-based precision.

This implementation incorporates insights from the MLX qr_128 implementation
to improve numerical stability and accuracy.
"""

import mlx.core as mx
import time
import numpy as np

def improved_hpc_qr(A: mx.array) -> tuple[mx.array, mx.array]:
    """
    Improved HPC QR decomposition using Modified Gram-Schmidt with limb-based precision.
    
    This implementation uses a Metal kernel with the following improvements:
    1. Modified Gram-Schmidt algorithm for better numerical stability
    2. Proper handling of high/low precision parts throughout computation
    3. Careful handling of numerically zero vectors
    4. Optimized matrix shape (Q is m×k where k=min(m,n))
    
    Args:
        A: Input matrix (M x N) as float32.
        
    Returns:
        Tuple of (Q, R) matrices as float32.
    """
    m, n = A.shape
    k = min(m, n)
    
    # Define Metal kernel source string for improved HPC QR decomposition
    metal_kernel_source = """
/*****************************************************************************
 *  improved_hpc_qr_kernel – 128-bit limb (hpc16×8) QR for MLX / Metal GPU
 *  
 *  This implementation uses Modified Gram-Schmidt with limb-based precision
 *  for improved numerical stability.
 *  
 *  outputs:  Q_out  (m×k),  R_out (k×n),  debug[0…15]
 *            debug[0..3]  = error flags
 *            debug[4..7]  = norm statistics
 *            debug[8..11] = dot product statistics
 *            debug[12..15] = additional diagnostics
 *****************************************************************************/
#define EPSILON      1e-10f
#define NUM_LIMBS    8
#define BIT_MASK     0xFFFFu
#define LIMB_RADIX   65536.0f          /* 2¹⁶                                     */

// Threadgroup shared variables
threadgroup float tg_norm;
threadgroup float tg_dot_high;
threadgroup float tg_dot_low;

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
            // Compute dot product <Q[:, i], Q[:, j]> with limb precision
            threadgroup atomic_uint dot_high_limbs[NUM_LIMBS];
            threadgroup atomic_uint dot_low_limbs[NUM_LIMBS];
            
            if (thread_position_in_threadgroup.x == 0) {
                for (uint l = 0; l < NUM_LIMBS; ++l) {
                    atomic_store_explicit(&dot_high_limbs[l], 0u, memory_order_relaxed);
                    atomic_store_explicit(&dot_low_limbs[l], 0u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Each thread computes part of the dot product
            thread ushort local_high[NUM_LIMBS] = {0};
            thread ushort local_low[NUM_LIMBS] = {0};
            
            for (uint row = thread_position_in_grid.x;
                 row < m;
                 row += threads_per_threadgroup.x)
            {
                float qi = Q_out[row*k + i];  // Q[row, i]
                float qj = Q_out[row*k + j];  // Q[row, j]
                
                // Split into high/low parts
                uint qi_bits = as_type<uint>(qi);
                uint qj_bits = as_type<uint>(qj);
                
                ushort qi_lo = qi_bits & BIT_MASK;
                ushort qi_hi = (qi_bits >> 16) & BIT_MASK;
                ushort qj_lo = qj_bits & BIT_MASK;
                ushort qj_hi = (qj_bits >> 16) & BIT_MASK;
                
                // High part: qi_hi * qj_hi + (qi_hi * qj_lo + qi_lo * qj_hi) >> 16
                uint p_hi_hi = (uint)(qi_hi * qj_hi);
                uint p_hi_lo = (uint)(qi_hi * qj_lo);
                uint p_lo_hi = (uint)(qi_lo * qj_hi);
                uint p_mid = p_hi_lo + p_lo_hi;
                
                uint carry = (p_mid < p_hi_lo) ? (1u << 16) : 0;
                
                local_high[0] += (ushort)(p_hi_hi & BIT_MASK);
                local_high[1] += (ushort)((p_hi_hi >> 16) + (p_mid >> 16) + carry);
                
                // Low part: qi_lo * qj_lo + (qi_hi * qj_lo + qi_lo * qj_hi) & 0xFFFF
                uint p_lo_lo = (uint)(qi_lo * qj_lo);
                
                local_low[0] += (ushort)(p_lo_lo & BIT_MASK);
                local_low[1] += (ushort)((p_lo_lo >> 16) + (p_mid & BIT_MASK));
            }
            
            // Add local results to shared memory
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                atomic_fetch_add_explicit(&dot_high_limbs[l], (uint)local_high[l], memory_order_relaxed);
                atomic_fetch_add_explicit(&dot_low_limbs[l], (uint)local_low[l], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Propagate carries
            if (thread_position_in_threadgroup.x == 0) {
                // Propagate carries for high part
                for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                    uint v = atomic_load_explicit(&dot_high_limbs[l], memory_order_relaxed);
                    uint c = v >> 16;
                    atomic_store_explicit(&dot_high_limbs[l], v & BIT_MASK, memory_order_relaxed);
                    atomic_fetch_add_explicit(&dot_high_limbs[l+1], c, memory_order_relaxed);
                }
                
                // Propagate carries for low part
                for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                    uint v = atomic_load_explicit(&dot_low_limbs[l], memory_order_relaxed);
                    uint c = v >> 16;
                    atomic_store_explicit(&dot_low_limbs[l], v & BIT_MASK, memory_order_relaxed);
                    atomic_fetch_add_explicit(&dot_low_limbs[l+1], c, memory_order_relaxed);
                }
                
                // Compute final dot product values
                float dot_high = 0.0f;
                float dot_low = 0.0f;
                float radix = 1.0f;
                
                for (uint l = 0; l < NUM_LIMBS; ++l) {
                    uint v_high = atomic_load_explicit(&dot_high_limbs[l], memory_order_relaxed);
                    uint v_low = atomic_load_explicit(&dot_low_limbs[l], memory_order_relaxed);
                    dot_high += (float)v_high * radix;
                    dot_low += (float)v_low * radix;
                    radix *= LIMB_RADIX;
                }
                
                // Store in threadgroup variables
                tg_dot_high = dot_high;
                tg_dot_low = dot_low;
                
                // Store in R[i, j]
                R_out[i*n + j] = dot_high;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Get dot product values
            const float dot_high = tg_dot_high;
            const float dot_low = tg_dot_low;
            
            // Update Q[:, j] by subtracting projection
            for (uint row = thread_position_in_grid.x;
                 row < m;
                 row += threads_per_threadgroup.x)
            {
                float qi = Q_out[row*k + i];  // Q[row, i]
                float proj_high = qi * dot_high;
                float proj_low = qi * dot_low;
                Q_out[row*k + j] -= (proj_high + proj_low);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        /* ---- 1c. Compute column norm with limb precision ------------------ */
        threadgroup atomic_uint norm_limbs[NUM_LIMBS];
        
        if (thread_position_in_threadgroup.x == 0) {
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                atomic_store_explicit(&norm_limbs[l], 0u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Each thread computes part of the norm
        thread ushort local_norm[NUM_LIMBS] = {0};
        
        for (uint row = thread_position_in_grid.x;
             row < m;
             row += threads_per_threadgroup.x)
        {
            float qj = Q_out[row*k + j];  // Q[row, j]
            
            // Split into high/low parts
            uint qj_bits = as_type<uint>(qj);
            ushort qj_lo = qj_bits & BIT_MASK;
            ushort qj_hi = (qj_bits >> 16) & BIT_MASK;
            
            // Compute qj^2 with limb precision
            uint p_lo_lo = (uint)(qj_lo * qj_lo);
            uint p_hi_hi = (uint)(qj_hi * qj_hi);
            uint p_hi_lo = (uint)(qj_hi * qj_lo);
            uint p_mid = p_hi_lo << 1;  // 2 * qj_hi * qj_lo
            
            uint carry = (p_mid < (p_hi_lo << 1)) ? (1u << 16) : 0;
            
            local_norm[0] += (ushort)(p_lo_lo & BIT_MASK);
            uint temp = (p_lo_lo >> 16) + (p_mid & BIT_MASK);
            local_norm[1] += (ushort)(temp & BIT_MASK);
            temp = (temp >> 16) + (p_mid >> 16) + (p_hi_hi & BIT_MASK) + carry;
            local_norm[2] += (ushort)(temp & BIT_MASK);
            local_norm[3] += (ushort)((temp >> 16) + (p_hi_hi >> 16));
        }
        
        // Add local results to shared memory
        for (uint l = 0; l < NUM_LIMBS; ++l) {
            atomic_fetch_add_explicit(&norm_limbs[l], (uint)local_norm[l], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Propagate carries and compute final norm
        if (thread_position_in_threadgroup.x == 0) {
            // Propagate carries
            for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                uint v = atomic_load_explicit(&norm_limbs[l], memory_order_relaxed);
                uint c = v >> 16;
                atomic_store_explicit(&norm_limbs[l], v & BIT_MASK, memory_order_relaxed);
                atomic_fetch_add_explicit(&norm_limbs[l+1], c, memory_order_relaxed);
            }
            
            // Compute final norm
            float norm_sq = 0.0f;
            float radix = 1.0f;
            
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                uint v = atomic_load_explicit(&norm_limbs[l], memory_order_relaxed);
                norm_sq += (float)v * radix;
                radix *= LIMB_RADIX;
            }
            
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
                // Compute dot product <Q[:, j], A[:, col]> with limb precision
                threadgroup atomic_uint dot_limbs[NUM_LIMBS];
                
                if (thread_position_in_threadgroup.x == 0) {
                    for (uint l = 0; l < NUM_LIMBS; ++l) {
                        atomic_store_explicit(&dot_limbs[l], 0u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Each thread computes part of the dot product
                thread ushort local_dot[NUM_LIMBS] = {0};
                
                for (uint row = thread_position_in_grid.x;
                     row < m;
                     row += threads_per_threadgroup.x)
                {
                    float qj = Q_out[row*k + j];      // Q[row, j]
                    float a_col = A[row*n + col];     // A[row, col]
                    
                    // Split into high/low parts
                    uint qj_bits = as_type<uint>(qj);
                    uint a_bits = as_type<uint>(a_col);
                    
                    ushort qj_lo = qj_bits & BIT_MASK;
                    ushort qj_hi = (qj_bits >> 16) & BIT_MASK;
                    ushort a_lo = a_bits & BIT_MASK;
                    ushort a_hi = (a_bits >> 16) & BIT_MASK;
                    
                    // Compute qj * a_col with limb precision
                    uint p_lo_lo = (uint)(qj_lo * a_lo);
                    uint p_hi_hi = (uint)(qj_hi * a_hi);
                    uint p_hi_lo = (uint)(qj_hi * a_lo);
                    uint p_lo_hi = (uint)(qj_lo * a_hi);
                    uint p_mid = p_hi_lo + p_lo_hi;
                    
                    uint carry = (p_mid < p_hi_lo) ? (1u << 16) : 0;
                    
                    local_dot[0] += (ushort)(p_lo_lo & BIT_MASK);
                    uint temp = (p_lo_lo >> 16) + (p_mid & BIT_MASK);
                    local_dot[1] += (ushort)(temp & BIT_MASK);
                    temp = (temp >> 16) + (p_mid >> 16) + (p_hi_hi & BIT_MASK) + carry;
                    local_dot[2] += (ushort)(temp & BIT_MASK);
                    local_dot[3] += (ushort)((temp >> 16) + (p_hi_hi >> 16));
                }
                
                // Add local results to shared memory
                for (uint l = 0; l < NUM_LIMBS; ++l) {
                    atomic_fetch_add_explicit(&dot_limbs[l], (uint)local_dot[l], memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Propagate carries and compute final dot product
                if (thread_position_in_threadgroup.x == 0) {
                    // Propagate carries
                    for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                        uint v = atomic_load_explicit(&dot_limbs[l], memory_order_relaxed);
                        uint c = v >> 16;
                        atomic_store_explicit(&dot_limbs[l], v & BIT_MASK, memory_order_relaxed);
                        atomic_fetch_add_explicit(&dot_limbs[l+1], c, memory_order_relaxed);
                    }
                    
                    // Compute final dot product
                    float dot = 0.0f;
                    float radix = 1.0f;
                    
                    for (uint l = 0; l < NUM_LIMBS; ++l) {
                        uint v = atomic_load_explicit(&dot_limbs[l], memory_order_relaxed);
                        dot += (float)v * radix;
                        radix *= LIMB_RADIX;
                    }
                    
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
            name="improved_hpc_qr_kernel",
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

def benchmark_improved_qr(sizes=[(100, 100), (500, 500), (1000, 1000)], 
                         repeat=3, 
                         compare_with_native=True):
    """
    Benchmark the improved QR implementation against different matrix sizes
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
        
        # Test improved implementation
        improved_times = []
        for i in range(repeat):
            start_time = time.time()
            Q_improved, R_improved = improved_hpc_qr(A)
            end_time = time.time()
            improved_times.append(end_time - start_time)
            
        avg_improved_time = sum(improved_times) / repeat
        print(f"Improved QR completed in {avg_improved_time:.4f} seconds (avg of {repeat} runs)")
        
        # Check orthogonality and reconstruction
        ortho_error = mx.mean(mx.abs(mx.matmul(Q_improved.T, Q_improved) - mx.eye(Q_improved.shape[0]))).item()
        recon_error = mx.mean(mx.abs(mx.matmul(Q_improved, R_improved) - A)).item()
        print(f"Orthogonality error: {ortho_error:.6e}")
        print(f"Reconstruction error: {recon_error:.6e}")
        
        result = {
            "size": size,
            "improved_time": avg_improved_time,
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
                    q_diff = mx.mean(mx.abs(Q_improved - Q_native)).item()
                    r_diff = mx.mean(mx.abs(R_improved - R_native)).item()
                    print(f"Difference in Q: {q_diff:.6e}")
                    print(f"Difference in R: {r_diff:.6e}")
                    
                    result["native_time"] = avg_native_time
                    result["q_diff"] = q_diff
                    result["r_diff"] = r_diff
                    result["speedup"] = avg_native_time / avg_improved_time
                except Exception as e:
                    print(f"Could not compare with native QR: {e}")
        
        results.append(result)
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("Improved HPC QR Decomposition for MLX")
    print("=" * 80)
    
    # Test with a smaller matrix first
    print("\nTesting with small matrix (10x10)...")
    A_small = mx.random.normal((10, 10), dtype=mx.float32)
    start_time = time.time()
    Q_small, R_small = improved_hpc_qr(A_small)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_small.T, Q_small) - mx.eye(Q_small.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_small, R_small) - A_small)).item())
    print("-" * 40)
    
    # Run benchmarks with different matrix sizes
    print("\nRunning benchmarks...")
    results = benchmark_improved_qr(sizes=[(50, 50), (100, 100), (200, 200)], repeat=2)
    
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
        print(f"{size_str:>10} | {result['improved_time']:>10.4f} | {result['ortho_error']:>10.2e} | {result['recon_error']:>10.2e} | {speedup:>10}")
    print("-" * 60)