from __future__ import annotations
"""
MLX QR decomposition operations for ember_ml.

This module provides MLX implementations of QR decomposition using
a custom Metal kernel for GPU acceleration.
"""

from typing import Tuple, Optional, Any
import time
import signal
import mlx.core as mx

# Import from tensor_ops and types
from ember_ml.backend.mlx.types import TensorLike


# ----------------------------------------------------------------------------
# 1 · Metal kernel                                                            #
# ----------------------------------------------------------------------------
_ENHANCED_QR_SRC = r"""

/* ------------------------------------------------------------------ constants */
#define EPSILON     1e-10f
#define NUM_LIMBS   8u          // 128-bit accumulator (8 × 16-bit)
#define LIMB_RADIX  65536.0f    // 2¹⁶
#define WARP_SIZE   32u         // Threads per warp/wavefront

/* ------------------------------------------------------------------ safety limits */
#define MAX_ITERATIONS 1000u    // Maximum number of iterations to prevent infinite loops
#define MAX_MATRIX_DIM 5000u    // Maximum matrix dimension
#define MAX_WORK_PER_THREAD 10000u // Maximum work units per thread
    // Correctly use Metal built-in variables
    uint                tid = thread_position_in_grid.x;     // Thread ID in grid
    uint3               tpg = threads_per_threadgroup;     // Threadgroup size
    uint3               gpg = grid_size;     // Grid size
    uint                simd_lane_id = tid % WARP_SIZE;  // Lane ID within SIMD group
    uint                simd_group_id = tid / WARP_SIZE; // SIMD group ID

    const uint m        = A_shape[0];
    const uint n        = A_shape[1];
    const uint min_dim  = (m < n ? m : n);
    const uint grid_sz  = gpg.x * tpg.x;    // Total number of threads in grid
    
    /* --- CIRCUIT BREAKER CHECKS --- */
    if (tid == 0) {
        // Check matrix dimensions
        if (m > MAX_MATRIX_DIM || n > MAX_MATRIX_DIM) {
            dbg[0] = 0.0f;  // Execution failed
            dbg[13] = 1.0f; // Matrix too large error flag
            dbg[14] = float(m > MAX_MATRIX_DIM ? m : n); // Store the problematic dimension
            return; // Early exit
        }
        
        // STRICT WORKLOAD CHECK: Exactly as suggested by the user
        if (m <= 1000 && n <= 1000) {  // Prevent overflow
            uint total_elements = m * m;  // Q matrix elements
            total_elements += m * n; // R matrix elements
            total_elements += m * n; // Additional work for reflections
            
            // Store total elements for debugging
            dbg[7] = float(total_elements);
            
            // Check if we have enough threads
            if (grid_sz == 0) {
                dbg[0] = 0.0f;  // Execution failed
                dbg[13] = 2.0f; // Workload too large error flag
                dbg[14] = 0.0f; // Division by zero
                return; // Early exit
            }
            
            // Calculate work per thread
            uint work_per_thread = total_elements / grid_sz;
            
            // Store work per thread for debugging
            dbg[10] = float(work_per_thread);
            
            // Check if work per thread is reasonable
            if (work_per_thread > MAX_WORK_PER_THREAD) {
                dbg[0] = 0.0f;  // Execution failed
                dbg[13] = 2.0f; // Workload too large error flag
                dbg[14] = float(work_per_thread); // Store the workload
                return; // Early exit
            }
            
            // All checks passed
            dbg[0] = 1.0f;  // Execution started successfully
        } else {
            // Matrix dimensions too large
            dbg[0] = 0.0f;  // Execution failed
            dbg[13] = 2.0f; // Workload too large error flag
            dbg[14] = float(m * n); // Store matrix size
            return; // Early exit
        }
        
        // Store matrix and allocation information
        dbg[1] = float(m);  // Number of rows
        dbg[2] = float(n);  // Number of columns
        dbg[3] = float(min_dim); // min(m, n)
        dbg[4] = float(tid);  // Thread ID
        dbg[5] = float(tpg.x);  // Threads per threadgroup
        dbg[6] = float(grid_sz);  // Total threads (grid_sz)
        
        // Store allocation information
        float q_elements_f = float(m * m);
        float r_elements_f = float(m * n);
        dbg[7] = q_elements_f;  // Q matrix elements
        dbg[8] = r_elements_f;  // R matrix elements
        dbg[9] = q_elements_f + r_elements_f;  // Total elements
        dbg[10] = float(work_per_thread);  // Work per thread
    }
    
    // Ensure all threads see the safety check results
    threadgroup_barrier(mem_flags::mem_device);
    
    // Exit if safety checks failed
    if (dbg[0] < 0.5f) {
        return; // All threads exit
    }

    /* 0 · initialise Q ← I,  R ← A (following prototype approach) */
    // Calculate total elements for initialization
    const uint q_elements = m * m;
    const uint r_elements = m * n;
    const uint total_init_elements = q_elements + r_elements;
    
    // Use conditional logic like the prototype to ensure each thread
    // only processes elements it's responsible for
    if (tid < total_init_elements) {
        // Initialize Q_out elements
        if (tid < q_elements) {
            uint row = tid / m; // Assuming row-major
            uint col = tid % m;
            Q_out[tid] = (row == col) ? 1.0f : 0.0f;
        }
        // Initialize R_out elements by copying from A
        else {
            uint r_idx = tid - q_elements;
            R_out[r_idx] = A[r_idx]; // Direct linear copy
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    /* ===== parallel QR decomposition ===== */
    uint iteration_count = 0; // Track iterations to prevent infinite loops
    
    for (uint k = 0; k < min_dim; ++k)
    {
        // CIRCUIT BREAKER: Check iteration count
        if (iteration_count++ > MAX_ITERATIONS) {
            if (tid == 0) {
                dbg[0] = 0.0f;  // Execution failed
                dbg[13] = 3.0f; // Too many iterations error flag
                dbg[14] = float(iteration_count); // Store the iteration count
                dbg[15] = 0.0f; // Clear success flag
            }
            return; // All threads exit
        }
        /* -- column scaling (improves robustness) ------------------ */
        // Each thread finds max in its assigned range
        float thread_max = 0.0f;
        for (uint i = k + tid; i < m; i += grid_sz) {
            thread_max = fmax(thread_max, fabs(R_out[i*n + k]));
        }
        
        // Reduce max across threads in same SIMD group
        thread_max = simd_max(thread_max);
        
        // First thread in each SIMD group writes to shared memory
        threadgroup float simd_max[8]; // Assuming max 8 SIMD groups
        if (simd_lane_id == 0 && simd_group_id < 8) {
            simd_max[simd_group_id] = thread_max;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread finds max across SIMD groups
        float cmax = 0.0f;
        if (tid == 0) {
            for (uint i = 0; i < min(8u, (grid_sz + WARP_SIZE - 1) / WARP_SIZE); ++i) {
                cmax = fmax(cmax, simd_max[i]);
            }
            dbg[10] = cmax;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Broadcast cmax to all threads
        if (tid == 0) {
            simd_max[0] = cmax;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        cmax = simd_max[0];
        float scale = (cmax > EPSILON ? 1.0f / cmax : 1.0f);
        
        // Scale column k in parallel
        for (uint i = k + tid; i < m; i += grid_sz) {
            R_out[i*n + k] *= scale;
        }
        
        threadgroup_barrier(mem_flags::mem_device);

        /* -- build Householder v ----------------------------------- */
        // Each thread computes partial sum for sigma
        float partial_sigma = 0.0f;
        for (uint i = k + tid; i < m; i += grid_sz) {
            float v = R_out[i*n + k];
            partial_sigma += v*v;
        }
        
        // Reduce sigma across threads
        partial_sigma = simd_sum(partial_sigma);
        
        // First thread in each SIMD group writes to shared memory
        threadgroup float simd_sigma[8];
        if (simd_lane_id == 0 && simd_group_id < 8) {
            simd_sigma[simd_group_id] = partial_sigma;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread combines results
        float sigma = 0.0f;
        float norm = 0.0f;
        float sign = 0.0f;
        if (tid == 0) {
            for (uint i = 0; i < min(8u, (grid_sz + WARP_SIZE - 1) / WARP_SIZE); ++i) {
                sigma += simd_sigma[i];
            }
            dbg[4] = sigma;
            norm = sqrt(sigma);
            dbg[5] = norm;
            
            // CIRCUIT BREAKER: Check for NaN or Inf
            if (isnan(sigma) || isinf(sigma) || isnan(norm) || isinf(norm)) {
                dbg[0] = 0.0f;  // Execution failed
                dbg[13] = 4.0f; // Numerical instability error flag
                dbg[14] = sigma; // Store the problematic value
                dbg[15] = 0.0f; // Clear success flag
                simd_sigma[0] = -2.0f; // Signal fatal error
                return; // Early exit
            }
            
            // Check if norm is too small
            if (norm < EPSILON) {
                // Unscale the diagonal element
                R_out[k*n + k] /= scale;
                simd_sigma[0] = -1.0f; // Signal to skip this iteration
            } else {
                sign = (R_out[k*n + k] >= 0.0f ? 1.0f : -1.0f);
                R_out[k*n + k] += sign * norm;  // v₀ update
                simd_sigma[0] = 0.0f; // Signal to continue
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Check if we should skip this iteration
        if (simd_sigma[0] < 0.0f) {
            continue;
        }
        
        threadgroup_barrier(mem_flags::mem_device);

        /* -- limb-precision vᵀv (parallel version) ----------------- */
        // Each thread processes its assigned elements
        threadgroup uint thread_limbs[WARP_SIZE * NUM_LIMBS];
        
        // Initialize thread_limbs to zero
        for (uint l = tid; l < WARP_SIZE * NUM_LIMBS; l += grid_sz) {
            thread_limbs[l] = 0u;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Each thread computes partial limbs
        uint local_limb[NUM_LIMBS] = {0u};
        for (uint i = k + tid; i < m; i += grid_sz) {
            uint bits = as_type<uint>(R_out[i*n + k]);
            ushort lo = bits & 0xFFFFu;
            ushort hi = (bits >> 16) & 0xFFFFu;
            uint p0 = uint(lo*lo);
            uint p1 = uint(hi*hi);
            uint pc = uint(lo*hi) << 1;

            local_limb[0] +=  p0 & 0xFFFFu;
            local_limb[1] += (p0 >> 16) + (pc & 0xFFFFu);
            local_limb[2] += (pc >> 16) + (p1 & 0xFFFFu);
            local_limb[3] +=  p1 >> 16;
        }
        
        // Store local limbs to shared memory
        for (uint l = 0; l < NUM_LIMBS; ++l) {
            thread_limbs[tid * NUM_LIMBS + l] = local_limb[l];
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // First thread combines all limbs
        uint combined_limb[NUM_LIMBS] = {0u};
        float vtv = 0.0f;
        float inv_vtv = 0.0f;
        
        if (tid == 0) {
            // Combine all thread limbs
            for (uint t = 0; t < grid_sz; ++t) {
                for (uint l = 0; l < NUM_LIMBS; ++l) {
                    combined_limb[l] += thread_limbs[t * NUM_LIMBS + l];
                }
            }
            
            // Carry propagation
            for (uint l = 0; l < NUM_LIMBS-1; ++l) {
                uint carry = combined_limb[l] >> 16;
                combined_limb[l] &= 0xFFFFu;
                combined_limb[l+1] += carry;
            }
            
            // Convert to float
            float radix = 1.0f;
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                vtv += float(combined_limb[l]) * radix;
                radix *= LIMB_RADIX;
            }
            
            dbg[6] = vtv;
            
            // CIRCUIT BREAKER: Check for extreme values in vtv
            if (isnan(vtv) || isinf(vtv) || vtv > 1e20f) {
                dbg[0] = 0.0f;  // Execution failed
                dbg[13] = 5.0f; // VTV numerical instability error flag
                dbg[14] = vtv;  // Store the problematic value
                dbg[15] = 0.0f; // Clear success flag
                thread_limbs[0] = 2u; // Signal fatal error
                return; // Early exit
            }
            
            inv_vtv = (vtv > EPSILON ? 1.0f / vtv : 0.0f);
            dbg[7] = inv_vtv;
            
            // Store for other threads to access
            thread_limbs[0] = (inv_vtv == 0.0f ? 1u : 0u); // Flag for skipping
            
            // Store vtv and inv_vtv for other threads
            simd_sigma[1] = vtv;
            simd_sigma[2] = inv_vtv;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Check thread_limbs[0] status
        if (thread_limbs[0] == 2u) {
            // Fatal error detected, exit kernel
            return;
        }
        else if (thread_limbs[0] == 1u) {
            // Skip this iteration (non-fatal)
            // Unscale the column in parallel
            for (uint i = k + tid; i < m; i += grid_sz) {
                R_out[i*n + k] /= scale;
            }
            continue;
        }
        
        // Get shared values
        vtv = simd_sigma[1];
        inv_vtv = simd_sigma[2];
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- reflect R (k … n-1) in parallel ----------------------- */
        // Each thread handles a subset of columns
        for (uint j = k + tid; j < n; j += grid_sz) {
            // Calculate dot product for this column
            float dot = 0.0f;
            for (uint i = k; i < m; ++i) {
                dot += R_out[i*n + k] * R_out[i*n + j];
            }
            
            // Store debug info for first column
            if (j == k && tid == 0) {
                dbg[8] = dot;
            }
            
            float beta = 2.0f * dot * inv_vtv;
            
            // Update column j
            for (uint i = k; i < m; ++i) {
                R_out[i*n + j] -= beta * R_out[i*n + k];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- reflect Q (0 … m-1) in parallel ----------------------- */
        // Each thread handles a subset of columns
        for (uint j = tid; j < m; j += grid_sz) {
            // Calculate dot product for this column
            float dot = 0.0f;
            for (uint i = k; i < m; ++i) {
                dot += R_out[i*n + k] * Q_out[i*m + j];
            }
            
            // Store debug info for first column
            if (j == 0 && tid == 0) {
                dbg[9] = dot;
            }
            
            float beta = 2.0f * dot * inv_vtv;
            
            // Update column j
            for (uint i = k; i < m; ++i) {
                Q_out[i*m + j] -= beta * R_out[i*n + k];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        /* -- un-scale column k in parallel ------------------------- */
        for (uint i = k + tid; i < m; i += grid_sz) {
            R_out[i*n + k] /= scale;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }

    /* -- force R upper-triangular (in parallel) -------------------- */
    for (uint r = 1 + tid; r < m; r += grid_sz) {
        for (uint c = 0; c < min(r, n); ++c) {
            R_out[r*n + c] = 0.0f;
        }
    }
    
    // CIRCUIT BREAKER: Final check for iteration count
    if (tid == 0) {
        // Store iteration count for debugging
        dbg[12] = float(iteration_count);
        
        // Set success flag only if we didn't hit the iteration limit
        if (iteration_count < MAX_ITERATIONS) {
            dbg[15] = 1.0f;     // success flag
        } else {
            dbg[15] = 0.0f;     // failure flag
        }
    }
"""

# ----------------------------------------------------------------------------
# 2 · compile the kernel                                                      #
# ----------------------------------------------------------------------------
_ENHANCED_QR_KERNEL = mx.fast.metal_kernel(
    name              = "enhanced_hpc_qr_kernel",
    source            = _ENHANCED_QR_SRC,
    input_names       = ["A"],  # Let MLX automatically handle A_shape
    output_names      = ["Q_out", "R_out", "dbg"],
    ensure_row_contiguous=True
)

# ----------------------------------------------------------------------------
# 3 · resource management functions                                           #
# ----------------------------------------------------------------------------

# Device-specific information for Apple M3 Ultra
DEVICE_INFO = {
    'resource_limit': 499000,
    'max_buffer_length': 167503724544,
    'architecture': 'applegpu_g15d',
    'memory_size': 274877906944,
    'max_recommended_working_set_size': 223338299392,
    'device_name': 'Apple M3 Ultra'
}

def calculate_optimal_thread_allocation(m: int, n: int, debug: bool = False) -> dict:
    """
    Calculate optimal thread allocation for QR decomposition based on matrix dimensions
    and device-specific information.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing grid and threadgroup configurations
    """
    # Calculate total elements and work
    q_elements = m * m
    r_elements = m * n
    total_elements = q_elements + r_elements
    total_work = total_elements + m * n  # Include reflection operations
    
    # Get Metal device capabilities from device info
    thread_execution_width = 32  # Typical for Apple GPUs
    max_threads_per_threadgroup = 1024  # Maximum threads per threadgroup
    
    # For Apple M3 Ultra, we can use device-specific optimizations
    if DEVICE_INFO['device_name'] == 'Apple M3 Ultra':
        # For very small matrices (up to 8x8), use minimal threads
        if max(m, n) <= 8:
            threadgroup_size = thread_execution_width
            total_threads = threadgroup_size
        # For small matrices (up to 32x32), use one thread per element
        elif max(m, n) <= 32:
            threadgroup_size = thread_execution_width
            total_threads = min(total_elements, 256)
        # For medium matrices, scale more conservatively
        else:
            threadgroup_size = thread_execution_width
            # Use at most 1024 threads for any matrix
            total_threads = min(1024, total_elements)
    else:
        # Generic approach for unknown devices
        # Calculate optimal threadgroup size (multiple of thread_execution_width)
        threadgroup_size = thread_execution_width
        while threadgroup_size * 2 <= max_threads_per_threadgroup and threadgroup_size * 2 <= total_elements:
            threadgroup_size *= 2
        
        # Calculate optimal number of threads (one per element, but capped)
        MAX_TOTAL_THREADS = 1024  # Hard cap on total threads - much more conservative
        total_threads = min(total_elements, MAX_TOTAL_THREADS)
    
    # Round up to multiple of threadgroup_size
    total_threads = ((total_threads + threadgroup_size - 1) // threadgroup_size) * threadgroup_size
    
    # Calculate grid and threadgroup
    grid = (total_threads, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)
    
    # Calculate work per thread
    work_per_thread = total_work / total_threads if total_threads > 0 else float('inf')
    
    # Check if work per thread is reasonable - use a more conservative limit for small matrices
    MAX_WORK_PER_THREAD = 1000 if max(m, n) <= 32 else 10000
    is_safe = work_per_thread <= MAX_WORK_PER_THREAD
    
    result = {
        'grid': grid,
        'threadgroup': threadgroup,
        'total_threads': total_threads,
        'threadgroup_size': threadgroup_size,
        'work_per_thread': work_per_thread,
        'is_safe': is_safe,
        'total_elements': total_elements,
        'total_work': total_work,
        'device': DEVICE_INFO['device_name']
    }
    
    if debug:
        print("\nThread Allocation Details:")
        print(f"  Device: {DEVICE_INFO['device_name']}")
        print(f"  Matrix dimensions: {m}x{n}")
        print(f"  Total elements: {total_elements}")
        print(f"  Total work: {total_work}")
        print(f"  Threadgroup size: {threadgroup_size}")
        print(f"  Total threads: {total_threads}")
        print(f"  Work per thread: {work_per_thread:.2f}")
        print(f"  Max work per thread: {MAX_WORK_PER_THREAD}")
        print(f"  Is safe: {is_safe}")
    
    return result

# ----------------------------------------------------------------------------
# 4 · qr function                                                             #
# ----------------------------------------------------------------------------

def qr(A: TensorLike,
           dtype: Optional[Any] = mx.float32,
         *,
         debug: bool = False,
         max_matrix_size: int = 2000,  # Safety limit for matrix dimensions
         max_threads: int = 100000,    # Safety limit for thread count
         timeout_seconds: float = 10.0 # Safety timeout for kernel execution
        ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Numerically-stable QR with limb accumulation using Metal GPU acceleration.
    
    This implementation uses a highly parallelized Metal kernel for maximum performance
    on Apple GPUs.

    Args:
        A: Input matrix
        dtype: Data type for computation
        debug: Whether to return debug info and print detailed debug information
        max_matrix_size: Maximum allowed dimension for input matrix (safety circuit breaker)
        max_threads: Maximum allowed thread count (safety circuit breaker)
        timeout_seconds: Maximum allowed execution time in seconds (safety circuit breaker)
        
    Returns (Q, R[, dbg]).
    
    Raises:
        ValueError: If matrix dimensions exceed max_matrix_size
        ValueError: If required thread count exceeds max_threads
        TimeoutError: If kernel execution exceeds timeout_seconds
    """
    from ember_ml.backend.mlx.tensor import MLXTensor

    if debug:
        print("\n=== QR Decomposition Debug ===")
        print(f"Input type: {type(A)}")
    
    # Convert input to MLX tensor
    A = MLXTensor().convert_to_tensor(A, dtype=dtype)
    
    if debug:
        print(f"Converted input type: {type(A)}")
        print(f"Input dtype: {A.dtype}")
    
    # Get dimensions
    m, n = A.shape
    
    # STRICT CIRCUIT BREAKER: Hard limit on matrix dimensions
    MAX_SAFE_DIM = 1000  # Absolute maximum dimension allowed
    if m > MAX_SAFE_DIM or n > MAX_SAFE_DIM:
        error_msg = f"Matrix dimensions ({m}x{n}) exceed maximum safe size ({MAX_SAFE_DIM})"
        if debug:
            print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Secondary check against user-provided limit
    if m > max_matrix_size or n > max_matrix_size:
        error_msg = f"Matrix dimensions ({m}x{n}) exceed maximum allowed size ({max_matrix_size})"
        if debug:
            print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    if debug:
        print(f"Matrix dimensions: {m}x{n}")
        print(f"Input matrix:\n{A}")
    
    if debug:
        print(f"Matrix shape: [{m}, {n}]")
    
    # Prepare outputs
    Q = mx.zeros((m, m), dtype=mx.float32)
    R = mx.zeros((m, n), dtype=mx.float32)
    dbg = mx.zeros(16, dtype=mx.float32)
    
    if debug:
        print(f"Output shapes: Q={Q.shape}, R={R.shape}, dbg={dbg.shape}")
        print(f"Output dtypes: Q={Q.dtype}, R={R.dtype}, dbg={dbg.dtype}")
    
    # Calculate total elements for initialization (following prototype approach)
    q_elements = m * m  # Elements in Q matrix
    r_elements = m * n  # Elements in R matrix
    total_init_elements = q_elements + r_elements  # Total elements to process
    
    # Configure kernel execution
    threadgroup_size = 32  # Use a reasonable threadgroup size
    
    # Define MAX_WORK_PER_THREAD consistently with kernel
    MAX_WORK_PER_THREAD = 10000
    
    # STRICT THREAD ALLOCATION: Use exactly the number of threads needed, with hard caps
    
    # Calculate total elements for initialization (following prototype approach)
    total_init_elements = q_elements + r_elements  # Total elements to process
    
    # Calculate total work including reflections
    total_work = q_elements + r_elements + m * n  # Include reflection operations
    
    # SAFETY CHECK: Verify total work is within reasonable bounds
    MAX_TOTAL_WORK = 1000000  # 1 million operations max
    if total_work > MAX_TOTAL_WORK:
        error_msg = f"Total work ({total_work}) exceeds maximum allowed ({MAX_TOTAL_WORK})"
        if debug:
            print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Launch exactly one thread per element for initialization, just like the prototype
    # But cap at a reasonable maximum
    MAX_THREADS = 10000  # Hard cap on thread count
    total_threads_to_launch = min(total_init_elements, MAX_THREADS)
    
    if debug:
        print(f"Following prototype approach: launching {total_threads_to_launch} threads")
        print(f"This is one thread per element, capped at {MAX_THREADS}")
        print(f"Total work to perform: {total_work} operations")
    
    # Round to multiple of threadgroup_size
    total_threads_to_launch = ((total_threads_to_launch + threadgroup_size - 1)
                              // threadgroup_size * threadgroup_size)
    
    if debug:
        print(f"Matrix dimensions (m×n): {m}×{n}")
        print(f"Q elements: {q_elements}, R elements: {r_elements}")
        print(f"Matrix size category: {'Small' if max(m, n) <= 32 else 'Medium' if max(m, n) <= 128 else 'Large'}")
        print(f"Thread allocation: {total_threads_to_launch} threads")
    
    # Make sure we don't exceed max_threads (safety check)
    if total_threads_to_launch > max_threads:
        total_threads_to_launch = max(threadgroup_size, max_threads // threadgroup_size * threadgroup_size)
        if debug:
            print(f"WARNING: Reduced thread count to {total_threads_to_launch} (max: {max_threads})")
    
    # Calculate work per thread for debugging
    required_work = m * m + m * n + m * n
    work_per_thread = required_work / total_threads_to_launch
    
    if debug:
        print(f"Total work: {required_work}")
        print(f"Work per thread: {work_per_thread:.2f}")
    
    # Set grid and threadgroup sizes
    grid = (total_threads_to_launch, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)
    
    if debug:
        print(f"Grid size: {grid}")
        print(f"Threadgroup size: {threadgroup}")
        print(f"Total threads: {grid[0] * grid[1] * grid[2]}")
        print(f"Threads per threadgroup: {threadgroup[0] * threadgroup[1] * threadgroup[2]}")
        print(f"Number of threadgroups: {grid[0] // threadgroup[0]}")
    
    # Execute the Metal kernel
    # Define output shapes and dtypes
    output_shapes = [(m, m), (m, n), (16,)]
    output_dtypes = [mx.float32, mx.float32, mx.float32]
    
    if debug:
        print(f"Output shapes: {output_shapes}")
        print(f"Output dtypes: {output_dtypes}")
        print("Calling Metal kernel...")
    
    # Call the kernel with proper parameters according to MLX API
    if debug:
        print("Calling QR kernel with verbose=True to show generated code...")
    
    start_time = time.time()
    
    # AGGRESSIVE TIMEOUT: Force a short timeout for all kernel executions
    # Default to 1 second if not specified
    if timeout_seconds <= 0:
        timeout_seconds = 1.0
    
    if debug:
        print(f"Setting kernel timeout to {timeout_seconds} seconds")
    
    # Set up a timeout handler
    def timeout_handler(signum, frame):
        error_msg = f"QR kernel execution timed out after {timeout_seconds} seconds"
        print(f"ERROR: {error_msg}")
        raise TimeoutError(error_msg)
    
    # Always set a timeout
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        outputs = _ENHANCED_QR_KERNEL(
            inputs=[A],  # Let MLX automatically handle A_shape
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=grid,
            threadgroup=threadgroup,
            verbose=debug  # Print generated code when debug is True
        )
    except Exception as e:
        print(f"Kernel execution failed with error: {e}")
        raise
    finally:
        signal.alarm(0)  # Disable the alarm
        signal.signal(signal.SIGALRM, original_handler)  # Restore original handler
    end_time = time.time()
    
    if debug:
        print(f"Metal kernel execution completed in {end_time - start_time:.4f} seconds")
    
    Q, R, dbg_out = outputs
    
    if debug:
        print(f"Output Q shape: {Q.shape}, dtype: {Q.dtype}")
        print(f"Output R shape: {R.shape}, dtype: {R.dtype}")
        print(f"Output dbg shape: {dbg_out.shape}, dtype: {dbg_out.dtype}")
        print(f"Debug info: {dbg_out}")
        
        # Check if Q and R contain non-zero values
        q_nonzero = mx.any(mx.abs(Q) > 0).item()
        r_nonzero = mx.any(mx.abs(R) > 0).item()
        print(f"Q contains non-zero values: {q_nonzero}")
        print(f"R contains non-zero values: {r_nonzero}")
        
        # Check if debug info contains non-zero values
        dbg_nonzero = mx.any(mx.abs(dbg_out) > 0).item()
        print(f"Debug info contains non-zero values: {dbg_nonzero}")
        
        # Display kernel allocation and calculation details
        print("\nKernel Allocation and Calculation Details:")
        print(f"  Matrix dimensions (m×n): {int(dbg_out[1])}×{int(dbg_out[2])}")
        print(f"  Min dimension: {int(dbg_out[3])}")
        print(f"  Thread ID: {int(dbg_out[4])}")
        print(f"  Threads per threadgroup: {int(dbg_out[5])}")
        print(f"  Total threads: {int(dbg_out[6])}")
        
        # Display allocation information
        print("\nAllocation Information:")
        if dbg_out[7] > 0:
            print(f"  Q matrix elements: {int(dbg_out[7])}")
        if dbg_out[8] > 0:
            print(f"  R matrix elements: {int(dbg_out[8])}")
        if dbg_out[9] > 0:
            print(f"  Total elements: {int(dbg_out[9])}")
        if dbg_out[10] > 0:
            print(f"  Work per thread: {int(dbg_out[10])}")
        
        # Display calculation details
        print("\nCalculation Details:")
        if dbg_out[11] > 0:
            print(f"  Column max (cmax): {dbg_out[11]}")
        
        # Display iteration and error information
        print("\nIteration and Error Information:")
        print(f"  Iteration count: {int(dbg_out[12]) if dbg_out[12] > 0 else 'N/A'}")
        
        if dbg_out[13] > 0:
            error_codes = {
                1.0: "Matrix too large",
                2.0: "Workload too large",
                3.0: "Too many iterations",
                4.0: "Numerical instability in norm calculation",
                5.0: "Numerical instability in vtv calculation"
            }
            error_code = dbg_out[13]
            error_value = dbg_out[14]
            error_msg = error_codes.get(error_code, "Unknown error")
            print(f"  Error detected: {error_msg} (code: {error_code}, value: {error_value})")
        
        print(f"  Success flag: {'Set' if dbg_out[15] > 0 else 'Not set'}")
        
        # Show all non-zero debug values
        print("\nAll Non-zero Debug Values:")
        for i in range(dbg_out.shape[0]):
            if abs(dbg_out[i]) > 0:
                print(f"  dbg[{i}] = {dbg_out[i]}")
        
        # Validate QR decomposition properties
        # 1. Check Q is orthogonal (Q^T Q = I)
        qtq = mx.matmul(Q.T, Q)
        identity = mx.eye(m)
        q_orthogonal = mx.allclose(qtq, identity, atol=1e-4).item()
        
        # 2. Check R is upper triangular
        r_upper = True
        for i in range(1, m):
            for j in range(min(i, n)):
                if abs(R[i, j]) > 1e-5:
                    r_upper = False
                    break
        
        # 3. Check A = QR reconstruction
        qr_product = mx.matmul(Q, R)
        reconstruction_valid = mx.allclose(qr_product, A, atol=1e-4).item()
        
        print(f"Q orthogonal: {'SUCCESS' if q_orthogonal else 'FAILURE'}")
        print(f"R upper triangular: {'SUCCESS' if r_upper else 'FAILURE'}")
        print(f"A = QR reconstruction: {'SUCCESS' if reconstruction_valid else 'FAILURE'}")
        
        # Check if debug values were set
        debug_values_set = dbg_out[0].item() == 1.0 and dbg_out[15].item() == 1.0
        print(f"Debug values set: {'SUCCESS' if debug_values_set else 'FAILURE'}")
        
        print("=== End of QR Decomposition Debug ===\n")
    
    # Return results
    return (Q, R, dbg_out) if debug else (Q, R)

# ----------------------------------------------------------------------------
# 4 · detailed test function                                                  #
# ----------------------------------------------------------------------------

def test_qr_detailed():
    """Detailed test of the QR implementation with comprehensive diagnostics."""
    print("\n=== QR Implementation Detailed Test ===\n")

    # Create a tiny test matrix (2x2 instead of 3x2)
    a_values = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    a = mx.array(a_values, dtype=mx.float32)
    print(f"Input matrix shape: {a.shape}")
    print(f"Input matrix:\n{a}")

    # Perform QR decomposition with debug info and a short timeout
    print("\nPerforming QR decomposition...")
    q, r, dbg = qr(a, debug=True, timeout_seconds=1.0)

    # Print shapes
    print(f"Q shape: {q.shape}")
    print(f"R shape: {r.shape}")
    print(f"Debug info shape: {dbg.shape}")

    # Print Q and R matrices
    print("\nQ matrix:")
    print(q)

    print("\nR matrix:")
    print(r)

    # Validate QR decomposition properties
    print("\nValidating QR Decomposition Properties:")
    
    # 1. Check Q is orthogonal (Q^T Q = I)
    qtq = mx.matmul(q.T, q)
    identity = mx.eye(q.shape[0])
    q_orthogonal = mx.allclose(qtq, identity, atol=1e-4).item()
    
    # 2. Check R is upper triangular
    r_upper = True
    for i in range(1, a.shape[0]):
        for j in range(min(i, a.shape[1])):
            if abs(r[i, j]) > 1e-5:
                r_upper = False
                break
    
    # 3. Check A = QR reconstruction
    qr_product = mx.matmul(q, r)
    reconstruction_valid = mx.allclose(qr_product, a, atol=1e-4).item()
    
    print(f"Q orthogonal: {'SUCCESS' if q_orthogonal else 'FAILURE'}")
    print(f"R upper triangular: {'SUCCESS' if r_upper else 'FAILURE'}")
    print(f"A = QR reconstruction: {'SUCCESS' if reconstruction_valid else 'FAILURE'}")

    # Check debug values
    print("\nChecking Debug Values:")
    kernel_executed = dbg[0].item() > 0
    print(f"Kernel executed: {'SUCCESS' if kernel_executed else 'FAILURE'}")
    
    # Print all debug values for inspection
    print("\nAll Debug Values:")
    for i in range(dbg.shape[0]):
        if abs(dbg[i]) > 0:
            print(f"  dbg[{i}] = {dbg[i]}")
    
    # Overall success based only on decomposition properties
    overall_success = q_orthogonal and r_upper and reconstruction_valid
    print(f"\nOverall test result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    return overall_success

# ----------------------------------------------------------------------------
# 5 · quick self-test                                                         #
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run detailed test with a short timeout
    try:
        # Use a short timeout to prevent hanging
        detailed_test_success = test_qr_detailed()
        print(f"Detailed test completed: {'SUCCESS' if detailed_test_success else 'FAILURE'}")
    except Exception as e:
        print(f"Detailed test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Run a simple test with a tiny matrix
    try:
        print("\n\nSimple Test with Tiny Matrix:")
        A = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        Q, R = qr(A, timeout_seconds=1.0)  # Set a short timeout
        print(f"Q shape: {Q.shape}")
        print(f"R shape: {R.shape}")
        print(f"Q matrix:\n{Q}")
        print(f"R matrix:\n{R}")
        
        # Check orthogonality and reconstruction
        ortho = mx.mean(mx.abs(mx.matmul(Q.T, Q) - mx.eye(2))).item()
        recon = mx.mean(mx.abs(mx.matmul(Q, R) - A)).item()
        print(f"2×2  ‖QᵀQ−I‖₁={ortho:9.2e}  ‖QR-A‖₁={recon:9.2e}")
        print("Simple test completed successfully")
    except Exception as e:
        print(f"Simple test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("done ✓")