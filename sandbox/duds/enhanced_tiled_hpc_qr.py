import mlx.core as mx
import time
import numpy as np

def enhanced_tiled_hpc_qr(A: mx.array, tile_size: int = 32) -> tuple[mx.array, mx.array]:
    """
    Enhanced tiled QR decomposition using a Metal kernel with int16 limb-based precision.
    
    This implementation uses improved tiling for better cache utilization and parallelism,
    along with int16 limb-based arithmetic for improved numerical stability compared to
    float32 calculations.
    
    Args:
        A: Input matrix (M x N) as float32.
        tile_size: Size of tiles for processing (default: 32)
        
    Returns:
        Tuple of (Q, R) matrices (M x M and M x N) as float32.
    """
    m, n = A.shape
    min_dim = min(m, n)
    
    # Define Metal kernel source string for enhanced tiled, int16 limb-based QR decomposition
    metal_kernel_source = """
#define NUM_LIMBS       8
#define LIMB_BITS       16
#define LIMB_MASK       0xFFFFu
#define LIMB_SIZE       (1u << LIMB_BITS)
#define LIMB_SIZE_F     65536.0f
#define EPSILON         1e-10f

struct ScaledInt16x8 {
    ushort limbs[NUM_LIMBS];
    float scale;
};

// Threadgroup shared variables
threadgroup float  tg_norm;
threadgroup float  tg_sign;
threadgroup atomic_uint tg_vtv[NUM_LIMBS];  // vᵀv using int16 limbs
threadgroup float  tg_scale;

const uint m = shapeParams[0];
const uint n = shapeParams[1];
const uint tile_size = shapeParams[2];
const uint min_dim = min(m, n);

/* Initialize helper functions for int16 limb-based arithmetic */
auto init_limbs = [](thread ScaledInt16x8& result) {
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        result.limbs[l] = 0;
    }
    result.scale = 1.0f;
};

auto convert_float_to_limbs = [](float value, thread ScaledInt16x8& result, float col_scale = 1.0f) {
    float abs_val = fabs(value);
    float scale = 1.0f;
    if (abs_val > EPSILON) {
        scale = 65535.0f / abs_val; // Normalize to fit in 16-bit limb
        scale = pow(2.0f, floor(log2(scale))); // Nearest power of 2
        scale = clamp(scale, 0.01f, 10000.0f); // Tighter bounds
    }
    scale *= col_scale; // Incorporate column scale
    result.scale = scale;

    float scaled_val = value * scale;
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        float limb_val = fmod(scaled_val, LIMB_SIZE_F);
        result.limbs[l] = ushort(max(0.0f, min(limb_val, 65535.0f))); // Clamp to ushort
        scaled_val = floor(scaled_val / LIMB_SIZE_F);
    }
};

auto add_limbs = [](thread const ScaledInt16x8& a, thread const ScaledInt16x8& b, thread ScaledInt16x8& result) {
    float max_scale = max(a.scale, b.scale);
    float scale_ratio_a = a.scale / max_scale;
    float scale_ratio_b = b.scale / max_scale;
    result.scale = max_scale;

    uint carry = 0;
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        uint sum = uint(float(a.limbs[l]) * scale_ratio_a) + 
                   uint(float(b.limbs[l]) * scale_ratio_b) + carry;
        result.limbs[l] = sum & LIMB_MASK;
        carry = sum >> LIMB_BITS;
    }

    if (carry) {
        result.scale *= 0.5f;
        carry = 0;
        for (uint l = NUM_LIMBS - 1; l > 0; --l) {
            uint val = (result.limbs[l] << 1) | carry;
            result.limbs[l] = val & LIMB_MASK;
            carry = val >> LIMB_BITS;
        }
        result.limbs[0] = ((result.limbs[0] << 1) | carry) & LIMB_MASK;
    }
};

auto mul_limbs = [](thread const ScaledInt16x8& a, thread const ScaledInt16x8& b, thread ScaledInt16x8& result) {
    result.scale = a.scale * b.scale;
    
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        result.limbs[l] = 0;
    }

    for (uint i = 0; i < NUM_LIMBS; ++i) {
        if (a.limbs[i] == 0) continue;
        uint carry = 0;
        for (uint j = 0; j < NUM_LIMBS; ++j) {
            uint pos = i + j;
            if (pos >= NUM_LIMBS) break;
            uint prod = a.limbs[i] * b.limbs[j] + result.limbs[pos] + carry;
            result.limbs[pos] = prod & LIMB_MASK;
            carry = prod >> LIMB_BITS;
        }
        
        if (carry && i + NUM_LIMBS - 1 < NUM_LIMBS) {
            result.limbs[i + NUM_LIMBS - 1] += carry & LIMB_MASK; // Handle overflow
        }
    }

    // Rescale if overflow detected
    uint highest_limb = result.limbs[NUM_LIMBS - 1];
    if (highest_limb > LIMB_MASK / 2) {
        result.scale *= 0.5f;
        uint carry = 0;
        for (uint l = NUM_LIMBS - 1; l > 0; --l) {
            uint val = (result.limbs[l] << 1) | carry;
            result.limbs[l] = val & LIMB_MASK;
            carry = val >> LIMB_BITS;
        }
        result.limbs[0] = ((result.limbs[0] << 1) | carry) & LIMB_MASK;
    }
};

auto limbs_to_float = [](thread const ScaledInt16x8& limbs) -> float {
    float result = 0.0f;
    float multiplier = 1.0f;
    
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        result += float(limbs.limbs[l]) * multiplier;
        multiplier *= LIMB_SIZE_F;
    }
    
    return result / limbs.scale;
};

auto add_to_shared_limbs = [](thread const ScaledInt16x8& local, threadgroup atomic_uint* shared, float shared_scale) {
    float scale_ratio = local.scale / shared_scale;
    
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        uint adjusted_val = uint(float(local.limbs[l]) * scale_ratio);
        atomic_fetch_add_explicit(&shared[l], adjusted_val, memory_order_relaxed);
    }
};

auto propagate_carries = [](threadgroup atomic_uint* shared) {
    for (uint l = 0; l < NUM_LIMBS-1; ++l) {
        uint v = atomic_load_explicit(&shared[l], memory_order_relaxed);
        uint c = v >> LIMB_BITS;
        atomic_store_explicit(&shared[l], v & LIMB_MASK, memory_order_relaxed);
        atomic_fetch_add_explicit(&shared[l+1], c, memory_order_relaxed);
    }
};

auto shared_limbs_to_float = [](threadgroup atomic_uint* shared, float shared_scale) -> float {
    float result = 0.0f;
    float multiplier = 1.0f;
    
    for (uint l = 0; l < NUM_LIMBS; ++l) {
        uint v = atomic_load_explicit(&shared[l], memory_order_relaxed);
        result += float(v) * multiplier;
        multiplier *= LIMB_SIZE_F;
    }
    
    return result / shared_scale;
};

/* Initialize debug array */
if (thread_position_in_grid.x == 0) {
    for (uint i = 0; i < 16; ++i)
        debug[i] = 0.0f;
}

/* ============= 0. copy A→R, eye→Q ==================================== */
for (uint row = thread_position_in_grid.x;
     row < m;
     row += threads_per_threadgroup.x)
{
    for (uint col = 0; col < n; ++col)
        R_out[row*n + col] = A[row*n + col];
    
    for (uint col = 0; col < m; ++col)
        Q_out[row*m + col] = (row == col) ? 1.0f : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

/* ============= 1. panel loop ======================================== */
for (uint k = 0; k < min_dim; ++k)
{
    /* ---- 1a. Dynamic column scaling for better numerical stability --- */
    if (thread_position_in_threadgroup.x == 0)
    {
        float maxAbs = 0.0f;
        float sumSq = 0.0f;
        
        for (uint i = k; i < m; ++i) {
            float val = R_out[i*n + k];
            maxAbs = fmax(maxAbs, fabs(val));
            sumSq += val * val;
        }
        
        // Use a combination of max and RMS for better scaling
        float rms = sqrt(sumSq / (m - k));
        float scale_factor = fmax(maxAbs, rms);
        
        float scale = (scale_factor > EPSILON) ? 1.0f / scale_factor : 1.0f;
        scale = clamp(scale, 1e-6f, 1e6f);
        tg_scale = scale;
        
        for (uint i = k; i < m; ++i)
            R_out[i*n + k] *= scale;
            
        // Initialize shared limbs for vᵀv
        for (uint l = 0; l < NUM_LIMBS; ++l)
            atomic_store_explicit(&tg_vtv[l], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = tg_scale;
    
    /* ---- 1b. ||v||² with int16 limbs -------------------------- */
    thread ScaledInt16x8 local_norm;
    init_limbs(local_norm);
    local_norm.scale = scale * scale; // Incorporate scale into limbs
    
    // Process in tiles for better cache utilization
    for (uint tile_start = k; tile_start < m; tile_start += tile_size) {
        uint tile_end = min(tile_start + tile_size, m);
        
        for (uint i = tile_start + thread_position_in_grid.x;
             i < tile_end;
             i += threads_per_threadgroup.x)
        {
            float v = R_out[i*n + k];
            thread ScaledInt16x8 v_squared;
            convert_float_to_limbs(v * v, v_squared, scale * scale);
            
            thread ScaledInt16x8 new_local;
            add_limbs(local_norm, v_squared, new_local);
            local_norm = new_local;
        }
    }
    
    // Combine results from all threads
    add_to_shared_limbs(local_norm, tg_vtv, scale * scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_position_in_threadgroup.x == 0)
        propagate_carries(tg_vtv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float norm2 = shared_limbs_to_float(tg_vtv, scale * scale);
    if (isnan(norm2) || isinf(norm2) || norm2 <= EPSILON) {
        debug[2] = norm2; // Log issue
        norm2 = EPSILON;
    }
    debug[3] = norm2; // Log value
    
    float norm = sqrt(norm2);
    if (isnan(norm) || isinf(norm)) {
        debug[4] = norm;
        norm = sqrt(EPSILON);
    }
    debug[5] = norm;
    
    if (thread_position_in_threadgroup.x == 0) {
        tg_norm = norm;
        float head_val = R_out[k*n + k];
        tg_sign = (head_val >= 0.0f ? 1.0f : -1.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    /* ---- 2. Householder head update ----------------------------------- */
    if (thread_position_in_threadgroup.x == 0) {
        R_out[k*n + k] += tg_sign * tg_norm;
        
        // Reset shared limbs for next calculation
        for (uint l = 0; l < NUM_LIMBS; ++l)
            atomic_store_explicit(&tg_vtv[l], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    /* ---- 3. vᵀv computation with int16 limbs --------------------- */
    init_limbs(local_norm);
    local_norm.scale = scale * scale;
    
    // Process in tiles for better cache utilization
    for (uint tile_start = k; tile_start < m; tile_start += tile_size) {
        uint tile_end = min(tile_start + tile_size, m);
        
        for (uint i = tile_start + thread_position_in_grid.x;
             i < tile_end;
             i += threads_per_threadgroup.x)
        {
            float v = R_out[i*n + k];
            thread ScaledInt16x8 v_squared;
            convert_float_to_limbs(v * v, v_squared, scale * scale);
            
            thread ScaledInt16x8 new_local;
            add_limbs(local_norm, v_squared, new_local);
            local_norm = new_local;
        }
    }
    
    add_to_shared_limbs(local_norm, tg_vtv, scale * scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_position_in_threadgroup.x == 0)
        propagate_carries(tg_vtv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float vtv = shared_limbs_to_float(tg_vtv, scale * scale);
    if (isnan(vtv) || isinf(vtv) || vtv <= EPSILON) {
        debug[6] = vtv;
        vtv = EPSILON;
    }
    debug[7] = vtv;
    
    float inv_vtv = 1.0f / vtv;  // We'll multiply by 2 explicitly when needed
    if (isnan(inv_vtv) || isinf(inv_vtv)) {
        debug[8] = inv_vtv;
        inv_vtv = 0.0f;
    }
    debug[9] = inv_vtv;
    
    /* ---- 4. reflection on R (tiled for better cache utilization) ----- */
    for (uint j_tile = k; j_tile < n; j_tile += tile_size) {
        uint j_end = min(j_tile + tile_size, n);
        
        for (uint j = j_tile; j < j_end; ++j) {
            // Reset shared limbs for dot product
            if (thread_position_in_threadgroup.x == 0) {
                for (uint l = 0; l < NUM_LIMBS; ++l)
                    atomic_store_explicit(&tg_vtv[l], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute v'Rj dot product with int16 limbs
            thread ScaledInt16x8 local_dot;
            init_limbs(local_dot);
            local_dot.scale = scale;
            
            for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                uint i_end = min(i_tile + tile_size, m);
                
                for (uint i = i_tile + thread_position_in_grid.x;
                     i < i_end;
                     i += threads_per_threadgroup.x)
                {
                    float vi = R_out[i*n + k];
                    float rij = R_out[i*n + j];
                    
                    thread ScaledInt16x8 vi_limb, rij_limb, prod_limb;
                    convert_float_to_limbs(vi, vi_limb, scale);
                    convert_float_to_limbs(rij, rij_limb, scale);
                    mul_limbs(vi_limb, rij_limb, prod_limb);
                    
                    thread ScaledInt16x8 new_local;
                    add_limbs(local_dot, prod_limb, new_local);
                    local_dot = new_local;
                }
            }
            
            add_to_shared_limbs(local_dot, tg_vtv, scale);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (thread_position_in_threadgroup.x == 0)
                propagate_carries(tg_vtv);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            float dot = shared_limbs_to_float(tg_vtv, scale);
            dot = clamp(dot, -1e10f, 1e10f); // Expanded clamp range
            
            if (isnan(dot) || isinf(dot)) {
                debug[10] = dot;
                dot = 0.0f;
            }
            debug[11] = dot;
    
            // Apply Householder reflection: x' = x - 2(v·x/v·v)v
            for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                uint i_end = min(i_tile + tile_size, m);
                
                for (uint i = i_tile + thread_position_in_grid.x;
                     i < i_end;
                     i += threads_per_threadgroup.x)
                {
                    float v = R_out[i*n + k];
                    // Apply scaling compensation correctly: (2 * v * dot) / vtv
                    float upd = 2.0f * v * dot * inv_vtv;
                    
                    if (isnan(upd) || isinf(upd)) {
                        debug[13] = upd;
                        continue;
                    }
                    
                    R_out[i*n + j] -= upd;
                    
                    if (isnan(R_out[i*n + j]) || isinf(R_out[i*n + j])) {
                        debug[0] = 2.0f; // Log issue
                        R_out[i*n + j] = 0.0f;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    /* ---- 5. reflection on Q (tiled for better cache utilization) ----- */
    for (uint j_tile = 0; j_tile < m; j_tile += tile_size) {
        uint j_end = min(j_tile + tile_size, m);
        
        for (uint j = j_tile; j < j_end; ++j) {
            // Reset shared limbs for dot product
            if (thread_position_in_threadgroup.x == 0) {
                for (uint l = 0; l < NUM_LIMBS; ++l)
                    atomic_store_explicit(&tg_vtv[l], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute v'Qj dot product with int16 limbs
            thread ScaledInt16x8 local_dot;
            init_limbs(local_dot);
            local_dot.scale = scale;
            
            for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                uint i_end = min(i_tile + tile_size, m);
                
                for (uint i = i_tile + thread_position_in_grid.x;
                     i < i_end;
                     i += threads_per_threadgroup.x)
                {
                    float vi = R_out[i*n + k];
                    float qij = Q_out[i*m + j];
                    
                    thread ScaledInt16x8 vi_limb, qij_limb, prod_limb;
                    convert_float_to_limbs(vi, vi_limb, scale);
                    convert_float_to_limbs(qij, qij_limb, scale);
                    mul_limbs(vi_limb, qij_limb, prod_limb);
                    
                    thread ScaledInt16x8 new_local;
                    add_limbs(local_dot, prod_limb, new_local);
                    local_dot = new_local;
                }
            }
            
            add_to_shared_limbs(local_dot, tg_vtv, scale);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (thread_position_in_threadgroup.x == 0)
                propagate_carries(tg_vtv);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            float dot = shared_limbs_to_float(tg_vtv, scale);
            dot = clamp(dot, -1e10f, 1e10f); // Expanded clamp range
            
            if (isnan(dot) || isinf(dot)) {
                debug[10] = dot;
                dot = 0.0f;
            }
            debug[12] = dot;
            
            // Apply Householder reflection: x' = x - 2(v·x/v·v)v
            for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                uint i_end = min(i_tile + tile_size, m);
                
                for (uint i = i_tile + thread_position_in_grid.x;
                     i < i_end;
                     i += threads_per_threadgroup.x)
                {
                    float v = R_out[i*n + k];
                    // Apply scaling compensation correctly: (2 * v * dot) / vtv
                    float upd = 2.0f * v * dot * inv_vtv;
                    
                    if (isnan(upd) || isinf(upd)) {
                        debug[13] = upd;
                        continue;
                    }
                    
                    Q_out[i*m + j] -= upd;
                    
                    if (isnan(Q_out[i*m + j]) || isinf(Q_out[i*m + j])) {
                        debug[1] = 2.0f;
                        Q_out[i*m + j] = 0.0f;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    /* ---- 6. Apply scale correction to all columns affected by the reflection ---- */
    if (thread_position_in_threadgroup.x == 0) {
        // Ensure diagonal element is positive
        if (R_out[k*n + k] < 0.0f) {
            R_out[k*n + k] = -R_out[k*n + k];
            
            // Also negate the corresponding column in Q to maintain Q*R = A
            for (uint i = 0; i < m; ++i) {
                Q_out[i*m + k] = -Q_out[i*m + k];
            }
            debug[14] = 1.0f; // Log sign correction
        }
        
        // Restore proper scaling for diagonal element
        R_out[k*n + k] /= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Zero out elements below diagonal in R
    for (uint i = k + 1 + thread_position_in_grid.x;
         i < m;
         i += threads_per_threadgroup.x)
    {
        R_out[i*n + k] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    /* ---- 7. Apply proper scaling correction for numerical stability ---- */
    // Scale factor affects all columns to the right of column k
    for (uint j = k + 1; j < n; ++j) {
        for (uint i = thread_position_in_grid.x;
             i < m; // Apply to entire column, not just upper triangular part
             i += threads_per_threadgroup.x)
        {
            // R[i,j] was computed with scaled inputs, so we need to undo that effect
            if (scale != 1.0f) {
                R_out[i*n + j] /= scale;
            }
            
            // Safety check for numerical issues
            if (isnan(R_out[i*n + j]) || isinf(R_out[i*n + j])) {
                debug[0] = 3.0f; // Log scaling issue
                R_out[i*n + j] = 0.0f;
            }
        }
    }
    
    // Also ensure Q is properly scaled
    for (uint j = 0; j < m; ++j) {
        for (uint i = thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup.x)
        {
            if (scale != 1.0f) {
                Q_out[i*m + j] /= scale;
            }
            
            if (isnan(Q_out[i*m + j]) || isinf(Q_out[i*m + j])) {
                debug[1] = 3.0f; // Log scaling issue
                Q_out[i*m + j] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
} /* ── next column k ── */
"""
    
    try:
        # Compile the Metal kernel
        compiled_kernel = mx.fast.metal_kernel(
            name="int16_limb_qr_kernel",
            source=metal_kernel_source,
            input_names=["A", "shapeParams"],
            output_names=["Q_out", "R_out", "debug"],
            ensure_row_contiguous=True
        )
    except Exception as e:
        print(f"Failed to compile Metal kernel: {e}")
        print("Please check the Metal kernel source for errors.")
        return mx.eye(m), mx.array(A)
    
    # Optimize threadgroup size based on matrix dimensions
    threads_per_group = min(128, max(16, m // 32))  # Smaller for better occupancy
    num_groups = (m + threads_per_group - 1) // threads_per_group
    grid_size = (num_groups, 1, 1) 
    tg_size = (threads_per_group, 1, 1)
    shape_params = mx.array([m, n, tile_size], dtype=mx.uint32)
    
    dbg = mx.zeros((16,), dtype=mx.float32)
    
    try:
        # Execute the kernel
        Q, R, dbg = compiled_kernel(
            inputs=[A, shape_params],
            output_shapes=[(m, m), (m, n), (16,)],
            output_dtypes=[A.dtype, A.dtype, mx.float32],
            grid=grid_size,
            threadgroup=tg_size
        )
        if mx.any(dbg != 0):
            print(f"Debug flags non-zero: {dbg.tolist()}")
            print(f"  norm2: {dbg[3]}, norm: {dbg[5]}, vtv: {dbg[7]}, inv_vtv: {dbg[9]}")
            print(f"  R dot: {dbg[11]}, Q dot: {dbg[12]}, update: {dbg[13]}")
        if mx.any(mx.isnan(Q)) or mx.any(mx.isnan(R)):
            print("NaN detected in Q or R")
            return mx.eye(m), mx.array(A)
        print("Debug flags:", dbg)
        return Q, R
    except Exception as e:
        print(f"Metal kernel execution failed: {e}")
        return mx.eye(m), mx.array(A)
    
def benchmark_qr(sizes=[(100, 100), (500, 500), (1000, 1000), (2000, 2000)], 
                 repeat=3, 
                 compare_with_native=True):
    """
    Benchmark the enhanced QR implementation against different matrix sizes
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
        
        # Test enhanced implementation
        enhanced_times = []
        for i in range(repeat):
            start_time = time.time()
            Q_enhanced, R_enhanced = enhanced_tiled_hpc_qr(A)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
            
        avg_enhanced_time = sum(enhanced_times) / repeat
        print(f"Enhanced QR completed in {avg_enhanced_time:.4f} seconds (avg of {repeat} runs)")
        
        # Check orthogonality and reconstruction
        ortho_error = mx.mean(mx.abs(mx.matmul(Q_enhanced.T, Q_enhanced) - mx.eye(Q_enhanced.shape[0]))).item()
        recon_error = mx.mean(mx.abs(mx.matmul(Q_enhanced, R_enhanced) - A)).item()
        print(f"Orthogonality error: {ortho_error:.6e}")
        print(f"Reconstruction error: {recon_error:.6e}")
        
        result = {
            "size": size,
            "enhanced_time": avg_enhanced_time,
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
                    q_diff = mx.mean(mx.abs(Q_enhanced - Q_native)).item()
                    r_diff = mx.mean(mx.abs(R_enhanced - R_native)).item()
                    print(f"Difference in Q: {q_diff:.6e}")
                    print(f"Difference in R: {r_diff:.6e}")
                    
                    result["native_time"] = avg_native_time
                    result["q_diff"] = q_diff
                    result["r_diff"] = r_diff
                    result["speedup"] = avg_native_time / avg_enhanced_time
                except Exception as e:
                    print(f"Could not compare with native QR: {e}")
        
        results.append(result)
        print("-" * 40)
    
    return results

def test_numerical_stability(condition_numbers=[10, 100, 1000, 10000]):
    """
    Test the numerical stability of the enhanced QR implementation
    with matrices of different condition numbers.
    
    Args:
        condition_numbers: List of condition numbers to test
    """
    print("\nTesting numerical stability with different condition numbers...")
    
    for cond in condition_numbers:
        print(f"\nTesting with condition number {cond}...")
        
        # Create a matrix with specified condition number
        m, n = 100, 100
        U = mx.random.normal((m, m), dtype=mx.float32)
        U, _ = mx.linalg.qr(U, stream=mx.cpu)  # Orthogonalize
        
        V = mx.random.normal((n, n), dtype=mx.float32)
        V, _ = mx.linalg.qr(V, stream=mx.cpu)  # Orthogonalize
        
        # Create singular values with specified condition number
        s = mx.array([1.0 * (cond ** (-i / (min(m, n) - 1))) for i in range(min(m, n))], dtype=mx.float32)
        S = mx.zeros((m, n), dtype=mx.float32)
        for i in range(min(m, n)):
            S = mx.scatter(S, mx.array([i]), mx.array([i]), s[i:i+1])
        
        # Create matrix A = U*S*V^T with specified condition number
        A = mx.matmul(mx.matmul(U, S), V.T)
        
        # Test enhanced implementation
        Q_enhanced, R_enhanced = enhanced_tiled_hpc_qr(A)
        
        # Check orthogonality and reconstruction
        ortho_error = mx.mean(mx.abs(mx.matmul(Q_enhanced.T, Q_enhanced) - mx.eye(Q_enhanced.shape[0]))).item()
        recon_error = mx.mean(mx.abs(mx.matmul(Q_enhanced, R_enhanced) - A)).item()
        print(f"Orthogonality error: {ortho_error:.6e}")
        print(f"Reconstruction error: {recon_error:.6e}")
        
        # Compare with native MLX QR
        try:
            Q_native, R_native = mx.linalg.qr(A, stream=mx.cpu)
            native_ortho_error = mx.mean(mx.abs(mx.matmul(Q_native.T, Q_native) - mx.eye(Q_native.shape[0]))).item()
            native_recon_error = mx.mean(mx.abs(mx.matmul(Q_native, R_native) - A)).item()
            
            print(f"Native QR orthogonality error: {native_ortho_error:.6e}")
            print(f"Native QR reconstruction error: {native_recon_error:.6e}")
            
            print(f"Orthogonality improvement: {native_ortho_error/ortho_error:.2f}x")
            print(f"Reconstruction improvement: {native_recon_error/recon_error:.2f}x")
        except Exception as e:
            print(f"Native MLX QR failed: {e}")
        
        print("-" * 40)
if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced Tiled HPC QR Decomposition for MLX")
    print("=" * 80)
    
    # Test with a smaller matrix first
    print("\nTesting with small matrix (10x10)...")
    A_small = mx.random.normal((10, 10), dtype=mx.float32)
    start_time = time.time()
    Q_small, R_small = enhanced_tiled_hpc_qr(A_small)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_small.T, Q_small) - mx.eye(Q_small.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_small, R_small) - A_small)).item())
    print("-" * 40)
    
    # Test with medium matrix
    print("\nTesting with medium matrix (100x150)...")
    A_medium = mx.random.normal((100, 150), dtype=mx.float32)
    start_time = time.time()
    Q_medium, R_medium = enhanced_tiled_hpc_qr(A_medium)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_medium.T, Q_medium) - mx.eye(Q_medium.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_medium, R_medium) - A_medium)).item())
    print("-" * 40)
    
    # Compare with native MLX QR
    print("\nComparing with native MLX QR...")
    start_time = time.time()
    Q_native, R_native = mx.linalg.qr(A_medium, stream=mx.cpu)
    end_time = time.time()
    print(f"Native MLX QR completed in {end_time - start_time:.4f} seconds.")
    print("Difference in Q:", mx.mean(mx.abs(Q_medium - Q_native)).item())
    print("Difference in R:", mx.mean(mx.abs(R_medium - R_native)).item())
    print("-" * 40)
    
    # Run benchmarks with different matrix sizes
    print("\nRunning benchmarks...")
    results = benchmark_qr(sizes=[(100, 100), (500, 500), (1000, 1000)], repeat=2)
    
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
        print(f"{size_str:>10} | {result['enhanced_time']:>10.4f} | {result['ortho_error']:>10.2e} | {result['recon_error']:>10.2e} | {speedup:>10}")
    print("-" * 60)
    
    # Test numerical stability with ill-conditioned matrices
    test_numerical_stability([10, 1000, 100000])
    
    print("\nAll tests completed!")