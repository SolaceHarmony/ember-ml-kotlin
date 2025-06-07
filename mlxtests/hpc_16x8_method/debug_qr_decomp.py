#!/usr/bin/env python3
"""
Debug version of the QR decomposition with additional debug information.
This version adds more debug values and fixes potential issues in the Metal kernel.
"""
import time
from typing import Tuple, Optional
import mlx.core as mx

# ----------------------------------------------------------------------------
# 1 · Metal kernel with additional debug information                          #
# ----------------------------------------------------------------------------
_DEBUG_QR_SRC = r"""

/* ------------------------------------------------------------------ constants */
#define EPSILON     1e-10f
#define NUM_LIMBS   8u          // 128-bit accumulator (8 × 16-bit)
#define LIMB_RADIX  65536.0f    // 2¹⁶

kernel void debug_qr_kernel(
    device const float* A                  [[buffer(0)]],
    device const uint*  shape              [[buffer(1)]],
    device float*       Q_out              [[buffer(2)]],
    device float*       R_out              [[buffer(3)]],
    device float*       dbg                [[buffer(4)]],
    uint                tid                [[thread_position_in_threadgroup]],
    uint3               tpg                [[threads_per_threadgroup]],
    uint3               gpg                [[threadgroups_per_grid]])
{
    const uint m        = shape[0];
    const uint n        = shape[1];
    const uint min_dim  = (m < n ? m : n);
    const uint grid_sz  = gpg.x * tpg.x;    // Total number of threads in grid
    
    // Store thread info in debug array
    if (tid == 0) {
        dbg[0] = float(tid);
        dbg[1] = float(tpg.x);
        dbg[2] = float(gpg.x);
        dbg[3] = float(grid_sz);
    }

    /* 0 · initialise Q ← I,  R ← A (parallel over full grid) */
    for (uint idx = tid; idx < m * m; idx += grid_sz) {
        uint r = idx / m, c = idx % m;
        Q_out[idx] = (r == c ? 1.0f : 0.0f);
    }
    
    for (uint idx = tid; idx < m * n; idx += grid_sz)
        R_out[idx] = A[idx];

    threadgroup_barrier(mem_flags::mem_device);

    /* ===== single-thread numerical core (safe, still fast for ~2k²) ===== */
    if (tid == 0)
    {
        // Store initial values in debug array
        dbg[11] = R_out[0];  // First element of R
        dbg[12] = Q_out[0];  // First element of Q
        
        for (uint k = 0; k < min_dim; ++k)
        {
            /* -- column scaling (improves robustness) ------------------ */
            float cmax = 0.0f;
            for (uint i = k; i < m; ++i)
                cmax = fmax(cmax, fabs(R_out[i*n + k]));
            
            // Store column max in debug array
            if (k == 0) dbg[13] = cmax;
            
            float scale = (cmax > EPSILON ? 1.0f / cmax : 1.0f);
            for (uint i = k; i < m; ++i)
                R_out[i*n + k] *= scale;
            
            if (k == 0) dbg[10] = scale;

            /* -- build Householder v ----------------------------------- */
            float sigma = 0.0f;
            for (uint i = k; i < m; ++i) {
                float v = R_out[i*n + k];
                sigma += v*v;
            }
            if (k == 0) dbg[4] = sigma;
            
            float norm = sqrt(sigma);
            if (k == 0) dbg[5] = norm;
            
            if (norm < EPSILON) {                    // zero column
                R_out[k*n + k] /= scale;
                continue;
            }
            
            float sign = (R_out[k*n + k] >= 0.0f ? 1.0f : -1.0f);
            R_out[k*n + k] += sign * norm;          // v₀ update
            
            if (k == 0) dbg[14] = R_out[k*n + k];  // Store updated v₀

            /* -- limb-precision vᵀv ----------------------------------- */
            uint limb[NUM_LIMBS] = {0u};
            for (uint i = k; i < m; ++i)
            {
                uint bits = as_type<uint>(R_out[i*n + k]);
                ushort lo = bits & 0xFFFFu;
                ushort hi = (bits >> 16) & 0xFFFFu;
                uint p0 = uint(lo*lo);
                uint p1 = uint(hi*hi);
                uint pc = uint(lo*hi) << 1;

                limb[0] +=  p0 & 0xFFFFu;
                limb[1] += (p0 >> 16) + (pc & 0xFFFFu);
                limb[2] += (pc >> 16) + (p1 & 0xFFFFu);
                limb[3] +=  p1 >> 16;
            }
            for (uint l = 0; l < NUM_LIMBS-1; ++l) {    // carry propagation
                uint carry = limb[l] >> 16;
                limb[l]   &= 0xFFFFu;
                limb[l+1] += carry;
            }
            float vtv = 0.0f, radix = 1.0f;
            for (uint l = 0; l < NUM_LIMBS; ++l) {
                vtv  += float(limb[l]) * radix;
                radix *= LIMB_RADIX;
            }
            if (k == 0) dbg[6] = vtv;
            
            float inv_vtv = (vtv < EPSILON ? 0.0f : 1.0f / vtv);
            if (k == 0) dbg[7] = inv_vtv;

            /* -- reflect R (k … n-1) ---------------------------------- */
            for (uint j = k; j < n; ++j) {
                float dot = 0.0f;
                for (uint i = k; i < m; ++i)
                    dot += R_out[i*n + k] * R_out[i*n + j];
                
                if (j == k && k == 0) dbg[8] = dot;
                
                float beta = 2.0f * dot * inv_vtv;
                for (uint i = k; i < m; ++i)
                    R_out[i*n + j] -= beta * R_out[i*n + k];
            }

            /* -- reflect Q (0 … m-1) ---------------------------------- */
            for (uint j = 0; j < m; ++j) {
                float dot = 0.0f;
                for (uint i = k; i < m; ++i)
                    dot += R_out[i*n + k] * Q_out[i*m + j];
                
                if (j == 0 && k == 0) dbg[9] = dot;
                
                float beta = 2.0f * dot * inv_vtv;
                for (uint i = k; i < m; ++i)
                    Q_out[i*m + j] -= beta * R_out[i*n + k];
            }

            /* -- un-scale column k ------------------------------------ */
            for (uint i = k; i < m; ++i)
                R_out[i*n + k] /= scale;
        }

        /* -- force R upper-triangular ---------------------------------- */
        for (uint r = 1; r < m; ++r)
            for (uint c = 0; c < min(r, n); ++c)
                R_out[r*n + c] = 0.0f;

        dbg[15] = 1.0f;     // success flag (changed to 1.0 to indicate success)
    }
}
"""

# ----------------------------------------------------------------------------
# 2 · compile the kernel                                                      #
# ----------------------------------------------------------------------------
_DEBUG_QR_KERNEL = mx.fast.metal_kernel(
    name              = "debug_qr_kernel",
    source            = _DEBUG_QR_SRC,
    input_names       = ["A", "shape"],
    output_names      = ["Q_out", "R_out", "dbg"],
    ensure_row_contiguous=True
)

# ----------------------------------------------------------------------------
# 3 · python wrapper                                                          #
# ----------------------------------------------------------------------------
def debug_qr(A,
                       *,
                       debug: bool = False,
                       grid_size: tuple = (512, 1, 1),
                       thread_size: tuple = (512, 1, 1)
                      ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Debug version of QR decomposition with additional debug information.
    
    Args:
        A: Input matrix
        debug: Whether to return debug information
        grid_size: Grid size for Metal kernel
        thread_size: Thread size for Metal kernel
        
    Returns:
        (Q, R[, dbg])
    """
    A     = mx.array(A, dtype=mx.float32)
    m, n  = A.shape
    shape = mx.array([m, n], dtype=mx.uint32)
    dbg   = mx.zeros((16,), dtype=mx.float32)

    Q, R, dbg = _DEBUG_QR_KERNEL(
        inputs        = [A, shape],
        output_shapes = [(m, m), (m, n), (16,)],
        output_dtypes = [mx.float32, mx.float32, mx.float32],
        grid          = grid_size,
        threadgroup   = thread_size
    )
    
    # Check if Q or R contain all zeros
    q_zeros = mx.all(Q == 0.0).item()
    r_zeros = mx.all(R == 0.0).item()
    
    if q_zeros or r_zeros:
        print("WARNING: Zero matrices detected!")
        print(f"Q contains all zeros: {q_zeros}")
        print(f"R contains all zeros: {r_zeros}")
        print("Debug values:")
        for i, v in enumerate(dbg):
            print(f"dbg[{i}]: {v.item()}")
    
    return (Q, R, dbg) if debug else (Q, R)

# ----------------------------------------------------------------------------
# 4 · test function                                                           #
# ----------------------------------------------------------------------------
def test_debug_qr():
    """Test the debug QR implementation with various configurations."""
    print("=" * 80)
    print("DEBUG QR DECOMPOSITION TEST")
    print("=" * 80)
    
    # Test case 1: Small well-conditioned matrix
    print("\nTest Case 1: Small well-conditioned matrix (4x4)")
    A1 = mx.array([
        [4.0, 1.0, -2.0, 2.0],
        [1.0, 2.0, 0.0, 1.0],
        [-2.0, 0.0, 3.0, -2.0],
        [2.0, 1.0, -2.0, -1.0]
    ], dtype=mx.float32)
    
    # Test with different thread/grid configurations
    configurations = [
        ((256, 1, 1), (256, 1, 1), "Default (256x256)"),
        ((512, 1, 1), (512, 1, 1), "Large (512x512)"),
        ((128, 1, 1), (128, 1, 1), "Small (128x128)"),
        ((64, 1, 1), (64, 1, 1), "Very Small (64x64)"),
        ((32, 1, 1), (32, 1, 1), "Tiny (32x32)")
    ]
    
    for grid_size, thread_size, config_name in configurations:
        print(f"\nTesting {config_name} configuration...")
        print(f"Grid: {grid_size}, Thread: {thread_size}")
        
        t0 = time.time()
        Q, R, dbg = debug_qr(A1, debug=True, grid_size=grid_size, thread_size=thread_size)
        dt = time.time() - t0
        
        print(f"Execution time: {dt:.6f} seconds")
        
        # Interpret debug values
        print("\nDebug values:")
        print(f"Thread ID: {dbg[0]}")
        print(f"Threadgroup size: {dbg[1]}")
        print(f"Grid size: {dbg[2]}")
        print(f"Total threads: {dbg[3]}")
        print(f"Sigma: {dbg[4]}")
        print(f"Norm: {dbg[5]}")
        print(f"vᵀv: {dbg[6]}")
        print(f"inv_vtv: {dbg[7]}")
        print(f"R dot: {dbg[8]}")
        print(f"Q dot: {dbg[9]}")
        print(f"Scale: {dbg[10]}")
        print(f"Initial R[0]: {dbg[11]}")
        print(f"Initial Q[0]: {dbg[12]}")
        print(f"Column max: {dbg[13]}")
        print(f"Updated v₀: {dbg[14]}")
        print(f"Success flag: {dbg[15]}")
        
        # Check if matrices contain all zeros
        q_zeros = mx.all(Q == 0.0).item()
        r_zeros = mx.all(R == 0.0).item()
        print(f"\nQ contains all zeros: {q_zeros}")
        print(f"R contains all zeros: {r_zeros}")
        
        if not q_zeros:
            # Verify QR property: A ≈ Q·R
            reconstruct = mx.matmul(Q, R)
            error = mx.mean(mx.abs(reconstruct - A1)).item()
            print(f"Reconstruction error: {error}")
            
            # Verify orthogonality of Q
            identity = mx.eye(Q.shape[0])
            ortho_error = mx.mean(mx.abs(mx.matmul(Q.T, Q) - identity)).item()
            print(f"Orthogonality error: {ortho_error}")
    
    # Test case 2: Matrix with exact values from the example
    print("\nTest Case 2: Sample from the failed case")
    A2 = mx.array([
        [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
        [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
        [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
        [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
        [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
    ], dtype=mx.float32)
    
    # Test with the configuration that worked best for the small matrix
    best_config = None
    best_error = float('inf')
    
    for grid_size, thread_size, config_name in configurations:
        print(f"\nTesting {config_name} configuration on example matrix...")
        
        Q, R, dbg = debug_qr(A2, debug=True, grid_size=grid_size, thread_size=thread_size)
        
        # Check if matrices contain all zeros
        q_zeros = mx.all(Q == 0.0).item()
        r_zeros = mx.all(R == 0.0).item()
        print(f"Q contains all zeros: {q_zeros}")
        print(f"R contains all zeros: {r_zeros}")
        
        if not q_zeros:
            # Verify QR property: A ≈ Q·R
            reconstruct = mx.matmul(Q, R)
            error = mx.mean(mx.abs(reconstruct - A2)).item()
            print(f"Reconstruction error: {error}")
            
            # Verify orthogonality of Q
            identity = mx.eye(Q.shape[0])
            ortho_error = mx.mean(mx.abs(mx.matmul(Q.T, Q) - identity)).item()
            print(f"Orthogonality error: {ortho_error}")
            
            # Track best configuration
            if error < best_error:
                best_error = error
                best_config = (grid_size, thread_size, config_name)
    
    if best_config:
        print(f"\nBest configuration: {best_config[2]} with error {best_error}")
    else:
        print("\nNo successful configuration found")

if __name__ == "__main__":
    # Set environment variables to control Metal execution
    import os
    os.environ["MLX_USE_METAL"] = "1"
    print("Running debug QR decomposition tests...")
    test_debug_qr()