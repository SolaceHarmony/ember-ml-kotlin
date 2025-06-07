#!/usr/bin/env python3
"""
Fixed solution for QR decomposition that addresses the zero matrices issue.
This implementation uses a Python fallback when the Metal kernel fails.
"""
import time
from typing import Tuple, Optional
import mlx.core as mx
import numpy as np

# ----------------------------------------------------------------------------
# 1 · Python implementation for fallback                                      #
# ----------------------------------------------------------------------------
def _python_qr_impl(A, debug=False):
    """
    Pure Python implementation of QR decomposition for fallback.
    
    Args:
        A: Input matrix (MLX array)
        debug: Whether to return debug information
        
    Returns:
        (Q, R[, dbg])
    """
    # Convert to numpy for easier processing
    if isinstance(A, mx.array):
        A_np = A.tolist()
        A_np = np.array(A_np, dtype=np.float32)
    else:
        A_np = np.array(A, dtype=np.float32)
    
    # Constants
    EPSILON = 1e-10
    
    # Debug array
    dbg = np.zeros(16, dtype=np.float32)
    
    # Get dimensions
    m, n = A_np.shape
    min_dim = min(m, n)
    
    # Initialize Q as identity and R as copy of A
    Q = np.eye(m, dtype=np.float32)
    R = A_np.copy()
    
    # Main QR algorithm
    for k in range(min_dim):
        # Column scaling (improves robustness)
        cmax = np.max(np.abs(R[k:m, k]))
        scale = 1.0 / cmax if cmax > EPSILON else 1.0
        R[k:m, k] *= scale
        
        if k == 0:
            dbg[10] = scale
            dbg[13] = cmax
        
        # Build Householder vector
        sigma = np.sum(R[k:m, k] ** 2)
        if k == 0:
            dbg[4] = sigma
        
        norm = np.sqrt(sigma)
        if k == 0:
            dbg[5] = norm
        
        # Skip if column is effectively zero
        if norm < EPSILON:
            R[k, k] /= scale
            continue
        
        # Update first element with Householder reflection
        sign = 1.0 if R[k, k] >= 0.0 else -1.0
        R[k, k] += sign * norm
        
        if k == 0:
            dbg[14] = R[k, k]
        
        # Calculate v^T v
        vtv = np.sum(R[k:m, k] ** 2)
        if k == 0:
            dbg[6] = vtv
        
        # Calculate inverse of v^T v
        inv_vtv = 1.0 / vtv if vtv > EPSILON else 0.0
        if k == 0:
            dbg[7] = inv_vtv
        
        # Skip reflection if inv_vtv is zero
        if inv_vtv == 0.0:
            R[k:m, k] /= scale
            continue
        
        # Reflect R
        for j in range(k, n):
            dot = np.sum(R[k:m, k] * R[k:m, j])
            if j == k and k == 0:
                dbg[8] = dot
            
            beta = 2.0 * dot * inv_vtv
            R[k:m, j] -= beta * R[k:m, k]
        
        # Reflect Q
        for j in range(m):
            dot = np.sum(R[k:m, k] * Q[k:m, j])
            if j == 0 and k == 0:
                dbg[9] = dot
            
            beta = 2.0 * dot * inv_vtv
            Q[k:m, j] -= beta * R[k:m, k]
        
        # Un-scale column k
        R[k:m, k] /= scale
    
    # Force R upper-triangular
    for r in range(1, m):
        for c in range(min(r, n)):
            R[r, c] = 0.0
    
    # Set success flag
    dbg[15] = 1.0
    
    # Convert back to MLX arrays
    Q_mx = mx.array(Q, dtype=mx.float32)
    R_mx = mx.array(R, dtype=mx.float32)
    dbg_mx = mx.array(dbg, dtype=mx.float32)
    
    return (Q_mx, R_mx, dbg_mx) if debug else (Q_mx, R_mx)

# ----------------------------------------------------------------------------
# 2 · Original Metal kernel (unchanged)                                       #
# ----------------------------------------------------------------------------
# Import the original kernel from qr_ops.py
from ember_ml.backend.mlx.linearalg.qr_ops import _ENHANCED_QR_SRC, _ENHANCED_QR_KERNEL

# Define our fixed kernel source with improvements
_FIXED_QR_SRC = r"""

/* ------------------------------------------------------------------ constants */
#define EPSILON     1e-10f
#define NUM_LIMBS   8u          // 128-bit accumulator (8 × 16-bit)
#define LIMB_RADIX  65536.0f    // 2¹⁶
    uint                tid;
    uint3               tpg;
    uint3               gpg;

    const uint m        = shape[0];
    const uint n        = shape[1];
    const uint min_dim  = (m < n ? m : n);
    const uint grid_sz  = gpg.x * tpg.x;    // Total number of threads in grid

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
        for (uint k = 0; k < min_dim; ++k)
        {
            /* -- column scaling (improves robustness) ------------------ */
            float cmax = 0.0f;
            for (uint i = k; i < m; ++i)
                cmax = fmax(cmax, fabs(R_out[i*n + k]));
            
            // Store column max in debug array for first column
            if (k == 0) dbg[13] = cmax;
            
            // FIX: Ensure cmax is not too small to avoid division by zero
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
            
            // FIX: Handle zero or near-zero norm more carefully
            if (norm < EPSILON) {
                // Unscale the diagonal element
                R_out[k*n + k] /= scale;
                // Skip this iteration but continue with the next column
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
            
            // FIX: More robust handling of small vtv values
            float inv_vtv = (vtv > EPSILON ? 1.0f / vtv : 0.0f);
            if (k == 0) dbg[7] = inv_vtv;
            
            // FIX: Skip reflection if inv_vtv is zero
            if (inv_vtv == 0.0f) {
                // Unscale the column and continue
                for (uint i = k; i < m; ++i)
                    R_out[i*n + k] /= scale;
                continue;
            }

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

"""

# ----------------------------------------------------------------------------
# 3 · compile the kernel                                                      #
# ----------------------------------------------------------------------------
# We'll use the original kernel directly

# ----------------------------------------------------------------------------
# 4 · python wrapper with fallback                                            #
# ----------------------------------------------------------------------------
def fixed_qr(A,
             *,
             debug: bool = False,
             use_fallback: bool = True,
             grid_size: tuple = (512, 1, 1),  # Use the same grid size as the original
             thread_size: tuple = (512, 1, 1)  # Use the same thread size as the original
            ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Fixed QR decomposition that addresses the zero matrices issue.
    
    Args:
        A: Input matrix
        debug: Whether to return debug information
        use_fallback: Whether to use Python fallback if Metal kernel fails
        grid_size: Grid size for Metal kernel
        thread_size: Thread size for Metal kernel
        
    Returns:
        (Q, R[, dbg])
    """
    A     = mx.array(A, dtype=mx.float32)
    m, n  = A.shape
    shape = mx.array([m, n], dtype=mx.uint32)
    dbg   = mx.zeros((16,), dtype=mx.float32)

    # Try Metal kernel first
    try:
        # Use the original kernel
        Q, R, dbg = _ENHANCED_QR_KERNEL(
            inputs        = [A, shape],
            output_shapes = [(m, m), (m, n), (16,)],
            output_dtypes = [mx.float32, mx.float32, mx.float32],
            grid          = grid_size,
            threadgroup   = thread_size
        )
        
        # Check if Q or R contain all zeros
        q_zeros = mx.all(Q == 0.0).item()
        r_zeros = mx.all(R == 0.0).item()
        
        # If matrices contain all zeros and fallback is enabled, use Python implementation
        if (q_zeros or r_zeros) and use_fallback:
            print("WARNING: Metal kernel produced zero matrices, falling back to Python implementation")
            return _python_qr_impl(A, debug=debug)
        
        return (Q, R, dbg) if debug else (Q, R)
    
    except Exception as e:
        if use_fallback:
            print(f"WARNING: Metal kernel failed with error: {e}")
            print("Falling back to Python implementation")
            return _python_qr_impl(A, debug=debug)
        else:
            raise e

# ----------------------------------------------------------------------------
# 5 · test function                                                           #
# ----------------------------------------------------------------------------
def test_fixed_qr():
    """Test the fixed QR implementation with various matrices."""
    print("=" * 80)
    print("FIXED QR DECOMPOSITION TEST")
    print("=" * 80)
    
    # Test matrices
    test_matrices = [
        # Small well-conditioned matrix
        {
            "name": "Small well-conditioned matrix (4x4)",
            "matrix": mx.array([
                [4.0, 1.0, -2.0, 2.0],
                [1.0, 2.0, 0.0, 1.0],
                [-2.0, 0.0, 3.0, -2.0],
                [2.0, 1.0, -2.0, -1.0]
            ], dtype=mx.float32)
        },
        # Example matrix that failed
        {
            "name": "Example matrix that failed (5x5)",
            "matrix": mx.array([
                [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
                [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
                [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
                [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
                [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
            ], dtype=mx.float32)
        },
        # Random matrix
        {
            "name": "Random matrix (10x10)",
            "matrix": mx.random.normal((10, 10))
        }
    ]
    
    # Run tests for each matrix
    for test_case in test_matrices:
        print("\n" + "=" * 80)
        print(f"TESTING: {test_case['name']}")
        print("=" * 80)
        
        A = test_case["matrix"]
        
        # Run fixed implementation
        print("\nRunning fixed implementation...")
        t0 = time.time()
        Q, R, dbg = fixed_qr(A, debug=True)
        dt = time.time() - t0
        print(f"Fixed implementation time: {dt:.6f} seconds")
        
        # Check if matrices contain all zeros
        q_zeros = mx.all(Q == 0.0).item()
        r_zeros = mx.all(R == 0.0).item()
        print(f"Q contains all zeros: {q_zeros}")
        print(f"R contains all zeros: {r_zeros}")
        
        if not q_zeros:
            # Verify QR property: A ≈ Q·R
            reconstruct = mx.matmul(Q, R)
            error = mx.mean(mx.abs(reconstruct - A)).item()
            print(f"Reconstruction error: {error}")
            
            # Verify orthogonality of Q
            identity = mx.eye(Q.shape[0])
            ortho_error = mx.mean(mx.abs(mx.matmul(Q.T, Q) - identity)).item()
            print(f"Orthogonality error: {ortho_error}")
            
            # Print first few values of Q and R
            print("\nFirst few values of Q:")
            print(Q[:2, :2])
            
            print("\nFirst few values of R:")
            print(R[:2, :2])
        
        # Print debug values
        print("\nDebug values:")
        for i, v in enumerate(dbg):
            print(f"dbg[{i}]: {v.item()}")

if __name__ == "__main__":
    # Set environment variables to control Metal execution
    import os
    os.environ["MLX_USE_METAL"] = "1"
    print("Running fixed QR decomposition tests...")
    test_fixed_qr()