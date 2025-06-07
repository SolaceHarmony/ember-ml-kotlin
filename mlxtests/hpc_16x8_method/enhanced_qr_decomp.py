# ============================================================================
#  enhanced_hpc_qr.py   •   drop-in, Metal + MLX
# ============================================================================

from __future__ import annotations
import time
from typing import Tuple, Optional
import mlx.core as mx

# ----------------------------------------------------------------------------
# 1 · Metal kernel                                                            #
# ----------------------------------------------------------------------------
_ENHANCED_QR_SRC = r"""

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
            float scale = (cmax > EPSILON ? 1.0f / cmax : 1.0f);
            for (uint i = k; i < m; ++i)
                R_out[i*n + k] *= scale;
            dbg[10] = scale;

            /* -- build Householder v ----------------------------------- */
            float sigma = 0.0f;
            for (uint i = k; i < m; ++i) {
                float v = R_out[i*n + k];
                sigma += v*v;
            }
            dbg[4] = sigma;
            float norm = sqrt(sigma);
            dbg[5] = norm;
            if (norm < EPSILON) {                    // zero column
                R_out[k*n + k] /= scale;
                continue;
            }
            float sign = (R_out[k*n + k] >= 0.0f ? 1.0f : -1.0f);
            R_out[k*n + k] += sign * norm;          // v₀ update

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
            dbg[6] = vtv;
            float inv_vtv = (vtv < EPSILON ? 0.0f : 1.0f / vtv);
            dbg[7] = inv_vtv;

            /* -- reflect R (k … n-1) ---------------------------------- */
            for (uint j = k; j < n; ++j) {
                float dot = 0.0f;
                for (uint i = k; i < m; ++i)
                    dot += R_out[i*n + k] * R_out[i*n + j];
                if (j == k) dbg[8] = dot;
                float beta = 2.0f * dot * inv_vtv;
                for (uint i = k; i < m; ++i)
                    R_out[i*n + j] -= beta * R_out[i*n + k];
            }

            /* -- reflect Q (0 … m-1) ---------------------------------- */
            for (uint j = 0; j < m; ++j) {
                float dot = 0.0f;
                for (uint i = k; i < m; ++i)
                    dot += R_out[i*n + k] * Q_out[i*m + j];
                if (j == 0) dbg[9] = dot;
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

        dbg[15] = 0.0f;     // success flag
    }

"""

# ----------------------------------------------------------------------------
# 2 · compile the kernel                                                      #
# ----------------------------------------------------------------------------
_ENHANCED_QR_KERNEL = mx.fast.metal_kernel(
    name              = "enhanced_hpc_qr_kernel",
    source            = _ENHANCED_QR_SRC,
    input_names       = ["A", "shape"],
    output_names      = ["Q_out", "R_out", "dbg"],
    ensure_row_contiguous=True
)

# ----------------------------------------------------------------------------
# 3 · python wrapper                                                          #
# ----------------------------------------------------------------------------
def enhanced_tiled_qr(A,
                      *,
                      debug: bool = False
                     ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
    """
    Numerically-stable QR with limb accumulation.

    Returns (Q, R[, dbg]).
    """
    A     = mx.array(A, dtype=mx.float32)
    m, n  = A.shape
    shape = mx.array([m, n], dtype=mx.uint32)
    dbg   = mx.zeros((16,), dtype=mx.float32)

    Q, R, dbg = _ENHANCED_QR_KERNEL(
        inputs        = [A, shape],
        output_shapes = [(m, m), (m, n), (16,)],
        output_dtypes = [mx.float32, mx.float32, mx.float32],
        grid          = (512, 1, 1),
        threadgroup   = (512, 1, 1)
    )
    return (Q, R, dbg) if debug else (Q, R)

# ----------------------------------------------------------------------------
# 4 · quick self-test                                                         #
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Enhanced HPC-QR (fixed build)")
    print("="*72)
    for (m, n) in [(10, 10), (100, 150), (300, 300)]:
        A   = mx.random.normal((m, n))
        t0  = time.time()
        Q, R = enhanced_tiled_qr(A)
        dt  = time.time() - t0

        ortho = mx.mean(mx.abs(mx.matmul(Q.T, Q) - mx.eye(m))).item()
        recon = mx.mean(mx.abs(mx.matmul(Q, R) - A)).item()
        print(f"{m:4d}×{n:<4d}  ‖QᵀQ−I‖₁={ortho:9.2e}   "
              f"‖QR−A‖₁={recon:9.2e}   {dt:6.3f}s")
    print("done ✓")