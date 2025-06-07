import mlx.core as mx
import math
import uuid
from typing import Optional, Union, Literal, Tuple

# Helper functions
def next_pow2(x):
    return 1 << (x - 1).bit_length()

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "shape"],
        output_names=["scratch"],
        ensure_row_contiguous=True
    )

# Metal kernel sources
_PANEL_SRC = """
#define TG_SIZE   64u
#define LIMBS     4u
#define LIMB_RADIX 4294967296.0f


    const uint tid = tidXYZ.x;
    const uint ltid = ltidXYZ.x;
    const uint m = shape[0], n = shape[1];
    const uint k = shape[2], panel = shape[3];

    device uint* colA = A;
    device uint* colV = A + m*n;
    device uint* tauBuf = colV + m*k;
    device uint* pivBuf = tauBuf + k;

    if (tid >= panel || tid + shape[4] >= k) return;

    threadgroup uint sh[LIMBS][TG_SIZE];
    threadgroup float shf[TG_SIZE];
    threadgroup float tg_inv;

    const uint col = tid + gsz.x * panel;

    // Copy A to V
    for (uint r = ltid; r < m; r += TG_SIZE)
        colV[r*k + col] = colA[r*n + col];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute norm (128-bit via limbs + FP32 fast path)
    uint loc[LIMBS] = {0};
    float fp32 = 0;
    for (uint r = ltid; r < m; r += TG_SIZE) {
        float v = as_type<float>(colV[r*k + col]);
        fp32 = fma(v, v, fp32);
        ulong p = ulong(as_type<uint>(v)) * ulong(as_type<uint>(v));
        loc[0] += uint(p);
        uint c = p >> 32;
        for (uint i = 1; i < LIMBS; ++i) {
            uint t = loc[i] + c;
            c = (t < loc[i]);
            loc[i] = t;
            if (!c) break;
        }
    }
    for (uint i = 0; i < LIMBS; ++i) sh[i][ltid] = loc[i];
    shf[ltid] = fp32;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD reduce fp32 for pivot selection
    float simdf = shf[ltid];
    for (uint off = TG_SIZE >> 1; off; off >>= 1) {
        simdf += (ltid < off) ? shf[ltid + off] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ltid < off) shf[ltid] = simdf;
    }
    if (ltid == 0) shf[0] = simdf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce limbs
    for (uint l = 0; l < LIMBS; ++l) {
        for (uint off = TG_SIZE >> 1; off; off >>= 1) {
            if (ltid < off) sh[l][ltid] += sh[l][ltid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    float norm = sqrt(max(({
        float acc = 0, sc = 1;
        for (uint i = 0; i < LIMBS; ++i) {
            acc += float(sh[i][0]) * sc;
            sc *= LIMB_RADIX;
        }
        acc;
    }), 1.0e-18f));

    if (ltid == 0) {
        tg_inv = 1.0f / norm;
        tauBuf[col] = as_type<uint>(norm);
        pivBuf[col] = col;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize v
    float inv = tg_inv;
    for (uint r = ltid; r < m; r += TG_SIZE) {
        float v = as_type<float>(colV[r*k + col]) * inv;
        colV[r*k + col] = as_type<uint>(v);
    }
"""

_APPLY_SRC = """

#define BLK 128u
#define SIMD_SIZE 8u

    device uint* A [[buffer(0)]],
    device uint* shape [[buffer(1)]],
    uint3 g [[thread_position_in_grid]],
    uint3 l [[thread_position_in_threadgroup]],
    uint3 sgid [[thread_position_in_simdgroup]],
    uint simd_lane [[simd_lane_id]])
{
    const uint m = shape[0], n = shape[1], k = shape[2], panel = shape[3];
    const uint blk_i = g.y, blk_j = g.x;
    const uint row0 = blk_i * BLK + l.y * SIMD_SIZE + sgid.y;
    const uint col0 = blk_j * BLK + l.x * SIMD_SIZE + sgid.x + panel;

    if (row0 >= m || col0 >= n) return;

    threadgroup float v_cache[BLK][SIMD_SIZE];
    threadgroup float tau_cache[SIMD_SIZE];

    if (l.y == 0 && sgid.y == 0) {
        for (uint p = sgid.x; p < panel; p += SIMD_SIZE) {
            tau_cache[p % SIMD_SIZE] = as_type<float>(A[m*n + m*k + p]);
        }
    }

    for (uint p = l.x; p < panel; p += SIMD_SIZE) {
        if (sgid.x < SIMD_SIZE) {
            v_cache[l.y * SIMD_SIZE + sgid.y][p % SIMD_SIZE] = as_type<float>(A[row0*k + p]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint p = 0; p < panel; p += SIMD_SIZE) {
        float v[SIMD_SIZE];
        float tau[SIMD_SIZE];
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            v[i] = v_cache[l.y * SIMD_SIZE + sgid.y][i];
            tau[i] = tau_cache[i];
        }

        float a = as_type<float>(A[row0*n + col0]);
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            acc += v[i] * a * tau[i];
        }
    }

    float newA = as_type<float>(A[row0*n + col0]) - 2.0f * acc;
    A[row0*n + col0] = as_type<uint>(newA);
"""

_BUILDQ_SRC = """
#define TG 32u

kernel void build_q_qrp128(
    device uint* A [[buffer(0)]],
    device uint* shape [[buffer(1)]],
    uint3 gsz [[grid_size]],
    uint3 tid [[thread_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
    const uint m = shape[0], n = shape[1], k = shape[2];
    if (tid.x >= k || tid.y >= m) return;

    float q = (tid.x == tid.y) ? 1.0f : 0.0f;
    for (int p = k - 1; p >= 0; --p) {
        float v = as_type<float>(A[tid.y*k + p]);
        float tau = as_type<float>(A[m*n + m*k + p]);
        q -= 2.0f * tau * v * q;
    }
    A[m*n + tid.y*k + tid.x] = as_type<uint>(q);
"""

# Compile kernels
panelK = _compile(_PANEL_SRC, "panel_factor_qrp128")
applyK = _compile(_APPLY_SRC, "apply_update_qrp128")
buildK = _compile(_BUILDQ_SRC, "build_q_qrp128")

# QR decomposition driver
def qr128_qrp(A: mx.array, want_q: bool = False):
    """
    Compute the QR decomposition of matrix A using MLX and Metal acceleration.
    
    Args:
        A: Input matrix of shape (m, n) with dtype float32
        want_q: Whether to explicitly form the Q matrix
        
    Returns:
        Q: Orthogonal matrix (if want_q=True, otherwise None)
        R: Upper triangular matrix
        piv: Pivot indices
    """
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    m, n = map(int, A.shape)
    k = min(m, n)
    panel = 64
    limbs = 4
    scratch_cols = n + k + k + k  # A | V | τ | piv
    S = mx.zeros((m, scratch_cols), dtype=mx.uint32)
    S[:, :n] = A.astype(mx.uint32)  # Safe conversion

    shape = mx.array([m, n, k, panel, limbs], dtype=mx.uint32)

    # P₀: Panel factorization
    for col0 in range(0, k, panel):
        grid = (min(panel, k - col0), 1, 1)
        panelK(inputs=[S, shape], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
               grid=grid, threadgroup=(panel, 1, 1))

        # P₁: Trailing update
        right0 = col0 + panel
        if right0 < n:
            blocks = (math.ceil((n - right0) / 128), math.ceil((m - col0) / 128), 1)
            applyK(inputs=[S, shape], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
                   grid=blocks, threadgroup=(8, 8, 1))

    # P₂: Build explicit Q
    if want_q:
        buildK(inputs=[S, shape], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
               grid=(math.ceil(m / 32), math.ceil(k / 32), 1), threadgroup=(32, 1, 1))
        Q = S[:, n:n+k].view(dtype=A.dtype)
    else:
        Q = None

    R = S[:, :n].view(dtype=A.dtype)
    piv = S[0, -k:].view(dtype=mx.int32)
    return Q, R, piv

# Test and validate
if __name__ == "__main__":
    # Generate random test matrix
    A = mx.random.normal((512, 512), dtype=mx.float32)
    Q, R, piv = qr128_qrp(A, want_q=True)
    
    # Validate reconstruction
    # Create permutation matrix based on pivot indices
    n = A.shape[1]
    P = mx.zeros((n, n), dtype=A.dtype)
    for i, p in enumerate(piv.tolist()):
        P[i, p] = 1.0
    
    # Apply permutation to R
    R_perm = mx.matmul(R, P)
    
    # Helper function for norm calculation
    def norm(x, ord=None, axis=None, keepdim=False):
        """Compute the matrix or vector norm."""
        if ord is None:
            # Default to Frobenius norm for matrices, L2 norm for vectors
            if x.ndim > 1:
                ord = 'fro'
            else:
                ord = 2
                
        if ord == 'fro':
            # Frobenius norm
            return mx.sqrt(mx.sum(mx.square(x)))
        elif ord == 2 and x.ndim == 1:
            # L2 norm for vectors
            return mx.sqrt(mx.sum(mx.square(x)))
        else:
            # For other norms, we would need to implement them
            # This is a simplified version
            return mx.sqrt(mx.sum(mx.square(x)))
    
    # Validate reconstruction
    recon_error = norm(mx.matmul(Q, R_perm) - A) / norm(A)
    print(f"‖QR−A‖/‖A‖ = {recon_error}")
    
    # Validate Q orthogonality
    QtQ = mx.matmul(Q.T, Q)
    eye = mx.eye(Q.shape[1], dtype=mx.float32)
    orth_error = norm(QtQ - eye) / norm(eye)
    print(f"‖QᵀQ−I‖/‖I‖ = {orth_error}")