import mlx.core as mx
import math
from typing import Optional, Tuple

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "shape", "col0"],
        output_names=["scratch"],
        ensure_row_contiguous=True
    )

# Metal kernel sources
_PANEL_SRC = """

#define TG_SIZE   64u
#define LIMBS     4u
#define LIMB_RADIX 4294967296.0f
#define EPSILON   1.0e-18f

    const uint tid = tidXYZ.x;
    const uint ltid = ltidXYZ.x;
    const uint m = shape[0], n = shape[1];
    const uint k = shape[2], panel = shape[3];
    const uint col0 = *col0_buf;
    const uint scratch_cols = n + 3 * k;

    device uint* colA = A;
    device uint* colV = A + m * scratch_cols + n;
    device uint* tauBuf = A + m * scratch_cols + n + k;
    device uint* pivBuf = A + m * scratch_cols + n + 2 * k;

    if (tid >= panel || col0 + tid >= k) return;

    threadgroup float shf[TG_SIZE];
    threadgroup float tg_norm;
    threadgroup uint tg_pivot;

    const uint col = col0 + tid;

    // Copy A to V
    for (uint r = ltid; r < m; r += TG_SIZE)
        colV[r * scratch_cols + col] = colA[r * scratch_cols + col];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute norm (simplified for float32)
    float fp32 = 0;
    for (uint r = ltid; r < m; r += TG_SIZE) {
        float v = as_type<float>(colV[r * scratch_cols + col]);
        fp32 = fma(v, v, fp32);
    }
    shf[ltid] = fp32;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce norms for pivot selection
    float simdf = shf[ltid];
    uint pivot = col;
    for (uint off = TG_SIZE >> 1; off; off >>= 1) {
        float other = (ltid < off) ? shf[ltid + off] : 0;
        if (other > simdf) {
            simdf = other;
            pivot = col + off;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ltid < off) shf[ltid] = simdf;
    }
    if (ltid == 0) {
        shf[0] = simdf;
        tg_norm = sqrt(max(simdf, EPSILON));
        tg_pivot = pivot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Householder reflector for pivot column
    if (col == tg_pivot && ltid == 0) {
        float norm = tg_norm;
        float x1 = as_type<float>(colV[col * scratch_cols + col]);
        float alpha = -sign(x1) * norm;
        float u1 = x1 - alpha;
        float u_norm_sq = norm * norm - x1 * x1;
        for (uint r = col + 1; r < m; r++) {
            float v = as_type<float>(colV[r * scratch_cols + col]);
            u_norm_sq = fma(v, v, u_norm_sq);
        }
        float u_norm = sqrt(max(u_norm_sq, EPSILON));
        float inv = u_norm > EPSILON ? 1.0f / u_norm : 0.0f;
        float tau = u_norm > EPSILON ? 2.0f / (1.0f + u1 * u1 / u_norm_sq) : 0.0f;
        tauBuf[col] = as_type<uint>(tau);
        pivBuf[col] = tg_pivot;
        colV[col * scratch_cols + col] = as_type<uint>(u1 * inv);
        for (uint r = col + 1; r < m; r++) {
            float v = as_type<float>(colV[r * scratch_cols + col]);
            colV[r * scratch_cols + col] = as_type<uint>(v * inv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_APPLY_SRC = """
#define BLK 128u
#define SIMD_SIZE 8u

    uint3 g = thread_position_in_grid.x;
    uint3 l = thread_position_in_threadgroup.x;
    uint3 sgid = thread_position_in_simdgroup.x;
    uint simd_lane = simd_lane_id.x;
    const uint m = shape[0], n = shape[1], k = shape[2], panel = shape[3];
    const uint col0 = *col0_buf;
    const uint scratch_cols = n + 3 * k;
    const uint blk_i = g.y, blk_j = g.x;
    const uint row0 = blk_i * BLK + l.y * SIMD_SIZE + sgid.y;
    const uint col0_global = blk_j * BLK + l.x * SIMD_SIZE + sgid.x + col0 + panel;

    if (row0 >= m || col0_global >= n) return;

    threadgroup float v_cache[BLK][SIMD_SIZE];
    threadgroup float tau_cache[SIMD_SIZE];

    if (l.y == 0 && sgid.y == 0) {
        for (uint p = sgid.x; p < panel; p += SIMD_SIZE) {
            tau_cache[p % SIMD_SIZE] = as_type<float>(A[m * scratch_cols + n + k + (col0 + p)]);
        }
    }

    for (uint p = l.x; p < panel; p += SIMD_SIZE) {
        if (sgid.x < SIMD_SIZE) {
            v_cache[l.y * SIMD_SIZE + sgid.y][p % SIMD_SIZE] = as_type<float>(A[row0 * scratch_cols + (n + col0 + p)]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    float error = 0.0f;
    for (uint p = 0; p < panel; p += SIMD_SIZE) {
        float v[SIMD_SIZE];
        float tau[SIMD_SIZE];
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            v[i] = v_cache[l.y * SIMD_SIZE + sgid.y][i];
            tau[i] = tau_cache[i];
        }

        float a = as_type<float>(A[row0 * scratch_cols + col0_global]);
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            float temp = acc;
            float y = v[i] * a * tau[i] - error;
            float t = temp + y;
            error = (t - temp) - y;
            acc = t;
        }
    }

    float newA = as_type<float>(A[row0 * scratch_cols + col0_global]) - 2.0f * acc;
    A[row0 * scratch_cols + col0_global] = as_type<uint>(newA);
"""

_BUILDQ_SRC = """
#define TG 32u

    uint3 gsz = threadgroups_per_grid.x;
    uint3 tid = thread_position_in_grid.x;
    uint3 ltid = thread_position_in_threadgroup.x;
    const uint m = shape[0], k = shape[2];
    const uint n = shape[1];
    const uint scratch_cols = n + 3 * k;
    if (tid.x >= k || tid.y >= m) return;

    float q = (tid.x == tid.y) ? 1.0f : 0.0f;
    for (int p = k - 1; p >= 0; --p) {
        float v = as_type<float>(A[tid.y * scratch_cols + (n + p)]);
        float tau = as_type<float>(A[m * scratch_cols + n + k + p]);
        q -= 2.0f * tau * v * q;
    }
    A[m * scratch_cols + n + tid.y * scratch_cols + tid.x] = as_type<uint>(q);
"""

# Compile kernels
panelK = _compile(_PANEL_SRC, "panel_factor_qrp128")
applyK = _compile(_APPLY_SRC, "apply_update_qrp128")
buildK = _compile(_BUILDQ_SRC, "build_q_qrp128")

def qr128_qrp(A: mx.array, want_q: bool = False) -> Tuple[Optional[mx.array], mx.array, mx.array]:
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    m, n = map(int, A.shape)
    k = min(m, n)
    panel = 64
    scratch_cols = n + k + k + k
    S = mx.zeros((m, scratch_cols), dtype=mx.uint32)
    S[:, :n] = A.view(dtype=mx.uint32)

    shape = mx.array([m, n, k, panel, 4], dtype=mx.uint32)

    for col0 in range(0, k, panel):
        col0_buf = mx.array([col0], dtype=mx.uint32)
        grid = (1, 1, 1)
        threadgroup = (min(panel, k - col0), 1, 1)
        panelK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
               grid=grid, threadgroup=threadgroup)

        right0 = col0 + panel
        if right0 < n:
            blocks = (math.ceil((n - right0) / 128), math.ceil(m / 128), 1)
            applyK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
                   grid=blocks, threadgroup=(8, 8, 1))

    if want_q:
        col0_buf = mx.array([0], dtype=mx.uint32)
        buildK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32], 
               grid=(math.ceil(m / 32), math.ceil(k / 32), 1), threadgroup=(32, 1, 1))
        Q = S[:, n:n+k].view(dtype=A.dtype)
    else:
        Q = None

    R = S[:, :n].view(dtype=A.dtype)
    piv = S[0, n+2*k:n+3*k].view(dtype=mx.int32)
    return Q, R, piv

if __name__ == "__main__":
    A = mx.array([[4, 1, 2], [2, 3, 1], [1, 2, 5]], dtype=mx.float32)
    Q, R, piv = qr128_qrp(A, want_q=True)
    
    n = A.shape[1]
    P = mx.zeros((n, n), dtype=A.dtype)
    for i, p in enumerate(piv.tolist()):
        if i < n and p < n:
            P[i, p] = 1.0
    
    R_perm = mx.matmul(R, P)
    
    def norm(x, ord=None):
        if ord is None:
            ord = 'fro' if x.ndim > 1 else 2
        if ord == 'fro' or (ord == 2 and x.ndim == 1):
            return mx.sqrt(mx.sum(mx.square(x)))
        return mx.sqrt(mx.sum(mx.square(x)))
    
    recon_error = norm(mx.matmul(Q, R_perm) - A) / norm(A)
    print(f"‖QR−A‖/‖A‖ = {recon_error}")
    
    QtQ = mx.matmul(Q.T, Q)
    eye = mx.eye(Q.shape[1], dtype=mx.float32)
    orth_error = norm(QtQ - eye) / norm(eye)
    print(f"‖QᵀQ−I‖/‖I‖ = {orth_error}")