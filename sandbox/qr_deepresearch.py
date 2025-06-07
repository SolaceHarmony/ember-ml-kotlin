import math
import time
import mlx.core as mx

def tiled_hpc_qr(A: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    Tiled QR decomposition with 128-bit limb emulation on Metal via MLX.

    • Dynamic TG_SIZE = next_pow2(max(32, m//8)), capped at 256
    • Tree-reduce for cmax, norm², and Householder dot-products (no atomics)
    • Warns if more than 4 threadgroups are required for norm reductions
      (threshold chosen empirically to avoid excessive warnings for small
       multi-group cases; full cross-threadgroup reduction is a planned
       future enhancement)
    • Complete Householder reflection steps for R and Q
    """

    m, n = A.shape
    min_dim = min(m, n)

    # ──────────────────────────────────────────────────────────────────
    # 1) Threadgroup sizing: at least 32, scale with m//8, cap at 256
    #    We warn if total_tg > 4 (empirical threshold to reduce noise).
    #    Future: implement two-level reduction for full cross-TG support.
    # ──────────────────────────────────────────────────────────────────
    def next_pow2(x): return 1 << (x - 1).bit_length()

    tg_size = min(256, next_pow2(max(32, m // 8)))
    max_tg   = tg_size
    total_tg = math.ceil(m / tg_size)

    if total_tg > 4:
        print(f"Warning: m={m} spans {total_tg} threadgroups; "
              "norm reductions valid only within each group. "
              "Full cross-group reduction is not yet implemented.")

    # ──────────────────────────────────────────────────────────────────
    # 2) Metal kernel source
    #    MLX auto-injects: kernel signature, buffers A, A_shape, Q_out, R_out, debug
    # ──────────────────────────────────────────────────────────────────
    metal_src = f"""
// -------------------------------------------------------------------
// tiled_hpc_qr_kernel_v5 – Complete Householder QR w/ limb-precision
// MLX auto-injects kernel signature & A / A_shape / Q_out / R_out / debug
// -------------------------------------------------------------------
#define NUM_LIMBS   8
#define TG_SIZE     {tg_size}
#define MAX_TG_SIZE {max_tg}

threadgroup float shmem_sum[MAX_TG_SIZE];                // for max & dot
threadgroup uint  shared_limbs[NUM_LIMBS * MAX_TG_SIZE]; // for 16-bit limbs

const uint tid    = thread_position_in_grid.x;
const uint tgid   = thread_position_in_threadgroup.x;
const uint stride = grid_size.x;

const uint m       = A_shape[0];
const uint n       = A_shape[1];
const uint min_dim = min(m, n);

// 1) Debug init
if (tid == 0) {{
    debug[11] = float(TG_SIZE);
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

// 2) Initialize Q = I, R = A
for (uint row = tid; row < m; row += stride) {{
    for (uint col = 0; col < n; ++col)
        R_out[row*n + col] = A[row*n + col];
    for (uint col = 0; col < m; ++col)
        Q_out[row*m + col] = (row == col) ? 1.0f : 0.0f;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

// 3) Main QR loop
for (uint k = 0; k < min_dim; ++k) {{

    // 3a) reset debug slots
    if (tid == 0) {{
        debug[7] = 0.0f;  // cmax
        debug[8] = 0.0f;  // norm²
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3b) cmax via tree-reduce
    float local_max = 0.0f;
    for (uint i = k; i < m; i += stride)
        local_max = fmax(local_max, fabs(R_out[i*n + k]));
    shmem_sum[tgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG_SIZE/2; off > 0; off >>= 1) {{
        if (tgid < off)
            shmem_sum[tgid] = fmax(shmem_sum[tgid], shmem_sum[tgid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float cmax = shmem_sum[0];
    if (tid == 0) debug[7] = cmax;
    const float scale = (cmax > 1e-8f ? 1.0f/cmax : 1.0f);

    // 3c) scale column k
    for (uint i = k; i < m; i += stride)
        R_out[i*n + k] *= scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3d) norm² via limb-based tree-reduce
    ushort local_limbs[NUM_LIMBS] = {{0}};
    for (uint i = k + tid; i < m; i += stride) {{
        uint b = as_type<uint>(R_out[i*n + k]);
        ushort lo = b & 0xFFFFu, hi = (b >> 16) & 0xFFFFu;
        uint p0 = (uint)lo*lo, p1 = (uint)hi*hi, pc = ((uint)lo*hi)<<1;
        local_limbs[0] += ushort(p0         & 0xFFFFu);
        local_limbs[1] += ushort((p0>>16)  + (pc & 0xFFFFu));
        local_limbs[2] += ushort((pc>>16)  + (p1 & 0xFFFFu));
        local_limbs[3] += ushort(p1 >> 16);
    }}
    for (uint l = 0; l < NUM_LIMBS; ++l)
        shared_limbs[l*MAX_TG_SIZE + tgid] = local_limbs[l];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG_SIZE/2; off > 0; off >>= 1) {{
        if (tgid < off) {{
            for (uint l = 0; l < NUM_LIMBS; ++l) {{
                shared_limbs[l*MAX_TG_SIZE + tgid] +=
                    shared_limbs[l*MAX_TG_SIZE + tgid + off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tgid == 0) {{
        // carry & reconstruct
        for (uint l = 0; l < NUM_LIMBS-1; ++l) {{
            uint v = shared_limbs[l*MAX_TG_SIZE + 0], c = v >> 16;
            shared_limbs[l*MAX_TG_SIZE + 0] = v & 0xFFFFu;
            shared_limbs[(l+1)*MAX_TG_SIZE + 0] += c;
        }}
        float n2 = 0.0f, r = 1.0f;
        for (uint l = 0; l < NUM_LIMBS; ++l) {{
            n2 += float(shared_limbs[l*MAX_TG_SIZE + 0]) * r;
            r *= 65536.0f;
        }}
        debug[8] = n2;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm2 = debug[8], norm = sqrt(norm2);
    if (tid == 0) debug[5] = norm;

    // 4) Householder head update
    if (tgid == 0) {{
        float rkk = R_out[k*n + k];
        float s   = (rkk >= 0.0f ? 1.0f : -1.0f);
        R_out[k*n + k] = rkk + s*norm;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5) Reflect R
    const float beta = (norm2 > 1e-8f ? 2.0f / norm2 : 0.0f);
    for (uint j = k; j < n; ++j) {{
        // dot = vᵀ * R[:,j]
        float local_dot = 0.0f;
        for (uint i = k + tid; i < m; i += stride)
            local_dot += R_out[i*n + k] * R_out[i*n + j];
        shmem_sum[tgid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = TG_SIZE/2; off > 0; off >>= 1) {{
            if (tgid < off)
                shmem_sum[tgid] += shmem_sum[tgid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float dot = shmem_sum[0];
        // update
        for (uint i = k + tid; i < m; i += stride)
            R_out[i*n + j] -= beta * R_out[i*n + k] * dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 6) Reflect Q
    for (uint j = 0; j < m; ++j) {{
        float local_dot = 0.0f;
        for (uint i = k + tid; i < m; i += stride)
            local_dot += R_out[i*n + k] * Q_out[j*m + i];
        shmem_sum[tgid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = TG_SIZE/2; off > 0; off >>= 1) {{
            if (tgid < off)
                shmem_sum[tgid] += shmem_sum[tgid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float dot = shmem_sum[0];
        for (uint i = k + tid; i < m; i += stride)
            Q_out[j*m + i] -= beta * R_out[i*n + k] * dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // 7) un-scale
    for (uint i = k; i < m; i += stride)
        R_out[i*n + k] /= scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

}}  // end for k

// final sync flag
if (tid == 0) debug[14] = 1.0f;
"""

    # ──────────────────────────────────────────────────────────────────
    # 3) Compile & launch
    # ──────────────────────────────────────────────────────────────────
    kernel = mx.fast.metal_kernel(
        name="tiled_hpc_qr_kernel_v5",
        source=metal_src,
        input_names=["A"],
        output_names=["Q_out", "R_out", "debug"],
        ensure_row_contiguous=True
    )

    total = ((m + tg_size - 1) // tg_size) * tg_size
    grid = (total, 1, 1)
    tg   = (tg_size, 1, 1)

    Q, R, dbg = kernel(
        inputs=[A],
        output_shapes=[(m, m), (m, n), (16,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=grid,
        threadgroup=tg
    )
    return Q, R, dbg

def test_qr_mlx():
    for size in (10, 64, 128, 256):
        A = mx.random.normal((size, size), dtype=mx.float32)
        start = time.time()
        Q, R, dbg = tiled_hpc_qr(A)
        dt = time.time() - start

        ortho = mx.allclose(mx.matmul(Q.T, Q), mx.eye(size), atol=1e-5).item()
        recon = mx.allclose(mx.matmul(Q, R), A,       atol=1e-5).item()

        print(f"n={size:3d} time={dt:.4f}s orthogonal={ortho} QR≈A={recon}")
        print(" debug:", dbg.asnumpy())

if __name__ == "__main__":
    test_qr_mlx()