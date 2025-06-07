from tabnanny import verbose
import mlx.core as mx
import time
import numpy as np

def simple_mgs_qr(A: mx.array) -> tuple[mx.array, mx.array]:
    """
    A minimal Metal‐kernel QR via Modified Gram–Schmidt.
    Proves that MLX dispatch, buffers, and barriers are correct.
    """
    m, n = A.shape
    k    = min(m, n)
    TG   = 32  # threadgroup size

    metal_src = """
    const uint tid = thread_position_in_grid.x;
    const uint tgid = thread_position_in_threadgroup.x;
    const uint stride = grid_size.x;

    const uint m = A_shape[0];
    const uint n = A_shape[1];
    const uint k = (m < n) ? m : n;
    const uint TG = 32;

    if (threads_per_threadgroup.x != TG) {
        return; // enforce threadgroup size
    }

    // Zero initialize Q_out (m x k)
    for (uint idx = tid; idx < m * k; idx += stride) {
        Q_out[idx] = 0.0f;
    }

    // Zero initialize R_out (k x n)
    for (uint idx = tid; idx < k * n; idx += stride) {
        R_out[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Modified Gram-Schmidt
    for (uint j = 0; j < k; ++j) {
        // Load A[:, j] into Q[:, j]
        for (uint i = tid; i < m; i += stride) {
            Q_out[i * k + j] = A[i * n + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Orthogonalize Q[:, j] against Q[:, 0..j-1]
        for (uint i = 0; i < j; ++i) {
            float local_dot = 0.0f;
            for (uint row = tid; row < m; row += stride) {
                local_dot += Q_out[row * k + i] * Q_out[row * k + j];
            }
            threadgroup float sh_dot[32]; // Fixed size to match TG
            sh_dot[tgid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduction with explicit steps to avoid divergence
            if (tgid < 16) sh_dot[tgid] += sh_dot[tgid + 16];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tgid < 8) sh_dot[tgid] += sh_dot[tgid + 8];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tgid < 4) sh_dot[tgid] += sh_dot[tgid + 4];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tgid < 2) sh_dot[tgid] += sh_dot[tgid + 2];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tgid < 1) sh_dot[tgid] += sh_dot[tgid + 1];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot_val = sh_dot[0];
            if (tgid == 0) {
                R_out[i * n + j] = dot_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint row = tid; row < m; row += stride) {
                Q_out[row * k + j] -= dot_val * Q_out[row * k + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Compute norm of Q[:, j]
        float local_norm = 0.0f;
        for (uint row = tid; row < m; row += stride) {
            float val = Q_out[row * k + j];
            local_norm += val * val;
        }
        threadgroup float sh_norm[32]; // Fixed size to match TG
        sh_norm[tgid] = local_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduction with explicit steps to avoid divergence
        if (tgid < 16) sh_norm[tgid] += sh_norm[tgid + 16];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < 8) sh_norm[tgid] += sh_norm[tgid + 8];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < 4) sh_norm[tgid] += sh_norm[tgid + 4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < 2) sh_norm[tgid] += sh_norm[tgid + 2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgid < 1) sh_norm[tgid] += sh_norm[tgid + 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm_val = 0.0f;
        if (tgid == 0) {
            norm_val = sqrt(max(sh_norm[0], 1e-8f));
            R_out[j * n + j] = norm_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint row = tid; row < m; row += stride) {
            Q_out[row * k + j] /= norm_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """


    kern = mx.fast.metal_kernel(
        name="simple_mgs_qr",
        source=metal_src,
        input_names=["A"],
        output_names=["Q_out", "R_out"],
        ensure_row_contiguous=True
    )
    total_threads = ((m + TG - 1) // TG) * TG
    Q, R = kern(
        inputs=[A],
        output_shapes=[(m, k), (k, n)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(total_threads, 1, 1),
        threadgroup=(TG, 1, 1),
        verbose=True,
    )
    Q=mx.reshape(Q, (m, k))
    R=mx.reshape(R, (k, n))
    return Q, R

if __name__ == "__main__":
    A = mx.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=mx.float32)
    Q, R = simple_mgs_qr(A)

    print("A =\n", A)
    print("Q =\n", Q)
    print("R =\n", R)

    # verify QR ≈ A
    recon_err = np.linalg.norm(Q.asnumpy().dot(R.asnumpy()) - A.asnumpy())
    print("reconstruction error:", recon_err)
