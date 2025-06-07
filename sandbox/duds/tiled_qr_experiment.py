import mlx.core as mx
import time

# This script will contain an experimental tiled QR implementation
# inspired by the SVD Metal kernel structure.

def tiled_qr(a: mx.array) -> tuple[mx.array, mx.array]:
    """
    Experimental tiled QR decomposition using a Metal kernel.
    Inspired by SVD kernel tiling patterns.
    """
    m, n = a.shape
    min_dim = min(m, n)

    # Define Metal kernel source string for tiled QR decomposition
    # This is a basic Householder QR implementation in Metal.
    # Tiling and HPC aspects are not yet implemented here.
    metal_kernel_source = """
    #define EPSILON 1e-10f

    uint m = shapeParams[0];
    uint n = shapeParams[1];
    uint min_dim = min(m, n);

    // Initialize Q to identity (Q_out is buffer 1)
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < m; j++) {
            Q_out[i * m + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Initialize R to A (R_out is buffer 2, A is buffer 0)
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++) {
            R_out[i * n + j] = A[i * n + j];
        }
    }

    // Perform QR decomposition using Householder reflections (single-threaded loop)
    // A full tiled implementation would parallelize the application of reflectors
    // to blocks of the matrix.
    if (thread_position_in_grid.x == 0) { // Ensure single thread for the main loop
        for (uint k = 0; k < min_dim; k++) {
            // Allocate x and v in thread memory. This assumes m is small enough.
            // For larger m, threadgroup memory or other strategies would be needed.
            thread float x[4096]; // Assuming max m is within a reasonable limit for thread memory
            thread float v[4096]; // Assuming max m is within a reasonable limit for thread memory

            float norm_sq = 0.0f;
            for (uint i = k; i < m; i++) {
                x[i-k] = R_out[i * n + k]; // R_out is buffer 2
                norm_sq += x[i-k] * x[i-k];
            }
            float norm = sqrt(norm_sq);
            if (norm < EPSILON) continue;
            float sign = (x[0] >= 0.0f) ? 1.0f : -1.0f;
            x[0] += sign * norm;
            norm_sq = 0.0f;
            for (uint i = 0; i < m-k; i++) norm_sq += x[i] * x[i];
            if (norm_sq < EPSILON) continue;
            float inv_norm_sq = 1.0f / norm_sq;
            for (uint i = 0; i < m-k; i++) v[i] = x[i];

            // Apply reflection to R_out (R_out is buffer 2)
            for (uint j = k; j < n; j++) {
                float dot = 0.0f;
                for (uint i = 0; i < m-k; i++) dot += v[i] * R_out[(i+k) * n + j];
                float factor = 2.0f * dot * inv_norm_sq;
                for (uint i = 0; i < m-k; i++) R_out[(i+k) * n + j] -= factor * v[i];
            }
            // Apply reflection to Q_out (Q_out is buffer 1)
            for (uint j = 0; j < m; j++) {
                 float dot = 0.0f;
                 for (uint i = 0; i < m-k; i++) dot += Q_out[j * m + (i+k)] * v[i];
                 float factor = 2.0f * dot * inv_norm_sq;
                 for (uint i = 0; i < m-k; i++) Q_out[j * m + (i+k)] -= factor * v[i];
            }
        }
        // Zero out lower R_out (R_out is buffer 2)
        for (uint i = 1; i < m; i++) for (uint j = 0; j < min(i, n); j++) R_out[i*n+j] = 0.0f;
    }
    """

    try:
        compiled_kernel = mx.fast.metal_kernel(
            name="tiled_qr_kernel",
            source=metal_kernel_source,
            input_names=["A", "shapeParams"],
            output_names=["Q_out", "R_out"],
            ensure_row_contiguous=True
        )
    except Exception as e:
        print(f"Failed to compile Metal kernel: {e}")
        print("Please check the Metal kernel source for errors.")
        return mx.eye(m), mx.array(a) # Return identity Q and original A as a fallback

    # Determine grid and threadgroup sizes. For this single-threaded kernel,
    # a single threadgroup with one thread is sufficient.
    grid_size = (1, 1, 1)
    threadgroup_size = (1, 1, 1)

    shape_params = mx.array([m, n], dtype=mx.uint32)

    try:
        # Execute the kernel
        result = compiled_kernel(
            inputs=[a, shape_params],
            output_shapes=[(m, m), (m, n)],
            output_dtypes=[a.dtype, a.dtype],
            grid=grid_size,
            threadgroup=threadgroup_size
        ) # type: ignore
        return result[0], result[1]
    except Exception as e:
        print(f"Metal kernel execution failed: {e}")
        return mx.eye(m), mx.array(a) # Return identity Q and original A as a fallback


if __name__ == "__main__":
    # Example Usage

    # Square matrix
    A_square = mx.random.normal((100, 100))
    print("Processing square matrix (100x100)...")
    start_time = time.time()
    Q_square, R_square = tiled_qr(A_square)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    # print("Q_square:", Q_square)
    # print("R_square:", R_square)
    print("Orthogonality check (Q_square.T @ Q_square - I):", mx.mean(mx.abs(mx.matmul(Q_square.T, Q_square) - mx.eye(100))).item())
    print("Reconstruction check (Q_square @ R_square - A_square):", mx.mean(mx.abs(mx.matmul(Q_square, R_square) - A_square)).item())
    print("-" * 20)

    # Tall matrix
    A_tall = mx.random.normal((200, 50))
    print("Processing tall matrix (200x50)...")
    start_time = time.time()
    Q_tall, R_tall = tiled_qr(A_tall)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    # print("Q_tall:", Q_tall)
    # print("R_tall:", R_tall)
    print("Orthogonality check (Q_tall.T @ Q_tall - I):", mx.mean(mx.abs(mx.matmul(Q_tall.T, Q_tall) - mx.eye(200))).item()) # Note: Q is M x M
    print("Reconstruction check (Q_tall @ R_tall - A_tall):", mx.mean(mx.abs(mx.matmul(Q_tall, R_tall) - A_tall)).item())
    print("-" * 20)

    # Wide matrix
    A_wide = mx.random.normal((50, 200))
    print("Processing wide matrix (50x200)...")
    start_time = time.time()
    Q_wide, R_wide = tiled_qr(A_wide)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    # print("Q_wide:", Q_wide)
    # print("R_wide:", R_wide)
    print("Orthogonality check (Q_wide.T @ Q_wide - I):", mx.mean(mx.abs(mx.matmul(Q_wide.T, Q_wide) - mx.eye(50))).item()) # Note: Q is M x M
    print("Reconstruction check (Q_wide @ R_wide - A_wide):", mx.mean(mx.abs(mx.matmul(Q_wide, R_wide) - A_wide)).item())
    print("-" * 20)

    # Note: This is a basic implementation. For performance, tiling and
    # parallelization within the kernel would be necessary.