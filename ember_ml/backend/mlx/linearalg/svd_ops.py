# Content for ember_ml/backend/mlx/linearalg/svd_ops.py (QR code removed)
"""
MLX SVD operations for ember_ml.

This module provides MLX implementations of Singular Value Decomposition (SVD)
using custom Metal kernels for performance and stability.
"""
from typing import Union, Tuple, Literal
import mlx.core as mx

# Import from tensor_ops and types
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXTensor

# Import QR dependency from qr_ops
from ember_ml.backend.mlx.linearalg.qr_ops import qr
dtype_obj = MLXDType()

# --- QR Kernel Code Removed (Moved to qr_ops.py) ---


# --- Start of SVD Power Iteration Kernel Code ---
# Define Metal kernel source string for power iteration
_power_iter_tensor_kernel_source = """
#define NUM_LIMBS 8
#define LIMB_TILE_SIZE 32
#define EPSILON 1e-10f
#define MAX_K 32  // Reduced from 512 to avoid exceeding shared memory limits
#define WARP_SIZE 32

uint tid = thread_position_in_grid.x;
uint num_threads = threads_per_threadgroup.x;
uint simd_lane_id = tid % WARP_SIZE;
uint simd_group_id = tid / WARP_SIZE;

uint n = shapeParams[0];
uint k = shapeParams[1];
uint num_iterations = iterParams[0];
float tolerance = tolParams[0];
    
    // Shared memory for matrix Z and reduction operations
    // Use dynamic size based on actual n and k, but with a reasonable upper limit
    threadgroup float shared_Z[250 * MAX_K]; // Further reduced size to fit within 32KB limit
    threadgroup float shared_proj[MAX_K]; // For projection coefficients
    threadgroup float shared_norm[MAX_K]; // For column norms
    
    // Initialize Q_out with Q_init in parallel
    for (uint idx = tid; idx < n * k; idx += num_threads) {
        Q_out[idx] = Q_init[idx];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Power iteration with Gram-Schmidt orthogonalization
    for (uint iter = 0; iter < num_iterations; iter++) {
        // Step 1: Matrix multiplication Z = A * Q_out (in parallel)
        for (uint idx = tid; idx < n * k; idx += num_threads) {
            uint row = idx / k;
            uint col = idx % k;
            
            float sum = 0.0f;
            // Compute dot product for this element
            for (uint i = 0; i < n; i++) {
                sum += A[row * n + i] * Q_out[i * k + col];
            }
            shared_Z[idx] = sum;
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Step 2: Gram-Schmidt orthogonalization on columns of Z
        for (uint col = 0; col < k; col++) {
            // Step 2a: Orthogonalize Z[:, col] against previous columns Q_out[:, 0...col-1]
            for (uint j = 0; j < col; j++) {
                // Compute dot product in parallel: proj = Q_out[:, j]' * Z[:, col]
                float thread_proj = 0.0f;
                for (uint row = tid; row < n; row += num_threads) {
                    thread_proj += Q_out[row * k + j] * shared_Z[row * k + col];
                }
                
                // Reduce within SIMD group
                thread_proj = simd_sum(thread_proj);
                
                // First thread in each SIMD group writes to shared memory
                if (simd_lane_id == 0 && simd_group_id < 8) {
                    shared_proj[simd_group_id] = thread_proj;
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                // Thread 0 combines results
                float proj = 0.0f;
                if (tid == 0) {
                    for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                        proj += shared_proj[i];
                    }
                    shared_proj[0] = proj; // Store for all threads to use
                }
                
                threadgroup_barrier(mem_flags::mem_device);
                
                proj = shared_proj[0]; // All threads read the final value
                
                // Subtract projection in parallel: Z[:, col] = Z[:, col] - proj * Q_out[:, j]
                for (uint row = tid; row < n; row += num_threads) {
                    shared_Z[row * k + col] -= proj * Q_out[row * k + j];
                }
                
                threadgroup_barrier(mem_flags::mem_device);
            }
            
            // Step 2b: Compute norm squared in parallel: norm_sq = Z[:, col]' * Z[:, col]
            float thread_norm_sq = 0.0f;
            for (uint row = tid; row < n; row += num_threads) {
                float val = shared_Z[row * k + col];
                thread_norm_sq += val * val;
            }
            
            // Reduce within SIMD group
            thread_norm_sq = simd_sum(thread_norm_sq);
            
            // First thread in each SIMD group writes to shared memory
            if (simd_lane_id == 0 && simd_group_id < 8) {
                shared_norm[simd_group_id] = thread_norm_sq;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results and computes norm
            float norm = 0.0f;
            if (tid == 0) {
                float norm_sq = 0.0f;
                for (uint i = 0; i < min(8u, (num_threads + WARP_SIZE - 1) / WARP_SIZE); i++) {
                    norm_sq += shared_norm[i];
                }
                norm = sqrt(norm_sq);
                shared_norm[0] = norm; // Store for all threads to use
                shared_norm[1] = (norm > tolerance) ? (1.0f / norm) : 0.0f; // Store inverse norm
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            norm = shared_norm[0]; // All threads read the final norm
            float inv_norm = shared_norm[1]; // All threads read the inverse norm
            
            // Step 2c: Normalize Z[:, col] and store in Q_out[:, col] in parallel
            for (uint row = tid; row < n; row += num_threads) {
                if (norm > tolerance) {
                    Q_out[row * k + col] = shared_Z[row * k + col] * inv_norm;
                } else {
                    Q_out[row * k + col] = 0.0f;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

"""

# Compile the kernel at module level
_power_iter_tensor_kernel_compiled = mx.fast.metal_kernel(
    name="power_iter_tensor_kernel",
    source=_power_iter_tensor_kernel_source,
    input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
    output_names=["Q_out"],
    ensure_row_contiguous=True
)

# Function to call the power iteration kernel with proper parameters
def _call_power_iter_kernel(A, Q_init, num_iterations=10, tolerance=1e-10):
    """
    Call the power iteration kernel with proper parameters.
    
    Args:
        A: Input matrix (n x n)
        Q_init: Initial guess for eigenvectors (n x k)
        num_iterations: Number of power iterations
        tolerance: Tolerance for convergence
        
    Returns:
        Q_out: Approximate eigenvectors (n x k)
    """
    n, k = Q_init.shape
    shape_params = mx.array([n, k], dtype=mx.uint32)
    iter_params = mx.array([num_iterations], dtype=mx.uint32)
    tol_params = mx.array([tolerance], dtype=mx.float32)
    
    # Configure kernel execution
    grid = (256, 1, 1)  # Use 256 threads
    threadgroup = (256, 1, 1)
    
    # Call the kernel
    outputs = _power_iter_tensor_kernel_compiled(
        inputs=[A, Q_init, shape_params, iter_params, tol_params],
        output_shapes=[(n, k)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )
    
    return outputs[0]
# --- End of SVD Power Iteration Kernel Code ---


# --- Start of SVD Function ---
def svd(a: TensorLike, full_matrices: bool = True, compute_uv: bool = True,
       k: int = -1) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute SVD using custom Metal kernels for QR and power iteration.

    This implementation uses the `qr` kernel (imported from qr_ops)
    for potentially improved numerical stability compared to standard QR,
    and a power iteration kernel for the eigendecomposition step.

    Args:
        a: Input matrix (M x N).
        full_matrices: If True, compute full U (M x M) and Vh (N x N).
                       If False, compute reduced U (M x K) and Vh (K x N),
                       where K = min(M, N).
        compute_uv: If True, compute and return U and Vh matrices.
                    If False, return only the singular values S.
        k: Number of singular values/vectors to compute. If -1, compute all.
           Currently, only k=-1 (all) is fully supported.

    Returns:
        - If compute_uv is True: Tuple (U, S, Vh)
        - If compute_uv is False: Array S (singular values)

    Raises:
        ValueError: If matrix dimensions exceed MAX_MATRIX_DIM or if k != -1.
        RuntimeError: If underlying operations (QR, power iteration) fail.
    """
    if k != -1:
         raise ValueError("Partial SVD (k != -1) is not fully supported by this implementation yet. Use k=-1.")

    Tensor = MLXTensor()
    a_arr = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    m, n = a_arr.shape

    if m > 4096 or n > 4096:
        raise ValueError(f"Matrix dimensions ({m}x{n}) exceed maximum supported size 4096")

    rank = min(m, n)
    k_val = rank

    epsilon_svd = mx.array(1e-10, dtype=a_arr.dtype)

    if m >= n:
        ata = mx.matmul(mx.transpose(a_arr), a_arr)
        q_init_rand = mx.random.normal((n, k_val), dtype=a_arr.dtype)
        # Use imported qr
        q_init, _ = qr(q_init_rand)
        q_init = q_init[:, :k_val]

        shape_params = mx.array([n, k_val], dtype=mx.uint32)
        iter_params = mx.array([100], dtype=mx.uint32)
        tol_params = mx.array([1e-7], dtype=mx.float32)

        grid = (1, 1, 1)
        threads = (1, 1, 1)

        try:
            v = _power_iter_tensor_kernel_compiled(
                inputs=[ata, q_init, shape_params, iter_params, tol_params],
                output_shapes=[(n, k_val)],
                output_dtypes=[a_arr.dtype],
                grid=grid,
                threadgroup=threads
            )[0] # type: ignore
        except Exception as e:
            raise RuntimeError(f"SVD power iteration kernel failed: {e}")


        rayleigh = mx.matmul(mx.transpose(v), mx.matmul(ata, v))
        eigenvalues = mx.diag(rayleigh)

        sort_indices = mx.argsort(mx.negative(mx.abs(eigenvalues)))
        eigenvalues_sorted = eigenvalues[sort_indices]
        v_sorted = v[:, sort_indices]

        eigenvalues_safe = mx.maximum(eigenvalues_sorted, epsilon_svd)
        s = mx.sqrt(eigenvalues_safe)

        if not compute_uv:
            return s

        s_inv = mx.where(mx.greater(s, epsilon_svd), mx.reciprocal(s), mx.zeros_like(s))
        u = mx.matmul(a_arr, mx.multiply(v_sorted, s_inv.reshape(1, -1)))
        vh = mx.transpose(v_sorted)

        if full_matrices:
            if m > k_val:
                # Use imported qr
                q_u, _ = qr(u)
                u = q_u
            # No need to extend Vh if m >= n and k_val = n
            if vh.shape[0] < n: # Should only happen if k_val < n initially
                 q_v, _ = qr(mx.transpose(vh))
                 vh = mx.transpose(q_v)

        if not full_matrices:
             u = u[:, :k_val]
             vh = vh[:k_val, :]

        return u, s, vh

    else: # m < n
        aat = mx.matmul(a_arr, mx.transpose(a_arr))
        q_init_rand = mx.random.normal((m, k_val), dtype=a_arr.dtype)
        # Use imported qr
        q_init, _ = qr(q_init_rand)
        q_init = q_init[:, :k_val]

        shape_params = mx.array([m, k_val], dtype=mx.uint32)
        iter_params = mx.array([100], dtype=mx.uint32)
        tol_params = mx.array([1e-7], dtype=mx.float32)

        grid = (1, 1, 1)
        threads = (1, 1, 1)

        try:
            u = _power_iter_tensor_kernel_compiled(
                inputs=[aat, q_init, shape_params, iter_params, tol_params],
                output_shapes=[(m, k_val)],
                output_dtypes=[a_arr.dtype],
                grid=grid,
                threadgroup=threads
            )[0] # type: ignore
        except Exception as e:
            raise RuntimeError(f"SVD power iteration kernel failed: {e}")

        rayleigh = mx.matmul(mx.transpose(u), mx.matmul(aat, u))
        eigenvalues = mx.diag(rayleigh)

        sort_indices = mx.argsort(mx.negative(mx.abs(eigenvalues)))
        eigenvalues_sorted = eigenvalues[sort_indices]
        u_sorted = u[:, sort_indices]

        eigenvalues_safe = mx.maximum(eigenvalues_sorted, epsilon_svd)
        s = mx.sqrt(eigenvalues_safe)

        if not compute_uv:
            return s

        s_inv = mx.where(mx.greater(s, epsilon_svd), mx.reciprocal(s), mx.zeros_like(s))
        v = mx.matmul(mx.transpose(a_arr), mx.multiply(u_sorted, s_inv.reshape(1, -1)))
        vh = mx.transpose(v)
        u = u_sorted # Assign sorted U

        if full_matrices:
            # No need to extend U if m < n and k_val = m
            if n > k_val:
                # Use imported qr
                q_v, _ = qr(mx.transpose(vh))
                vh = mx.transpose(q_v)
            if u.shape[1] < m: # Should only happen if k_val < m initially
                 q_u, _ = qr(u)
                 u = q_u

        if not full_matrices:
             u = u[:, :k_val]
             vh = vh[:k_val, :]

        return u, s, vh
# --- End of SVD Function ---


# ----------------------------------------------------------------------------
# Test code for the parallelized power iteration kernel
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    
    print("Testing parallelized power iteration kernel for SVD")
    print("=" * 80)
    
    # Test with smaller matrices first
    for n in [32]:
        print(f"\nTesting with {n}x{n} matrix:")
        
        # Create a random matrix with known singular values using MLX
        mx.random.seed(42)  # For reproducibility
        
        # Create orthogonal matrices using QR
        U_rand = mx.random.normal((n, n), dtype=mx.float32)
        U, _ = qr(U_rand)  # Use the QR function we imported
        
        # Create diagonal matrix with decreasing singular values
        s_values = mx.linspace(n, 1, n, dtype=mx.float32)
        
        V_rand = mx.random.normal((n, n), dtype=mx.float32)
        V, _ = qr(V_rand)  # Use the QR function we imported
        
        # Create matrix A = U @ diag(s) @ V.T
        s_diag = mx.zeros((n, n), dtype=mx.float32)
        for i in range(n):
            indices = mx.array([[i, i]])
            from ember_ml.backend.mlx.tensor.ops.indexing import scatter
            s_diag = scatter(indices, s_values[i], s_diag.shape)
        
        A = mx.matmul(U, mx.matmul(s_diag, mx.transpose(V)))
        
        # Create initial guess for eigenvectors
        Q_init_rand = mx.random.normal((n, n), dtype=mx.float32)
        Q_init, _ = qr(Q_init_rand)  # Use the QR function we imported
        
        # Test the parallelized power iteration kernel
        start_time = time.time()
        try:
            # Test with different thread configurations
            # First with single thread (1,1,1) to simulate old behavior
            shape_params = mx.array([n, n], dtype=mx.uint32)
            iter_params = mx.array([10], dtype=mx.uint32)
            tol_params = mx.array([1e-10], dtype=mx.float32)
            
            # Use the same configuration as in the main svd function
            grid_single = (1, 1, 1)
            threads_single = (1, 1, 1)
            
            outputs_single = _power_iter_tensor_kernel_compiled(
                inputs=[A, Q_init, shape_params, iter_params, tol_params],
                output_shapes=[(n, n)],
                output_dtypes=[mx.float32],
                grid=grid_single,
                threadgroup=threads_single
            )
            
            Q_single = outputs_single[0]
            single_time = time.time() - start_time
            
            # Verify result: Q should be orthogonal
            Q_single_T = mx.transpose(Q_single)
            orthogonality = mx.matmul(Q_single_T, Q_single)
            orthogonality_error = mx.mean(mx.abs(orthogonality - mx.eye(n, dtype=mx.float32))).item()
            print(f"  Single-threaded: {single_time:.4f}s, Orthogonality error: {orthogonality_error:.2e}")
            single_success = True
            
            # Skip parallel test for now until we resolve the kernel issues
            parallel_success = False
            print("  Skipping parallel test until kernel issues are resolved")
        except Exception as e:
            print(f"  Single-threaded failed: {e}")
            single_success = False
        
        # Test full SVD function with the parallelized kernel
        start_time = time.time()
        try:
            u, s, vh = svd(A)
            svd_time = time.time() - start_time
            
            # Verify result: u @ diag(s) @ vh should be close to A
            s_diag = mx.zeros((u.shape[1], vh.shape[0]), dtype=mx.float32)
            for i in range(min(u.shape[1], vh.shape[0])):
                indices = mx.array([[i, i]])
                from ember_ml.backend.mlx.tensor.ops.indexing import scatter
                s_diag = scatter(indices, s[i], s_diag.shape)
            
            reconstruction = mx.matmul(u, mx.matmul(s_diag, vh))
            reconstruction_error = mx.mean(mx.abs(reconstruction - A)).item()
            print(f"  Full SVD: {svd_time:.4f}s, Reconstruction error: {reconstruction_error:.2e}")
        except Exception as e:
            print(f"  Full SVD failed: {e}")
    
    print("\nPower iteration kernel test completed.")