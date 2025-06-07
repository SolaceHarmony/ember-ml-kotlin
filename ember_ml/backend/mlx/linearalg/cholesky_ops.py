# Content for ember_ml/backend/mlx/linearalg/decomp_ops.py (QR and HPC fallbacks removed)
"""
MLX solver operations for ember_ml.

This module provides MLX implementations of matrix decomposition operations
like Cholesky and Eigendecomposition. QR has been moved to qr_ops.py.
SVD has been moved to svd_ops.py.
"""
from typing import Union, Tuple, Literal, List, Optional
import mlx.core as mx

from ember_ml.backend.mlx.types import TensorLike

def _update_array(arr: mx.array, indices: Union[int, Tuple[int, ...], slice], value: mx.array) -> mx.array:
    """Helper function for MLX array updates."""
    result = mx.array(arr)
    result[indices] = value
    return result
def is_spd(A: TensorLike) -> bool:
    """
    Check if matrix is symmetric positive definite using Cholesky.

    Args:
        A: Input matrix to check

    Returns:
        Boolean indicating if matrix is SPD
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    A_arr = Tensor.convert_to_tensor(A, dtype=mx.float32)

    if len(A_arr.shape) != 2 or A_arr.shape[0] != A_arr.shape[1]:
        return False # Not square

    # Check for symmetry
    diff = mx.subtract(A_arr, mx.transpose(A_arr))
    abs_diff = mx.abs(diff)
    # Use a tolerance appropriate for float32
    is_symmetric = mx.all(mx.less(abs_diff, mx.array(1e-5))).item()
    if not is_symmetric:
        return False

    # Attempt Cholesky decomposition
    try:
        # Use the appropriate Cholesky implementation (which might have internal fallbacks)
        cholesky(A_arr)
        return True
    except (ValueError, RuntimeError): # Catch errors indicating non-SPD
        return False
def mlx_cholesky_single_thread(A: TensorLike) -> mx.array:
    """
    Stable implementation of Cholesky decomposition using single-threaded Metal approach.

    Args:
        A: Input positive definite matrix

    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    A_arr = Tensor.convert_to_tensor(A, dtype=mx.float32)
    if len(A_arr.shape) != 2 or A_arr.shape[0] != A_arr.shape[1]:
         raise ValueError("Input must be a square matrix.")

    @mx.custom_function
    def _inner_impl(A_inner: mx.array) -> mx.array:
        source = """
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;

        kernel void cholesky_kernel(
            const device float *A [[buffer(0)]],
            device float *out [[buffer(1)]],
            constant uint *A_shape [[buffer(2)]],
            uint thread_position_in_grid [[thread_position_in_grid]])
        {
            if (thread_position_in_grid.x == 0) {
                uint n = A_shape[0];
                // Initialize output matrix (copy lower triangle, zero upper)
                for(uint i=0; i<n; ++i) {
                    for(uint j=0; j<n; ++j) {
                        if (j > i) out[i*n + j] = 0.0f;
                        else out[i*n + j] = A[i*n + j]; // Copy lower including diagonal
                    }
                }

                for (uint j = 0; j < n; j++) {
                    float diag_sum = 0.0f;
                    for (uint k = 0; k < j; k++) {
                        float val = out[j*n + k];
                        diag_sum += val * val;
                    }
                    float diag_val = out[j*n + j] - diag_sum; // Use value from out
                    if (diag_val <= 1e-10f) {
                         // Raise error or handle non-SPD case appropriately
                         // For now, clamp like before, but ideally signal failure
                         diag_val = 1e-10f;
                    }
                    out[j*n + j] = sqrt(diag_val);

                    float diag_elem = out[j*n + j];
                    if (diag_elem <= 1e-10f) {
                         for (uint i = j+1; i < n; i++) out[i*n + j] = 0.0f;
                    } else {
                        float inv_diag_elem = 1.0f / diag_elem;
                        for (uint i = j+1; i < n; i++) {
                            float sum = 0.0f;
                            for (uint k = 0; k < j; k++) {
                                sum += out[i*n + k] * out[j*n + k];
                            }
                            // Use value from out for A[i,j] equivalent
                            out[i*n + j] = (out[i*n + j] - sum) * inv_diag_elem;
                        }
                    }
                }
            }
        }
        """
        kernel = mx.fast.metal_kernel(
            name="cholesky_kernel",
            input_names=["A"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True
        )
        grid = (1, 1, 1)
        threads = (1, 1, 1)
        result = kernel(
            inputs=[A_inner],
            output_shapes=[A_inner.shape],
            output_dtypes=[A_inner.dtype],
            grid=grid,
            threadgroup=threads
        ) # type: ignore
        return result[0]
    return _inner_impl(A_arr) # type: ignore

def mlx_cholesky_block_based(A: TensorLike, block_size: int = 16) -> mx.array:
    """
    Block-based Cholesky implementation using Metal.

    Args:
        A: Input positive definite matrix
        block_size: Size of blocks for tiled computation (default: 16)

    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    A_arr = Tensor.convert_to_tensor(A, dtype=mx.float32)
    if len(A_arr.shape) != 2 or A_arr.shape[0] != A_arr.shape[1]:
         raise ValueError("Input must be a square matrix.")
    n = A_arr.shape[0]

    @mx.custom_function
    def _inner_impl(A_inner: mx.array, block_size_inner: int) -> mx.array:
        source = """
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;

        kernel void block_cholesky_kernel(
            const device float *A [[buffer(0)]],
            device float *out [[buffer(1)]],
            constant uint *A_shape [[buffer(2)]],
            constant uint *block_param [[buffer(3)]],
            constant uint *thread_count [[buffer(4)]],
            uint thread_position_in_grid [[thread_position_in_grid]],
            uint threads_per_threadgroup [[threads_per_threadgroup]])
        {
            uint thread_id = thread_position_in_grid;
            uint n = A_shape[0];
            uint block_size = block_param[0];
            uint num_blocks = (n + block_size - 1) / block_size;
            uint num_threads = threads_per_threadgroup;

            // Initialize out matrix (copy lower triangle of A, zero upper)
            for (uint idx = thread_id; idx < n * n; idx += num_threads) {
                 uint i = idx / n;
                 uint j = idx % n;
                 out[idx] = (j > i) ? 0.0f : A[idx];
            }
            threadgroup_barrier(mem_flags::mem_device);

            for (uint k_block = 0; k_block < num_blocks; k_block++) {
                uint k_start = k_block * block_size;
                uint k_end = min(k_start + block_size, n);

                // Phase 1: Cholesky on diagonal block (single thread for stability)
                if (thread_id == 0) {
                    for (uint j = k_start; j < k_end; ++j) {
                        float diag_sum = 0.0f;
                        for (uint p = k_start; p < j; ++p) diag_sum += out[j*n + p] * out[j*n + p];
                        float diag_val = out[j*n + j] - diag_sum;
                        if (diag_val <= 1e-10f) diag_val = 1e-10f; // Clamp or error
                        out[j*n + j] = sqrt(diag_val);
                        float inv_diag = (out[j*n + j] > 1e-10f) ? (1.0f / out[j*n + j]) : 0.0f;
                        for (uint i = j + 1; i < k_end; ++i) {
                            float sum = 0.0f;
                            for (uint p = k_start; p < j; ++p) sum += out[i*n + p] * out[j*n + p];
                            out[i*n + j] = (out[i*n + j] - sum) * inv_diag;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                // Phase 2: Triangular solve for off-diagonal blocks below
                uint num_row_blocks = (n - k_end + block_size - 1) / block_size;
                for (uint i_block_idx = thread_id; i_block_idx < num_row_blocks; i_block_idx += num_threads) {
                     uint i_start = k_end + i_block_idx * block_size;
                     uint i_end = min(i_start + block_size, n);
                     for (uint j = k_start; j < k_end; ++j) {
                          float l_jj = out[j*n + j];
                          float inv_l_jj = (l_jj > 1e-10f) ? (1.0f / l_jj) : 0.0f;
                          for (uint i = i_start; i < i_end; ++i) {
                               float sum = 0.0f;
                               for (uint p = k_start; p < j; ++p) sum += out[i*n + p] * out[j*n + p];
                               out[i*n + j] = (out[i*n + j] - sum) * inv_l_jj;
                          }
                     }
                }
                threadgroup_barrier(mem_flags::mem_device);

                // Phase 3: Symmetric rank-k update for remaining submatrix
                uint num_rem_blocks = (n - k_end + block_size - 1) / block_size;
                for (uint ij_block_idx = thread_id; ij_block_idx < num_rem_blocks * (num_rem_blocks + 1) / 2; ij_block_idx += num_threads) {
                     uint i_block_rel = floor((-1.0f + sqrt(1.0f + 8.0f * ij_block_idx)) / 2.0f);
                     uint j_block_rel = ij_block_idx - i_block_rel * (i_block_rel + 1) / 2;
                     uint i_start = k_end + i_block_rel * block_size;
                     uint i_end = min(i_start + block_size, n);
                     uint j_start = k_end + j_block_rel * block_size;
                     uint j_end = min(j_start + block_size, n);

                     for (uint i = i_start; i < i_end; ++i) {
                          for (uint j = j_start; j < min(j_end, i + 1); ++j) {
                               float update_sum = 0.0f;
                               for (uint p = k_start; p < k_end; ++p) update_sum += out[i*n + p] * out[j*n + p];
                               out[i*n + j] -= update_sum;
                          }
                     }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
        """
        kernel = mx.fast.metal_kernel(
            name="block_cholesky_kernel",
            input_names=["A", "block_param", "thread_count"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True
        )
        num_threads = 256
        grid = (num_threads, 1, 1)
        threads = (num_threads, 1, 1)
        block_param = mx.array([block_size_inner], dtype=mx.uint32)
        thread_count = mx.array([num_threads], dtype=mx.uint32)
        result = kernel(
            inputs=[A_inner, block_param, thread_count], # Pass A_inner
            output_shapes=[A_inner.shape],
            output_dtypes=[A_inner.dtype],
            grid=grid,
            threadgroup=threads
        ) # type: ignore
        return result[0]
    return _inner_impl(A_arr, block_size) # type: ignore

def cholesky_standard(a: TensorLike) -> mx.array:
    """
    Standard Python implementation of Cholesky decomposition.

    Args:
        a: Input positive definite matrix

    Returns:
        Lower triangular matrix L such that L @ L.T = A
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=mx.float32)
    n = a_array.shape[0]
    if len(a_array.shape) != 2 or a_array.shape[0] != a_array.shape[1]:
         raise ValueError("Input must be a square matrix.")

    l = mx.zeros((n, n), dtype=a_array.dtype)
    for i in range(n):
        for j in range(i + 1):
            s = mx.array(0.0, dtype=a_array.dtype)
            for k in range(j):
                s = mx.add(s, mx.multiply(l[i, k], l[j, k]))
            if i == j:
                diag_val = mx.subtract(a_array[i, i], s)
                if mx.less_equal(diag_val, mx.array(1e-10)):
                    raise ValueError("Matrix is not positive definite")
                l = _update_array(l, (i, i), mx.sqrt(diag_val))
            else:
                diag_l_jj = l[j, j]
                if mx.greater(mx.abs(diag_l_jj), mx.array(1e-10)):
                    val = mx.divide(mx.subtract(a_array[i, j], s), diag_l_jj)
                    l = _update_array(l, (i, j), val)
                else:
                     # Should not happen if matrix is SPD and previous diagonals were > 0
                     raise RuntimeError("Division by zero encountered in Cholesky - matrix might be ill-conditioned or not SPD.")
    return l

def cholesky(a: TensorLike) -> mx.array:
    """
    Compute the Cholesky decomposition of a positive definite matrix.
    Selects implementation based on size and device (Metal preferred).

    Args:
        a: Input positive definite matrix

    Returns:
        Lower triangular matrix L such that L @ L.T = A

    Raises:
        ValueError: If matrix is not square or not positive definite.
        RuntimeError: If Metal kernels fail.
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=mx.float32)
    if len(a_array.shape) != 2 or a_array.shape[0] != a_array.shape[1]:
         raise ValueError("Input must be a square matrix.")
    n = a_array.shape[0]

    try:
        if n < 32:
            print("Using standard Cholesky implementation.")
            return cholesky_standard(a_array)
        elif n < 128:
            print("Using single-threaded Metal Cholesky implementation.")
            return mlx_cholesky_single_thread(a_array)
        else:
            block_size = min(32, max(16, n // 8))
            print(f"Using block-based Metal Cholesky implementation (block_size={block_size}).")
            return mlx_cholesky_block_based(a_array, block_size=block_size)
    except Exception as e:
        # Re-raise errors from underlying implementations
        raise RuntimeError(f"Cholesky decomposition failed: {e}")


def _solve_triangular(L: TensorLike, b: TensorLike, upper: bool = False) -> mx.array:
    """
    Solve a triangular system of equations Lx = b using MLX.

    Args:
        L: Lower or Upper triangular matrix (M x M)
        b: Right-hand side vector/matrix (M x N)
        upper: If True, L is assumed upper triangular

    Returns:
        Solution x to Lx = b (M x N)

    Raises:
        ValueError: If dimensions are incompatible.
        RuntimeError: If matrix is singular (zero on diagonal).
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    L_arr = Tensor.convert_to_tensor(L, dtype=mx.float32)
    b_arr = Tensor.convert_to_tensor(b, dtype=mx.float32)

    if len(L_arr.shape) != 2 or L_arr.shape[0] != L_arr.shape[1]:
        raise ValueError("L must be a square matrix.")
    if L_arr.shape[0] != b_arr.shape[0]:
        raise ValueError("L and b must have compatible dimensions.")

    n = L_arr.shape[0]
    is_vector = len(b_arr.shape) == 1
    if is_vector:
        b_arr = b_arr.reshape(-1, 1)
    x = mx.zeros_like(b_arr)
    num_rhs = b_arr.shape[1]

    if not upper: # Forward substitution
        for k in range(num_rhs):
            xk = mx.zeros((n,), dtype=x.dtype)
            for i in range(n):
                rhs = b_arr[i, k]
                sum_val = mx.array(0.0, dtype=rhs.dtype)
                for j in range(i):
                    sum_val = mx.add(sum_val, mx.multiply(L_arr[i, j], xk[j]))
                diag_L_ii = L_arr[i, i]
                if mx.greater(mx.abs(diag_L_ii), mx.array(1e-10)):
                     xk = _update_array(xk, i, mx.divide(mx.subtract(rhs, sum_val), diag_L_ii))
                else:
                     raise RuntimeError(f"Matrix is singular (zero on diagonal element {i}) during forward substitution.")
            x = _update_array(x, (slice(None), k), xk)
    else: # Back substitution
         for k in range(num_rhs):
            xk = mx.zeros((n,), dtype=x.dtype)
            for i in range(n - 1, -1, -1):
                rhs = b_arr[i, k]
                sum_val = mx.array(0.0, dtype=rhs.dtype)
                for j in range(i + 1, n):
                    sum_val = mx.add(sum_val, mx.multiply(L_arr[i, j], xk[j]))
                diag_U_ii = L_arr[i, i]
                if mx.greater(mx.abs(diag_U_ii), mx.array(1e-10)):
                    xk = _update_array(xk, i, mx.divide(mx.subtract(rhs, sum_val), diag_U_ii))
                else:
                    raise RuntimeError(f"Matrix is singular (zero on diagonal element {i}) during back substitution.")
            x = _update_array(x, (slice(None), k), xk)

    return x.flatten() if is_vector else x

def cholesky_inv(a: TensorLike) -> mx.array:
    """
    Compute the inverse of a symmetric positive definite matrix using Cholesky.

    Args:
        a: Input symmetric positive definite matrix

    Returns:
        Inverse of input matrix

    Raises:
        ValueError: If matrix is not positive definite or Cholesky fails.
        RuntimeError: If triangular solve fails (e.g., due to singularity).
    """
    from ember_ml.backend.mlx.tensor import MLXTensor
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=mx.float32)

    L = cholesky(a_array) # Let cholesky handle SPD check and errors
    n = L.shape[0]
    identity = mx.eye(n, dtype=a_array.dtype)

    # Solve L Y = I
    Y = _solve_triangular(L, identity, upper=False)
    # Solve L^T X = Y
    L_T = mx.transpose(L)
    X = _solve_triangular(L_T, Y, upper=True)

    return X