# Content for ember_ml/backend/mlx/linearalg/eigen_ops.py
"""
MLX solver operations for ember_ml.

This module provides MLX implementations of matrix decomposition operations
like Cholesky and Eigendecomposition. QR has been moved to qr_ops.py.
SVD has been moved to svd_ops.py.
"""
from typing import Union, Tuple, Literal, List, Optional
import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.tensor import MLXTensor
# Import HPC for robust eigendecomposition
from .hpc16x8_ops import HPC16x8

dtype_obj = MLXDType()

def _update_array(arr: mx.array, indices: Union[int, Tuple[int, ...], slice], value: mx.array) -> mx.array:
    """Helper function for MLX array updates."""
    result = mx.array(arr)
    result[indices] = value
    return result

def _slice_indices_to_array(indices: Union[List[int], List[List[int]]]) -> mx.array:
    """Convert Python list indices to MLX array."""
    return mx.array(indices, dtype=mx.int32)

def is_spd(A: TensorLike) -> bool:
    """
    Check if matrix is symmetric positive definite using Cholesky.

    Args:
        A: Input matrix to check

    Returns:
        Boolean indicating if matrix is SPD
    """
    Tensor = MLXTensor()
    A_arr = Tensor.convert_to_tensor(A, dtype=dtype_obj.float32)

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
    A_arr = mx.array(A, dtype=dtype_obj.float32) # Ensure float32
    if len(A_arr.shape) != 2 or A_arr.shape[0] != A_arr.shape[1]:
         raise ValueError("Input must be a square matrix.")

    @mx.custom_function
    def _inner_impl(A_inner: mx.array) -> mx.array:
        source = """
        uint tid = thread_position_in_grid.x;
        uint n = A_shape[0];
        uint num_threads = threads_per_threadgroup.x;
        
        // Initialize output matrix in parallel (copy lower triangle, zero upper)
        for (uint idx = tid; idx < n * n; idx += num_threads) {
            uint i = idx / n;
            uint j = idx % n;
            if (j > i) out[idx] = 0.0f;
            else out[idx] = A[idx]; // Copy lower including diagonal
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        // Process columns sequentially, but parallelize work within each column
        for (uint j = 0; j < n; j++) {
            // Step 1: Compute diagonal element (reduction operation)
            threadgroup float partial_sums[32]; // Assuming max 32 threads per group
            
            // Each thread computes partial sum for diagonal
            float thread_sum = 0.0f;
            for (uint k = tid; k < j; k += num_threads) {
                float val = out[j*n + k];
                thread_sum += val * val;
            }
            
            // Reduce within SIMD group
            thread_sum = simd_sum(thread_sum);
            
            // First thread in each SIMD group writes to shared memory
            if (tid % 32 == 0 && tid / 32 < 32) {
                partial_sums[tid / 32] = thread_sum;
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Thread 0 combines results and updates diagonal
            if (tid == 0) {
                float diag_sum = 0.0f;
                for (uint i = 0; i < min(32u, (num_threads + 31) / 32); i++) {
                    diag_sum += partial_sums[i];
                }
                
                float diag_val = out[j*n + j] - diag_sum;
                if (diag_val <= 1e-10f) {
                    diag_val = 1e-10f;
                }
                out[j*n + j] = sqrt(diag_val);
            }
            
            threadgroup_barrier(mem_flags::mem_device);
            
            // Step 2: Update column j below diagonal in parallel
            float diag_elem = out[j*n + j];
            float inv_diag_elem = (diag_elem > 1e-10f) ? (1.0f / diag_elem) : 0.0f;
            
            for (uint i = j + 1 + tid; i < n; i += num_threads) {
                if (diag_elem <= 1e-10f) {
                    out[i*n + j] = 0.0f;
                } else {
                    // Each thread computes its own sum
                    float sum = 0.0f;
                    for (uint k = 0; k < j; k++) {
                        sum += out[i*n + k] * out[j*n + k];
                    }
                    out[i*n + j] = (out[i*n + j] - sum) * inv_diag_elem;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_device);
        }
        """
        kernel = mx.fast.metal_kernel(
            name="cholesky_kernel",
            input_names=["A"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True
        )
        grid = (256, 1, 1)
        threads = (256, 1, 1)
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
    Tensor = MLXTensor()
    A_arr = Tensor.convert_to_tensor(A, dtype=dtype_obj.float32)
    if len(A_arr.shape) != 2 or A_arr.shape[0] != A_arr.shape[1]:
         raise ValueError("Input must be a square matrix.")
    n = A_arr.shape[0]

    @mx.custom_function
    def _inner_impl(A_inner: mx.array, block_size_inner: int) -> mx.array:
        source = """
        uint thread_id = thread_position_in_grid.x;
        uint n = A_shape[0];
        uint block_size = block_param[0];
        uint num_blocks = (n + block_size - 1) / block_size;
        uint num_threads = threads_per_threadgroup.x;

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

                // Phase 1: Cholesky on diagonal block (parallel implementation)
                for (uint j = k_start; j < k_end; ++j) {
                    // Step 1: Compute diagonal element (reduction operation)
                    threadgroup float partial_sums[32]; // Assuming max 32 threads per group
                    
                    // Each thread computes partial sum for diagonal
                    float thread_sum = 0.0f;
                    for (uint p = k_start + thread_id % (j - k_start + 1); p < j; p += max(1u, min(num_threads, j - k_start))) {
                        float val = out[j*n + p];
                        thread_sum += val * val;
                    }
                    
                    // Reduce within SIMD group
                    thread_sum = simd_sum(thread_sum);
                    
                    // First thread in each SIMD group writes to shared memory
                    if (thread_id % 32 == 0 && thread_id / 32 < 32) {
                        partial_sums[thread_id / 32] = thread_sum;
                    }
                    
                    threadgroup_barrier(mem_flags::mem_device);
                    
                    // Thread 0 combines results and updates diagonal
                    if (thread_id == 0) {
                        float diag_sum = 0.0f;
                        for (uint i = 0; i < min(32u, (num_threads + 31) / 32); i++) {
                            diag_sum += partial_sums[i];
                        }
                        
                        float diag_val = out[j*n + j] - diag_sum;
                        if (diag_val <= 1e-10f) diag_val = 1e-10f; // Clamp or error
                        out[j*n + j] = sqrt(diag_val);
                    }
                    
                    threadgroup_barrier(mem_flags::mem_device);
                    
                    // Step 2: Update column j below diagonal in parallel
                    float diag_elem = out[j*n + j];
                    float inv_diag = (diag_elem > 1e-10f) ? (1.0f / diag_elem) : 0.0f;
                    
                    for (uint i = j + 1 + thread_id % (k_end - j - 1 + 1); i < k_end; i += max(1u, min(num_threads, k_end - j - 1))) {
                        if (diag_elem <= 1e-10f) {
                            out[i*n + j] = 0.0f;
                        } else {
                            // Each thread computes its own sum
                            float sum = 0.0f;
                            for (uint p = k_start; p < j; ++p) {
                                sum += out[i*n + p] * out[j*n + p];
                            }
                            out[i*n + j] = (out[i*n + j] - sum) * inv_diag;
                        }
                    }
                    
                    threadgroup_barrier(mem_flags::mem_device);
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
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
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
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    if len(a_array.shape) != 2 or a_array.shape[0] != a_array.shape[1]:
         raise ValueError("Input must be a square matrix.")
    n = a_array.shape[0]

    try:
        is_metal = mx.default_device().type == mx.DeviceType.gpu
    except Exception:
        is_metal = False

    try:
        if not is_metal or n < 32:
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
    Tensor = MLXTensor()
    L_arr = Tensor.convert_to_tensor(L, dtype=dtype_obj.float32)
    b_arr = Tensor.convert_to_tensor(b, dtype=L_arr.dtype)

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
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)

    L = cholesky(a_array) # Let cholesky handle SPD check and errors
    n = L.shape[0]
    identity = mx.eye(n, dtype=a_array.dtype)

    # Solve L Y = I
    Y = _solve_triangular(L, identity, upper=False)
    # Solve L^T X = Y
    L_T = mx.transpose(L)
    X = _solve_triangular(L_T, Y, upper=True)

    return X


# --- QR Code Removed (Moved to qr_ops.py) ---


def eigvals(a: TensorLike) -> mx.array:
    """
    Compute the eigenvalues of a square matrix using the robust HPC implementation.

    Args:
        a: Input square matrix

    Returns:
        Eigenvalues of the matrix (potentially complex)
    """
    Tensor = MLXTensor()
    tensor_obj = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    if len(tensor_obj.shape) != 2 or tensor_obj.shape[0] != tensor_obj.shape[1]:
         raise ValueError("Input must be a square matrix.")

    try:
        # Use the consolidated eig function which leverages HPC internally
        eigenvalues, _ = eig(tensor_obj)
        return eigenvalues
    except Exception as e:
        raise RuntimeError(f"Eigendecomposition for eigenvalues failed: {e}")


def eig(a: TensorLike) -> Tuple[mx.array, mx.array]:
    """
    Compute eigenvalues and right eigenvectors of a general square matrix.
    Uses the HPC16x8 implementation internally for robustness. This handles
    both symmetric (using mx.linalg.eigh with descending sort) and
    non-symmetric cases (using power iteration with extended precision).

    Args:
        a: Input matrix (must be square)

    Returns:
        Tuple of (eigenvalues, right_eigenvectors).
        Eigenvalues are sorted in descending order.
        May be complex for non-symmetric matrices.

    Raises:
        ValueError: If input is not a square matrix.
        RuntimeError: If underlying HPC or MLX operation fails.
    """
    Tensor = MLXTensor()
    a_array = Tensor.convert_to_tensor(a, dtype=dtype_obj.float32)
    if len(a_array.shape) != 2 or a_array.shape[0] != a_array.shape[1]:
         raise ValueError("Input must be a square matrix.")

    try:
        print("Using HPC16x8 implementation for eigendecomposition.")
        # Instantiate HPC object and call its eig method
        hpc_obj = HPC16x8.from_array(a_array)
        w, v = hpc_obj.eig()
        return w, v
    except Exception as e:
        # Re-raise error from HPC/MLX functions
        raise RuntimeError(f"Eigendecomposition failed: {e}")


def eigh(matrix: TensorLike) -> tuple[mx.array, mx.array]:
    """Compute eigenvalues and eigenvectors of a Hermitian/symmetric matrix
    using mx.linalg.eigh. Eigenvalues are sorted in descending order.

    Args:
        matrix: Square matrix (..., M, M) that is Hermitian/symmetric

    Returns:
        Tuple of:
            - eigenvalues (..., M) in descending order (real)
            - eigenvectors (..., M, M)

    Raises:
        ValueError: If input is not a square matrix or not symmetric/Hermitian.
        RuntimeError: If mx.linalg.eigh fails.
    """
    Tensor = MLXTensor()
    matrix_arr = Tensor.convert_to_tensor(matrix, dtype=dtype_obj.float32)
    if len(matrix_arr.shape) < 2 or matrix_arr.shape[-1] != matrix_arr.shape[-2]:
         raise ValueError("Input must be a square matrix (or batch of square matrices).")

    # Verify symmetry
    diff = mx.subtract(matrix_arr, mx.transpose(matrix_arr, axes=(-2, -1)))
    abs_diff = mx.abs(diff)
    if not mx.all(mx.less(abs_diff, mx.array(1e-5))).item():
        raise ValueError("Input matrix must be symmetric/Hermitian for eigh.")

    try:
        print("Using native mx.linalg.eigh and sorting results descending.")
        eigenvals, eigenvecs = mx.linalg.eigh(matrix_arr)
        # Sort descending
        sort_idx = mx.argsort(-eigenvals, axis=-1)
        # Apply sorting to eigenvalues and eigenvectors
        # Need to gather along the last axis for eigenvalues and second-to-last for eigenvectors
        sorted_eigenvals = mx.take_along_axis(eigenvals, sort_idx[..., None], axis=-1).squeeze(-1) # Ensure correct shape
        # For eigenvectors, we need to sort the columns (last dimension) based on eigenvalue order
        # Expand sort_idx to match eigenvector dimensions for gathering
        expanded_sort_idx = mx.expand_dims(sort_idx, axis=-2) # Shape becomes (..., 1, M)
        repeated_sort_idx = mx.repeat(expanded_sort_idx, repeats=matrix_arr.shape[-1], axis=-2) # Shape becomes (..., M, M)
        sorted_eigenvecs = mx.take_along_axis(eigenvecs, repeated_sort_idx, axis=-1)

        return sorted_eigenvals, sorted_eigenvecs
    except Exception as e:
         raise RuntimeError(f"mx.linalg.eigh failed: {e}")

# Removed _add_double_single
# Removed _standard_qr
# Removed qr
# Removed _complete_orthogonal_basis
# Removed _eigsh_power_iteration


# ----------------------------------------------------------------------------
# Test code for the parallelized Cholesky implementations
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    
    print("Testing parallelized Cholesky decomposition implementations")
    print("=" * 80)
    
    # Test matrices of different sizes
    for n in [32, 128, 512]:
        print(f"\nTesting with {n}x{n} matrix:")
        
        # Create a random SPD matrix using MLX
        mx.random.seed(42)  # For reproducibility
        A_rand = mx.random.normal((n, n), dtype=mx.float32)
        A_sym = mx.matmul(A_rand, mx.transpose(A_rand))  # Make symmetric
        A = A_sym + n * mx.eye(n, dtype=mx.float32)  # Make positive definite
        
        # Test single-threaded Metal implementation
        start_time = time.time()
        try:
            L_single = mlx_cholesky_single_thread(A)
            single_time = time.time() - start_time
            
            # Verify result: L @ L.T should be close to A
            L_T = mx.transpose(L_single)
            reconstruction = mx.matmul(L_single, L_T)
            reconstruction_error = mx.mean(mx.abs(reconstruction - A)).item()
            print(f"  Single-threaded Metal: {single_time:.4f}s, Reconstruction error: {reconstruction_error:.2e}")
            single_success = True
        except Exception as e:
            print(f"  Single-threaded Metal failed: {e}")
            single_success = False
        
        # Test block-based Metal implementation
        start_time = time.time()
        try:
            L_block = mlx_cholesky_block_based(A)
            block_time = time.time() - start_time
            
            # Verify result: L @ L.T should be close to A
            L_T = mx.transpose(L_block)
            reconstruction = mx.matmul(L_block, L_T)
            reconstruction_error = mx.mean(mx.abs(reconstruction - A)).item()
            print(f"  Block-based Metal: {block_time:.4f}s, Reconstruction error: {reconstruction_error:.2e}")
            block_success = True
        except Exception as e:
            print(f"  Block-based Metal failed: {e}")
            block_success = False
        
        # Test standard implementation for comparison
        start_time = time.time()
        try:
            L_std = cholesky_standard(A)
            std_time = time.time() - start_time
            
            # Verify result: L @ L.T should be close to A
            L_T = mx.transpose(L_std)
            reconstruction = mx.matmul(L_std, L_T)
            reconstruction_error = mx.mean(mx.abs(reconstruction - A)).item()
            print(f"  Standard Python: {std_time:.4f}s, Reconstruction error: {reconstruction_error:.2e}")
            std_success = True
        except Exception as e:
            print(f"  Standard Python failed: {e}")
            std_success = False
        
        # Compare implementations if all succeeded
        if single_success and block_success and std_success:
            # Compare results between implementations
            diff_single_std = mx.mean(mx.abs(L_single - L_std)).item()
            diff_block_std = mx.mean(mx.abs(L_block - L_std)).item()
            diff_single_block = mx.mean(mx.abs(L_single - L_block)).item()
            
            print(f"  Difference between single-threaded and standard: {diff_single_std:.2e}")
            print(f"  Difference between block-based and standard: {diff_block_std:.2e}")
            print(f"  Difference between single-threaded and block-based: {diff_single_block:.2e}")
            
            # Report speedups
            if std_time > 0:
                print(f"  Speedup single-threaded vs standard: {std_time / single_time:.2f}x")
            if std_time > 0:
                print(f"  Speedup block-based vs standard: {std_time / block_time:.2f}x")
            if single_time > 0:
                print(f"  Speedup block-based vs single-threaded: {single_time / block_time:.2f}x")
    
    print("\nCholesky decomposition test completed.")
