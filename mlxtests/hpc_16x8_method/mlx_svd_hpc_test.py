"""
Test SVD implementation using HPC16x8 eigendecomposition for MLX backend.
"""
from typing import Union, Tuple
import mlx.core as mx
import sys
import os

# Add project root to path to allow importing ember_ml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from metal_kernel_method.svd_metal import svd
from ember_ml.backend.mlx.types import TensorLike
from ember_ml.backend.mlx.linearalg.hpc16x8_ops import HPC16x8
from ember_ml.backend.mlx.linearalg.eigen_ops import _update_array, _convert_to_float32

def svd_hpc(a: TensorLike, full_matrices: bool = True, compute_uv: bool = True) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array]]:
    """
    Compute SVD using HPC16x8 eigendecomposition of A^T A or A A^T.

    Args:
        a: Input matrix (TensorLike).
        full_matrices: If True, compute full U and Vh.
        compute_uv: If True, compute U and Vh.

    Returns:
        U, S, Vh if compute_uv is True, otherwise S.
    """
    a_arr = mx.array(a, dtype=mx.float32) # Ensure float32 mx.array
    m, n = a_arr.shape
    k = min(m, n)

    if m >= n:
        # Compute A^T A
        ata = mx.matmul(mx.transpose(a_arr), a_arr)
        # Use HPC eigendecomposition
        matrix_hpc = HPC16x8.from_array(ata)
        eigenvalues_hpc, v_hpc = matrix_hpc.eig() # Eigenvalues might not be sorted descending

        # Convert results back to standard float32 mx.array
        eigenvalues = _convert_to_float32(eigenvalues_hpc)
        v = _convert_to_float32(v_hpc)

        # Singular values are sqrt of eigenvalues of A^T A
        s_squared = mx.maximum(eigenvalues, 0.) # Ensure non-negative before sqrt
        s = mx.sqrt(s_squared)

        # Sort singular values and corresponding vectors (descending)
        sort_indices = mx.argsort(-s)
        s = s[sort_indices][:k]
        v = v[:, sort_indices][:, :k] # V are the eigenvectors of A^T A

        if not compute_uv:
            return s

        # Compute U = A V S^-1 (handle division by zero)
        s_inv = mx.where(s > 1e-10, mx.reciprocal(s), 0.0)
        u = mx.matmul(a_arr, v * s_inv.reshape(1, -1)) # Equivalent to A @ V @ diag(S)^-1

        # Complete basis if needed
        if full_matrices:
            if m > k:
                 u_hpc = HPC16x8.from_array(u)
                 u = u_hpc.complete_basis().to_float32()
                 # Ensure shape is (m, m)
                 if u.shape[1] < m:
                     padding = mx.zeros((m, m - u.shape[1]), dtype=u.dtype)
                     u = mx.concatenate([u, padding], axis=1)

            if n > k: # This case should ideally not happen if m >= n and we took top k
                 v_hpc = HPC16x8.from_array(v)
                 v = v_hpc.complete_basis().to_float32()
                 if v.shape[1] < n:
                     padding = mx.zeros((n, n - v.shape[1]), dtype=v.dtype)
                     v = mx.concatenate([v, padding], axis=1)


        return u, s, mx.transpose(v)

    else: # m < n
        # Compute A A^T
        aat = mx.matmul(a_arr, mx.transpose(a_arr))
        # Use HPC eigendecomposition
        matrix_hpc = HPC16x8.from_array(aat)
        eigenvalues_hpc, u_hpc = matrix_hpc.eig()

        # Convert results back to standard float32 mx.array
        eigenvalues = _convert_to_float32(eigenvalues_hpc)
        u = _convert_to_float32(u_hpc)

        # Singular values are sqrt of eigenvalues of A A^T
        s_squared = mx.maximum(eigenvalues, 0.)
        s = mx.sqrt(s_squared)

        # Sort singular values and corresponding vectors (descending)
        sort_indices = mx.argsort(-s)
        s = s[sort_indices][:k]
        u = u[:, sort_indices][:, :k] # U are the eigenvectors of A A^T

        if not compute_uv:
            return s

        # Compute V = A^T U S^-1 (handle division by zero)
        s_inv = mx.where(s > 1e-10, mx.reciprocal(s), 0.0)
        v = mx.matmul(mx.transpose(a_arr), u * s_inv.reshape(1, -1))

        # Complete basis if needed
        if full_matrices:
            if n > k:
                 v_hpc = HPC16x8.from_array(v)
                 v = v_hpc.complete_basis().to_float32()
                 if v.shape[1] < n:
                     padding = mx.zeros((n, n - v.shape[1]), dtype=v.dtype)
                     v = mx.concatenate([v, padding], axis=1)

            if m > k: # This case should ideally not happen if m < n and we took top k
                 u_hpc = HPC16x8.from_array(u)
                 u = u_hpc.complete_basis().to_float32()
                 if u.shape[1] < m:
                     padding = mx.zeros((m, m - u.shape[1]), dtype=u.dtype)
                     u = mx.concatenate([u, padding], axis=1)

        return u, s, mx.transpose(v)


if __name__ == "__main__":
    print("Testing SVD via HPC eigendecomposition...")
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    print("Input Matrix A:\\n", A)

    try:
        U, S, Vh = svd_hpc(A, full_matrices=True, compute_uv=True)
        print("\\nResult (HPC):")
        print("U:\\n", U)
        print("S:\\n", S)
        print("Vh:\\n", Vh)

        # Reconstruction Check
        A_rec = mx.matmul(mx.matmul(U, mx.diag(S)), Vh)
        print("\\nReconstruction Check (HPC):")
        print("Reconstructed A:\\n", A_rec)
        # Use a tighter tolerance to see if HPC helps
        print("Close (atol=1e-6)?:", mx.allclose(A, A_rec, atol=1e-6))
        print("Close (atol=1e-5)?:", mx.allclose(A, A_rec, atol=1e-5))

    except Exception as e:
        print(f"\\nError during HPC SVD test: {e}")
        import traceback
        traceback.print_exc()

    # Compare with standard precision SVD from decomp_ops for reference
    try:
        # Temporarily add decomp_ops path
        sys.path.insert(0, os.path.join(project_root, 'ember_ml/backend/mlx/linearalg'))

        U_metal, S_metal, Vh_metal = svd(A, full_matrices=True, compute_uv=True)
        print("\\nResult (Metal Kernel SVD for comparison):")
        print("U:\\n", U_metal)
        print("S:\\n", S_metal)
        print("Vh:\\n", Vh_metal)
        A_rec_metal = mx.matmul(mx.matmul(U_metal, mx.diag(S_metal)), Vh_metal)
        print("\\nReconstruction Check (Metal Kernel SVD):")
        print("Reconstructed A:\\n", A_rec_metal)
        print("Close (atol=1e-3)?:", mx.allclose(A, A_rec_metal, atol=1e-3))

    except ImportError:
         print("\\nCould not import standard SVD for comparison.")
    except Exception as e:
        print(f"\\nError during standard SVD comparison: {e}")