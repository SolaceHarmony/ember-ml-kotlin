"""
MLX solver operations for ember_ml.

This module provides MLX implementations of solver operations.
"""

import mlx.core as mx
from typing import Literal

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike

dtype_obj = MLXDType()


class MLXLinearAlgOps:
    """MLX implementation of solver operations."""
    
    def solve(self, a : TensorLike, b : TensorLike):
        """Solve a linear system of equations Ax = b for x."""
        from ember_ml.backend.mlx.linearalg.ops import solve as solve_func
        return solve_func(a, b)
    
    def inv(self, a: TensorLike):
        """Compute the inverse of a square matrix."""
        from ember_ml.backend.mlx.linearalg.ops import inv as inv_func
        return inv_func(a)
    
    def svd(self, a: TensorLike, full_matrices=True, compute_uv=True):
        """Compute the singular value decomposition of a matrix."""
        from ember_ml.backend.mlx.linearalg.ops import svd as svd_func
        return svd_func(a, full_matrices=full_matrices, compute_uv=compute_uv)
    
    def norm(self, x: TensorLike, ord=None, axis=None, keepdims=False):
        """Compute the matrix or vector norm."""
        from ember_ml.backend.mlx.linearalg.ops import norm as norm_func
        return norm_func(x, ord=ord, axis=axis, keepdims=keepdims)
    
    def eigvals(self, a: TensorLike):
        """Compute the eigenvalues of a square matrix."""
        from ember_ml.backend.mlx.linearalg.ops import eigvals as eigvals_func
        return eigvals_func(a)
    
    def lstsq(self, a: TensorLike, b: TensorLike, rcond=None):
        """Compute the least-squares solution to a linear matrix equation."""
        from ember_ml.backend.mlx.linearalg.ops import lstsq as lstsq_func
        return lstsq_func(a, b, rcond=rcond)
    
    def det(self, a: TensorLike):
        """Compute the determinant of a square matrix."""
        from ember_ml.backend.mlx.linearalg.ops import det as det_func
        return det_func(a)
    
    def norm(self, x: TensorLike, ord=None, axis=None, keepdims=False):
        """Compute the matrix or vector norm."""
        from ember_ml.backend.mlx.linearalg.ops import norm as norm_func
        return norm_func(x, ord=ord, axis=axis, keepdims=keepdims)
    
    def qr(self, a: TensorLike, mode: Literal['reduced','complete','r','raw'] ='reduced'):
        """Compute the QR decomposition of a matrix."""
        from ember_ml.backend.mlx.linearalg.ops import qr as qr_func
        return qr_func(a, mode=mode)
    
    def cholesky(self, a: TensorLike):
        """Compute the Cholesky decomposition of a matrix."""
        from ember_ml.backend.mlx.linearalg.ops import cholesky as cholesky_func
        return cholesky_func(a)
    
    def lstsq(self, a: TensorLike, b: TensorLike, rcond=None):
        """Compute the least-squares solution to a linear matrix equation."""
        from ember_ml.backend.mlx.linearalg.ops import lstsq as lstsq_func
        return lstsq_func(a, b, rcond=rcond)
    
    def diag(self, x: TensorLike, k: int = 0):
        """
        Extract a diagonal or construct a diagonal matrix.
        
        Args:
            x: Input array. If x is 2-D, return the k-th diagonal.
               If x is 1-D, return a 2-D array with x on the k-th diagonal.
            k: Diagonal offset. Use k>0 for diagonals above the main diagonal,
               and k<0 for diagonals below the main diagonal.
               
        Returns:
            The extracted diagonal or constructed diagonal matrix.
        """
        from ember_ml.backend.mlx.linearalg.ops import diag as diag_func
        return diag_func(x, k=k)
    
    def diagonal(self, x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1):
        """
        Return specified diagonals of an array.
        
        Args:
            x: Input array
            offset: Offset of the diagonal from the main diagonal
            axis1: First axis of the 2-D sub-arrays from which the diagonals should be taken
            axis2: Second axis of the 2-D sub-arrays from which the diagonals should be taken
            
        Returns:
            Array of diagonals
        """
        from ember_ml.backend.mlx.linearalg.ops import diagonal as diagonal_func
        return diagonal_func(x, offset=offset, axis1=axis1, axis2=axis2)