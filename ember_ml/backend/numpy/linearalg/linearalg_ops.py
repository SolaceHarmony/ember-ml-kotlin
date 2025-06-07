"""NumPy linear algebra operations."""

from typing import Any, Optional, Union, Tuple

import numpy as np

from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.linearalg.ops import (
    solve, inv, det, norm, qr, svd, cholesky, lstsq, eig, eigvals, diag, diagonal
)

class NumpyLinearAlgOps:
    """NumPy linear algebra operations."""

    def __init__(self):
        """Initialize NumPy linear algebra operations."""
        pass

    def solve(self, a: TensorLike, b: TensorLike) -> np.ndarray:
        """
        Solve a linear system of equations Ax = b for x.
        
        Args:
            a: Coefficient matrix A
            b: Right-hand side vector or matrix b
            
        Returns:
            Solution to the system of equations
        """
        return solve(a, b)
    
    def inv(self, a: TensorLike) -> np.ndarray:
        """
        Compute the inverse of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Inverse of the matrix
        """
        return inv(a)
    
    def det(self, a: TensorLike) -> np.ndarray:
        """
        Compute the determinant of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Determinant of the matrix
        """
        return det(a)
    
    def norm(self, x: TensorLike, ord: Optional[Union[int, str]] = None,
             axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> np.ndarray:
        """
        Compute the matrix or vector norm.
        
        Args:
            x: Input matrix or vector
            ord: Order of the norm
            axis: Axis along which to compute the norm
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Norm of the matrix or vector
        """
        return norm(x, ord=ord, axis=axis, keepdims=keepdims)
    
    def qr(self, a: TensorLike, mode: str = 'reduced') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the QR decomposition of a matrix.
        
        Args:
            a: Input matrix
            mode: Mode of decomposition ('reduced', 'complete', 'r', 'raw')
            
        Returns:
            Tuple of (Q, R) matrices
        """
        return qr(a, mode=mode)
    
    def svd(self, a: TensorLike, full_matrices: bool = True, compute_uv: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute the singular value decomposition of a matrix.
        
        Args:
            a: Input matrix
            full_matrices: If True, return full U and Vh matrices
            compute_uv: If True, compute U and Vh matrices
            
        Returns:
            If compute_uv is True, returns (U, S, Vh), otherwise returns S
        """
        return svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    
    def cholesky(self, a: TensorLike) -> np.ndarray:
        """
        Compute the Cholesky decomposition of a positive definite matrix.
        
        Args:
            a: Input positive definite matrix
            
        Returns:
            Lower triangular matrix L such that L @ L.T = A
        """
        return cholesky(a)
    
    def lstsq(self, a: TensorLike, b: TensorLike, rcond: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the least-squares solution to a linear matrix equation.
        
        Args:
            a: Coefficient matrix
            b: Dependent variable
            rcond: Cutoff for small singular values
            
        Returns:
            Tuple of (solution, residuals, rank, singular values)
        """
        return lstsq(a, b, rcond=rcond)
    
    def eig(self, a: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigenvalues and eigenvectors of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        return eig(a)
    
    def eigvals(self, a: TensorLike) -> np.ndarray:
        """
        Compute the eigenvalues of a square matrix.
        
        Args:
            a: Input square matrix
            
        Returns:
            Eigenvalues of the matrix
        """
        return eigvals(a)
    
    def diag(self, x: TensorLike, k: int = 0) -> np.ndarray:
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
        return diag(x, k)

    def diagonal(self, x: TensorLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> np.ndarray:
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
        return diagonal(x, offset, axis1, axis2)