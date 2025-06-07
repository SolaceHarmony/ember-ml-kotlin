"""
NumPy high-precision computing for matrix decomposition operations.

This module provides high-precision matrix computation implementations for the NumPy backend.
It allows for more numerically stable computations by implementing a limb-based precision approach.
"""
from typing import Union, Tuple, Optional, Any
import numpy as np

# Import from tensor_ops
from ember_ml.backend.numpy.tensor import NumpyDType
from ember_ml.backend.numpy.types import TensorLike
from ember_ml.backend.numpy.linearalg.qr_128 import qr_128
from ember_ml.backend.numpy.linearalg.hpc_nonsquare import orthogonalize_nonsquare

dtype_obj = NumpyDType()

def _add_limb_precision(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Add two numbers with limb-based extended precision."""
    s = np.add(a, b)
    # Compute error term using Knuth's algorithm
    v = np.subtract(s, a)
    e = np.add(np.subtract(a, np.subtract(s, v)), np.subtract(b, v))
    return s, e

def _mul_limb_precision(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply two numbers with limb-based extended precision."""
    p = np.multiply(a, b)
    # Split computation to maintain precision
    c = np.multiply(np.array(0x10000, dtype=a.dtype), np.multiply(a, b))
    high = np.subtract(c, np.subtract(c, p))
    low = np.subtract(p, high)
    return high, low

def _update_array_md(arr: np.ndarray, indices: Union[int, Tuple[Any, ...], slice, Tuple[Union[slice, int], ...]], value: np.ndarray) -> np.ndarray:
    """Helper function for NumPy array updates with multi-dimensional support."""
    result = np.array(arr)
    result[indices] = value
    return result

class HPC16x8:
    """High-precision computing using limb-based arithmetic."""
    
    def __init__(self, high: np.ndarray, low: Optional[np.ndarray] = None):
        """Initialize HPC object with high and optional low components."""
        self.high = np.array(high, dtype=np.float32)
        self.low = np.zeros_like(self.high) if low is None else np.array(low, dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: TensorLike) -> 'HPC16x8':
        """Create HPC object from tensor-like input, splitting into high/low components."""
        # Create instances for each call to avoid circular imports
        from ember_ml.backend.numpy.tensor import NumpyTensor
        tensor_ops = NumpyTensor()
        
        # Convert to NumPy tensor first
        arr = tensor_ops.convert_to_tensor(arr)
        # Convert to float32
        arr = np.array(arr, dtype=np.float32)
        # Split into high and low parts using limb arithmetic
        high = np.array(arr, dtype=np.float32)
        low = np.subtract(arr, high)
        return cls(high, low)
    
    def to_float32(self) -> np.ndarray:
        """Convert back to float32 NumPy array."""
        result, _ = _add_limb_precision(self.high, self.low)
        return result
    
    def complete_basis(self) -> 'HPC16x8':
        """Complete orthogonal basis using Metal-accelerated implementation."""
        # Use Metal-accelerated orthogonalization
        completed = orthogonalize_nonsquare(self.high)
        # Return as new HPC object
        return self.__class__.from_array(completed)
    
    def qr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute QR decomposition with extended precision."""
        return qr_128(self.to_float32())
    
    def eig(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        For symmetric matrices, uses NumPy's native eigh.
        For non-symmetric cases, uses power iteration with extended precision.
        """
        # Convert to standard precision for initial computation
        matrix = self.to_float32()
        
        # Check if matrix is symmetric
        diff = np.subtract(matrix, np.transpose(matrix))
        abs_diff = np.abs(diff)
        is_symmetric = np.all(np.less(abs_diff, np.array(1e-6))).item()
        
        if is_symmetric:
            try:
                # Use NumPy's native eigh for symmetric case
                w, v = np.linalg.eigh(matrix)
                # Sort in descending order
                sort_idx = np.argsort(-w)
                return w[sort_idx], v[:, sort_idx]
            except Exception as e:
                print(f"Native eigh failed with error: {e}")
                # Fall through to power iteration
        
        # Use power iteration with extended precision
        n = matrix.shape[0]
        max_iters = 100
        tol = 1e-10
        
        # Initialize with random vectors
        v = np.random.normal(size=(n, n))
        
        # Orthogonalize initial vectors
        q = np.zeros((n, n), dtype=matrix.dtype)
        for j in range(n):
            vj = v[:, j]
            # Orthogonalize against previous vectors
            for i in range(j):
                vj = np.subtract(vj, np.multiply(
                    np.sum(np.multiply(q[:, i], vj)),
                    q[:, i]
                ))
            # Normalize
            vj_norm = np.linalg.norm(vj)
            if np.greater(vj_norm, np.array(tol)):
                q_col = np.divide(vj, vj_norm)
            else:
                q_col = np.zeros_like(vj)
                q_col = _update_array_md(q_col, j % n, np.array(1.0))
            q = _update_array_md(q, (slice(None), j), q_col)
        
        # Power iteration with extended precision
        for _ in range(max_iters):
            # Matrix multiply with extended precision
            v_high = np.matmul(matrix, q)
            v_low = np.subtract(
                np.matmul(self.high, q),
                v_high
            )
            v = np.add(v_high, v_low)
            
            # Orthogonalize
            for j in range(n):
                vj = v[:, j]
                for i in range(j):
                    vj = np.subtract(vj, np.multiply(
                        np.sum(np.multiply(q[:, i], vj)),
                        q[:, i]
                    ))
                # Normalize
                vj_norm = np.linalg.norm(vj)
                if np.greater(vj_norm, np.array(tol)):
                    q_new = np.divide(vj, vj_norm)
                else:
                    q_new = np.zeros_like(vj)
                    q_new = _update_array_md(q_new, j % n, np.array(1.0))
                q = _update_array_md(q, (slice(None), j), q_new)
        
        # Compute eigenvalues using Rayleigh quotient
        eigenvalues = np.zeros(n, dtype=matrix.dtype)
        for i in range(n):
            qi = q[:, i]
            eigenvalues = _update_array_md(eigenvalues, i,
                np.sum(np.multiply(qi, np.matmul(matrix, qi)))
            )
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        q = q[:, idx]
        
        return eigenvalues, q