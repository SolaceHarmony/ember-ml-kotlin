"""
MLX high-precision computing for matrix decomposition operations.

This module provides high-precision matrix computation implementations for the MLX backend.
It allows for more numerically stable computations by implementing a limb-based precision approach.
"""
from typing import Union, Tuple, Optional, Any
import mlx.core as mx

# Import from tensor_ops
from ember_ml.backend.mlx.tensor import MLXDType
from ember_ml.backend.mlx.types import TensorLike
# Import the consolidated QR function from qr_ops
from .qr_ops import qr
from .orthogonal_nonsquare import orthogonalize_nonsquare

dtype_obj = MLXDType()

def _add_limb_precision(a: mx.array, b: mx.array) -> Tuple[mx.array, mx.array]:
    """Add two numbers with limb-based extended precision."""
    s = mx.add(a, b)
    # Compute error term using Knuth's algorithm
    v = mx.subtract(s, a)
    e = mx.add(mx.subtract(a, mx.subtract(s, v)), mx.subtract(b, v))
    return s, e

def _mul_limb_precision(a: mx.array, b: mx.array) -> Tuple[mx.array, mx.array]:
    """Multiply two numbers with limb-based extended precision."""
    p = mx.multiply(a, b)
    # Split computation to maintain precision
    c = mx.multiply(mx.array(0x10000, dtype=a.dtype), mx.multiply(a, b))
    high = mx.subtract(c, mx.subtract(c, p))
    low = mx.subtract(p, high)
    return high, low

def _update_array_md(arr: mx.array, indices: Union[int, Tuple[Any, ...], slice, Tuple[Union[slice, int], ...]], value: mx.array) -> mx.array:
    """Helper function for MLX array updates with multi-dimensional support."""
    result = mx.array(arr)
    result[indices] = value
    return result

class HPC16x8:
    """High-precision computing using limb-based arithmetic."""

    def __init__(self, high: mx.array, low: Optional[mx.array] = None):
        """Initialize HPC object with high and optional low components."""
        self.high = mx.array(high, dtype=mx.float32)
        self.low = mx.zeros_like(self.high) if low is None else mx.array(low, dtype=mx.float32)

    @classmethod
    def from_array(cls, arr: TensorLike) -> 'HPC16x8':
        """Create HPC object from tensor-like input, splitting into high/low components."""
        # Create instances for each call to avoid circular imports
        from ember_ml.backend.mlx.tensor import MLXTensor
        tensor_ops = MLXTensor()

        # Convert to MLX tensor first
        arr = tensor_ops.convert_to_tensor(arr)
        # Convert to float32
        arr = mx.array(arr, dtype=mx.float32)
        # Split into high and low parts using limb arithmetic
        high = mx.array(arr, dtype=mx.float32)
        low = mx.subtract(arr, high)
        return cls(high, low)

    def to_float32(self) -> mx.array:
        """Convert back to float32 MLX array."""
        result, _ = _add_limb_precision(self.high, self.low)
        return result

    def complete_basis(self) -> 'HPC16x8':
        """Complete orthogonal basis using Metal-accelerated implementation."""
        # Use Metal-accelerated orthogonalization
        completed = orthogonalize_nonsquare(self.high)
        # Return as new HPC object
        return self.__class__.from_array(completed)

    def qr(self) -> Tuple[mx.array, mx.array]:
        """
        Compute QR decomposition using the consolidated implementation from qr_ops.

        Note: This method converts the HPC16x8 object to a standard float32 matrix
        before calling the underlying QR function. It does not perform the QR
        decomposition using the extended precision representation directly.
        """
        # Use the consolidated qr function from qr_ops on the float32 representation
        return qr(self.to_float32())

    def eig(self) -> Tuple[mx.array, mx.array]:
        """
        Compute eigenvalues and eigenvectors.

        For symmetric matrices, uses MLX's native eigh.
        For non-symmetric cases, uses power iteration with extended precision.
        """
        # Convert to standard precision for initial computation
        matrix = self.to_float32()

        # Check if matrix is symmetric
        diff = mx.subtract(matrix, mx.transpose(matrix))
        abs_diff = mx.abs(diff)
        is_symmetric = mx.all(mx.less(abs_diff, mx.array(1e-6))).item()

        if is_symmetric:
            try:
                # Use MLX's native eigh for symmetric case
                w, v = mx.linalg.eigh(matrix)
                # Sort in descending order
                sort_idx = mx.argsort(-w)
                return w[sort_idx], v[:, sort_idx]
            except Exception as e:
                print(f"Native eigh failed with error: {e}")
                # Fall through to power iteration

        # Use power iteration with extended precision
        n = matrix.shape[0]
        max_iters = 100
        tol = 1e-10

        # Initialize with random vectors
        v = mx.random.normal((n, n))

        # Orthogonalize initial vectors
        q = mx.zeros((n, n), dtype=matrix.dtype)
        for j in range(n):
            vj = v[:, j]
            # Orthogonalize against previous vectors
            for i in range(j):
                vj = mx.subtract(vj, mx.multiply(
                    mx.sum(mx.multiply(q[:, i], vj)),
                    q[:, i]
                ))
            # Normalize
            vj_norm = mx.linalg.norm(vj)
            if mx.greater(vj_norm, mx.array(tol)):
                q_col = mx.divide(vj, vj_norm)
            else:
                q_col = mx.zeros_like(vj)
                q_col = _update_array_md(q_col, j % n, mx.array(1.0))
            q = _update_array_md(q, (slice(None), j), q_col)

        # Power iteration with extended precision
        for _ in range(max_iters):
            # Matrix multiply with extended precision
            v_high = mx.matmul(matrix, q)
            v_low = mx.subtract(
                mx.matmul(self.high, q),
                v_high
            )
            v = mx.add(v_high, v_low)

            # Orthogonalize
            for j in range(n):
                vj = v[:, j]
                for i in range(j):
                    vj = mx.subtract(vj, mx.multiply(
                        mx.sum(mx.multiply(q[:, i], vj)),
                        q[:, i]
                    ))
                # Normalize
                vj_norm = mx.linalg.norm(vj)
                if mx.greater(vj_norm, mx.array(tol)):
                    q_new = mx.divide(vj, vj_norm)
                else:
                    q_new = mx.zeros_like(vj)
                    q_new = _update_array_md(q_new, j % n, mx.array(1.0))
                q = _update_array_md(q, (slice(None), j), q_new)

        # Compute eigenvalues using Rayleigh quotient
        eigenvalues = mx.zeros(n, dtype=matrix.dtype)
        for i in range(n):
            qi = q[:, i]
            eigenvalues = _update_array_md(eigenvalues, i,
                mx.sum(mx.multiply(qi, mx.matmul(matrix, qi)))
            )

        # Sort by eigenvalue magnitude
        idx = mx.argsort(-mx.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        q = q[:, idx]

        return eigenvalues, q