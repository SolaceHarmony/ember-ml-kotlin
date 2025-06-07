from ember_ml.backend.numpy.linearalg.ops.matrix_ops import (
    norm,
    det,
    diag,
    diagonal
)
from ember_ml.backend.numpy.linearalg.ops.decomp_ops import (
    cholesky,
    svd,
    eig,
    eigvals,
    qr
)
from ember_ml.backend.numpy.linearalg.ops.inverses_ops import (
    inv
)

from ember_ml.backend.numpy.linearalg.ops.solvers_ops import (
    solve,
    lstsq
)

__all__ = [
    'norm',
    'det',
    'cholesky',
    'svd',
    'eig',
    'eigvals',
    'qr',
    'inv',
    'solve',
    'lstsq',
    'diag',
    'diagonal'
]
"""NumPy linear algebra operations."""

from ember_ml.backend.numpy.linearalg.ops.matrix_ops import diag, diagonal