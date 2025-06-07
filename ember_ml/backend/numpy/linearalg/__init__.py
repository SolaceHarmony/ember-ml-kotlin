"""NumPy linear algebra operations for ember_ml."""

# Removed NumpyLinearAlgOps import
# from ember_ml.backend.numpy.linearalg.linearalg_ops import NumpyLinearAlgOps

# Import directly from moved files using absolute paths
from ember_ml.backend.numpy.linearalg.decomp_ops import qr, svd, cholesky
from ember_ml.backend.numpy.linearalg.inverses_ops import inv
from ember_ml.backend.numpy.linearalg.matrix_ops import det, norm, diag, diagonal
from ember_ml.backend.numpy.linearalg.solvers_ops import solve, lstsq # eig, eigvals moved
from ember_ml.backend.numpy.linearalg.decomp_ops import eig, eigvals # Import from correct file
from ember_ml.backend.numpy.linearalg.orthogonal_ops import orthogonal # Import orthogonal function

__all__ = [
    # "NumpyLinearAlgOps", # Removed class export
    "norm",
    "inv",
    "solve",
    "eig",
    "eigvals",
    "qr",
    "det",
    "cholesky",
    "lstsq",
    "svd",
    "diag",
    "diagonal",
    "orthogonal" # Add orthogonal to exports
]