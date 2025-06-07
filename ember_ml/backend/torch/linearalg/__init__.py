"""PyTorch linear algebra operations for ember_ml."""

from ember_ml.backend.torch.linearalg.decomp_ops import qr, svd, cholesky, eig, eigvals, eigh
from ember_ml.backend.torch.linearalg.inverses_ops import inv
from ember_ml.backend.torch.linearalg.matrix_ops import det, norm, diag, diagonal
from ember_ml.backend.torch.linearalg.solvers_ops import solve, lstsq # eig, eigvals moved
from ember_ml.backend.torch.linearalg.decomp_ops import eig, eigvals # Import from correct file
from ember_ml.backend.torch.linearalg.orthogonal_ops import orthogonal # Import orthogonal function

__all__ = [
    # Functions exported from submodules
    "norm",
    "inv",
    "solve",
    "eig",
    "eigh",
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