"""MLX linear algebra operations for ember_ml."""

# Removed MLXLinearAlgOps import
# from ember_ml.backend.mlx.linearalg.linearalg_ops import MLXLinearAlgOps

# Import directly from moved files using absolute paths
from ember_ml.backend.mlx.linearalg.eigen_ops import cholesky, eig, eigvals, eigh # Added eigh
from ember_ml.backend.mlx.linearalg.svd_ops import svd
from ember_ml.backend.mlx.linearalg.qr_ops import qr
from ember_ml.backend.mlx.linearalg.inverses_ops import inv # Assuming function is here
from ember_ml.backend.mlx.linearalg.matrix_ops import det, norm, diag, diagonal # Assuming functions are here
from ember_ml.backend.mlx.linearalg.solvers_ops import solve, lstsq # Assuming functions are here
from ember_ml.backend.mlx.linearalg.orthogonal_ops import orthogonal # Import orthogonal function

# Import HPC-specific components
from ember_ml.backend.mlx.linearalg.hpc16x8_ops import _add_limb_precision, HPC16x8
from ember_ml.backend.mlx.linearalg.orthogonal_nonsquare import orthogonalize_nonsquare

# Note: decomp_ops_hpc.py and qr_128.py might contain specialized versions not directly imported here

__all__ = [
    # "MLXLinearAlgOps", # Removed class export
    "norm",
    "inv",
    "solve",
    "eig",
    "eigvals",
    "eigh", # Added eigh
    "qr",
    "det",
    "cholesky",
    "lstsq",
    "svd",
    "diag",
    "diagonal",
    "orthogonal", # Add orthogonal to exports
    # Add HPC components to exports
    "orthogonalize_nonsquare",
    "_add_limb_precision",
    "HPC16x8"
]