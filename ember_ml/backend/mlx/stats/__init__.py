"""
MLX statistical operations for ember_ml.

This module provides MLX implementations of statistical operations.
"""

# Removed MLXStatsOps import
# from ember_ml.backend.mlx.stats.stats_ops import MLXStatsOps
from ember_ml.backend.mlx.stats.descriptive import ( # Import from moved file
    # Descriptive statistics
    median,
    std,
    percentile,
    mean,
    median,
    std,
    percentile,
    var,
    max,
    min,
    sum,
    cumsum,
    argmax,
    sort,
    argsort,
    gaussian,
)

__all__ = [
    # "MLXStatsOps", # Removed class export
    "mean",
    "var",
    "median",
    "std",
    "percentile",
    "max",
    "min",
    "sum",
    "cumsum",
    "argmax",
    "sort",
    "argsort",
    "gaussian",
]