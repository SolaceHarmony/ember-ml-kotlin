"""
PyTorch statistical operations for ember_ml.

This module provides PyTorch implementations of statistical operations.
"""

# Removed TorchStatsOps import
# from ember_ml.backend.torch.stats.stats_ops import TorchStatsOps
from ember_ml.backend.torch.stats.descriptive import ( # Import from moved file
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