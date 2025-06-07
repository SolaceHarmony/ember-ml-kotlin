"""
NumPy statistical operations for ember_ml.

This module provides NumPy implementations of statistical operations.
"""

from ember_ml.backend.numpy.stats.descriptive import ( # Import from moved file
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