"""
Statistical operations module.

This module provides a unified interface to statistical operations from the current backend
(NumPy, PyTorch, MLX) using the proxy module pattern. It dynamically forwards
attribute access to the appropriate backend module.
"""

# Import the stats proxy from the ops proxy module
from ember_ml.ops.proxy import stats as stats_proxy

# Import all operations from the stats proxy
mean = stats_proxy.mean
var = stats_proxy.var
median = stats_proxy.median
std = stats_proxy.std
percentile = stats_proxy.percentile
max = stats_proxy.max
min = stats_proxy.min
sum = stats_proxy.sum
cumsum = stats_proxy.cumsum
argmax = stats_proxy.argmax
sort = stats_proxy.sort
argsort = stats_proxy.argsort
gaussian = stats_proxy.gaussian

# Define __all__ to include all operations
__all__ = [
    'mean',
    'var',
    'median',
    'std',
    'percentile',
    'max',
    'min',
    'sum',
    'cumsum',
    'argmax',
    'sort',
    'argsort',
    'gaussian',
]
