"""
NumPy backend configuration for ember_ml.

This module provides configuration settings for the NumPy backend.
"""

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

# Default device for NumPy operations
DEFAULT_DEVICE = 'cpu'

# Default data type for NumPy operations
DEFAULT_DTYPE = NumpyDType().float32

# Current random seed
_current_seed = None