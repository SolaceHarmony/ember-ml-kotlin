"""
PyTorch backend configuration for ember_ml.

This module provides configuration settings for the PyTorch backend.
"""

import torch
from ember_ml.backend.torch.tensor.dtype import TorchDType

# Default device for PyTorch operations
has_gpu = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
DEFAULT_DEVICE = 'cuda' if has_gpu else 'mps' if has_mps else 'cpu'

# Default data type for PyTorch operations
DEFAULT_DTYPE = TorchDType().float32
