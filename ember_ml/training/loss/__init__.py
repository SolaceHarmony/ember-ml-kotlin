"""
Loss module for ember_ml.

This module provides backend-agnostic implementations of loss functions
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.training.loss.base import Loss
from ember_ml.training.loss.mse import MSELoss
from ember_ml.training.loss.cross_entropy import CrossEntropyLoss

__all__ = ['Loss', 'MSELoss', 'CrossEntropyLoss']