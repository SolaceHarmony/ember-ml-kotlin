"""
Optimizer module for ember_ml.

This module provides backend-agnostic implementations of optimizers
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.training.optimizer.base import Optimizer
from ember_ml.training.optimizer.sgd import SGD
from ember_ml.training.optimizer.adam import Adam
from ember_ml.training.optimizer.gradient_tape import GradientTape
__all__ = ['Optimizer', 'SGD', 'Adam', 'GradientTape']