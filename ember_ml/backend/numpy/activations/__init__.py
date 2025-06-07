"""NumPy activation operations for ember_ml."""

# Removed NumpyActivationOps import
# from ember_ml.backend.numpy.activations.activations_ops import NumpyActivationOps
from ember_ml.backend.numpy.activations.relu import relu
from ember_ml.backend.numpy.activations.sigmoid import sigmoid
from ember_ml.backend.numpy.activations.tanh import tanh
from ember_ml.backend.numpy.activations.softmax import softmax
from ember_ml.backend.numpy.activations.softplus import softplus

__all__ = [
    # "NumpyActivationOps", # Removed class export
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus"
]