"""Machine learning models module.

This module provides backend-agnostic implementations of machine learning models
using the ops abstraction layer.

Available Models:
    RBM: Restricted Boltzmann Machine with configurable backend
    
Implementation Notes:
    - All models use the ops abstraction for computations
    - Models maintain strict backend independence
    - Training procedures are backend-agnostic
"""


from ember_ml.models.rbm import *

__all__ = [
    'rbm',
]
