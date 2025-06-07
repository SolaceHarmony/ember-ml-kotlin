"""Neural network parameter initialization module.

This module provides backend-agnostic weight initialization schemes for 
neural network parameters.

Components:
    Standard Initializations:
        - glorot_uniform: Glorot/Xavier uniform initialization
        - glorot_normal: Glorot/Xavier normal initialization
        - orthogonal: Orthogonal matrix initialization
        
    Specialized Initializations:
        - BinomialInitializer: Discrete binary initialization
        - binomial: Helper function for binomial initialization

    Helper Functions:
        - get_initializer: Get an initializer function by name

All initializers maintain numerical stability and proper scaling
while preserving backend independence.
"""

from typing import Callable, Any, Dict, Tuple, Optional

# Use relative imports for files within the same package
from .glorot import glorot_uniform, glorot_normal, orthogonal
from .binomial import BinomialInitializer, binomial
from ember_ml.nn import tensor

# Dictionary mapping initializer names to functions
_INITIALIZERS = {
    'glorot_uniform': glorot_uniform,
    'glorot_normal': glorot_normal,
    'orthogonal': orthogonal,
    'zeros': tensor.zeros,
    'ones': tensor.ones,
    'random_uniform': tensor.random_uniform,
    'random_normal': tensor.random_normal,
}

def get_initializer(name: str) -> Callable:
    """
    Get an initializer function by name.
    
    Args:
        name: Name of the initializer
        
    Returns:
        Initializer function
        
    Raises:
        ValueError: If the initializer name is not recognized
    """
    if name not in _INITIALIZERS:
        raise ValueError(f"Unknown initializer: {name}. Available initializers: {', '.join(_INITIALIZERS.keys())}")
    
    return _INITIALIZERS[name]

__all__ = [
    'glorot_uniform',
    'glorot_normal',
    'orthogonal',
    'BinomialInitializer',
    'binomial',
    'get_initializer',
]
