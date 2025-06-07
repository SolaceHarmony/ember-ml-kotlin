"""Neural network primitives and core components.

This module provides the foundational building blocks for constructing neural
networks with backend-agnostic implementations.
        
All components maintain strict backend independence through the ops abstraction.
"""

import ember_ml.nn.modules
import ember_ml.nn.container
import ember_ml.nn.features
import ember_ml.nn.attention
import ember_ml.nn.initializers
import ember_ml.nn.modulation
import ember_ml.nn.tensor


__all__ = [
    'modules',
    'container',
    'features',
    'attention',
    'initializers', 
    'modulation',
    'tensor'
]
