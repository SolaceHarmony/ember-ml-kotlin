"""
Binomial Initializer for ember_ml.

This module provides a binomial initializer for neural network weights and biases.
"""

from typing import Tuple, Optional, Any

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor

class BinomialInitializer:
    """
    Initializer that generates a binary mask with given probability.
    
    Args:
        probability: Probability of 1s in the mask
        seed: Random seed
    """
    def __init__(self, probability: float = 0.5, seed: Optional[int] = None):
        """
        Initialize the BinomialInitializer.
        
        Args:
            probability: Probability of 1s in the mask
            seed: Random seed
        """
        self.probability = probability
        self.seed = seed
    
    def __call__(self, shape: Tuple[int, ...], dtype: Any = None) -> EmberTensor:
        """
        Generate a binary mask.
        
        Args:
            shape: Shape of the mask
            dtype: Data type
            
        Returns:
            Binary mask
        """
        if dtype is None:
            dtype = "float32"
        
        # Set random seed if provided
        if self.seed is not None:
            tensor.set_seed(self.seed)
        
        # Generate random values using tensor.random_uniform
        random_values = tensor.random_uniform(shape, minval=0.0, maxval=1.0)
        
        # Create binary mask by comparing with probability
        binary_mask = tensor.cast(ops.less(random_values, self.probability), dtype=dtype)
        
        return binary_mask
    
    def get_config(self) -> dict:
        """
        Get configuration.
        
        Returns:
            Configuration dictionary
        """
        return {"probability": self.probability, "seed": self.seed}

def binomial(shape: Tuple[int, ...], probability: float = 0.5, seed: Optional[int] = None, dtype: Any = None) -> EmberTensor:
    """
    Generate a binary mask with given probability.
    
    Args:
        shape: Shape of the mask
        probability: Probability of 1s in the mask
        seed: Random seed
        dtype: Data type
        
    Returns:
        Binary mask
    """
    initializer = BinomialInitializer(probability=probability, seed=seed)
    return initializer(shape, dtype=dtype)