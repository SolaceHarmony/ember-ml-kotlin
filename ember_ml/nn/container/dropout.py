"""
Dropout layer implementation for ember_ml.

This module provides a backend-agnostic implementation of a dropout layer
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Optional, Union, Tuple, Any, Dict, List

from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor
class Dropout(Module):
    """
    Applies Dropout to the input.
    
    During training, randomly zeroes some of the elements of the input tensor with probability p.
    
    Args:
        rate: Probability of an element to be zeroed. Default: 0.5
        seed: Optional random seed for reproducibility
    """
    
    def __init__(
        self,
        rate: float = 0.5,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.rate = rate
        self.seed = seed
        self.training = True
    
    def forward(self, x):
        """
        Forward pass of the dropout layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with dropout applied (if training is True)
        """
        # Ensure x is a tensor
        x = tensor.convert_to_tensor(x)
        
        # If not in training mode or rate is 0, return input unchanged
        if not self.training or self.rate == 0.0:
            return x
            
        # Create a random mask
        if self.seed is not None:
            tensor.set_seed(self.seed)
            
        mask = ops.greater_equal(
            tensor.random_uniform(tensor.shape(x)),
            tensor.convert_to_tensor(self.rate)
        )
        
        # Apply mask and scale
        scale = tensor.convert_to_tensor(1.0 / (1.0 - self.rate))
        return ops.multiply(ops.multiply(x, tensor.cast(mask, tensor.dtype(x))), scale)
    
    def add(self, layer: Any) -> None:
        """
        Add a layer to the container.
        
        Args:
            layer: Layer to add
            
        Raises:
            NotImplementedError: Dropout layer does not support adding layers
        """
        raise NotImplementedError("Dropout layer does not support adding layers")
    
    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """
        Build the container for a specific input shape.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Dropout layer does not need to be built
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the container.
        
        Returns:
            Dictionary containing the configuration
        """
        return {
            'rate': self.rate,
            'seed': self.seed
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the container.
        
        Returns:
            Dictionary containing the state
        """
        return {
            'rate': self.rate,
            'seed': self.seed,
            'training': self.training
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary into the container.
        
        Args:
            state_dict: Dictionary containing the state
        """
        self.rate = state_dict.get('rate', 0.5)
        self.seed = state_dict.get('seed', None)
        self.training = state_dict.get('training', True)
    
    def train(self, mode: bool = True) -> 'Dropout':
        """
        Set the layer in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            Self
        """
        self.training = mode
        return self
    
    def eval(self) -> 'Dropout':
        """
        Set the layer in evaluation mode.
        
        Returns:
            Self
        """
        return self.train(False)
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"rate={self.rate}, seed={self.seed}"
    
    def __repr__(self):
        return f"Dropout(rate={self.rate}, seed={self.seed})"