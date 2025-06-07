"""
Module Cell abstract base class.

This module provides the ModuleCell abstract base class, which defines
the interface for all cell types in ember_ml.
"""

from typing import Optional, List, Dict, Any, Union, Tuple, TypeVar

from ember_ml import ops
import ember_ml.nn.tensor as tensor
from ember_ml.nn.modules import Module, Parameter

# Type variable for state size
StateSize = TypeVar('StateSize', int, List[int])

class ModuleCell(Module):
    """
    Abstract base class for all cell types.
    
    This class defines the interface for all cell types, including RNN cells,
    LSTM cells, GRU cells, etc.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        use_bias: bool = True,
        **kwargs
    ):
        """
        Initialize a ModuleCell.
        
        Args:
            input_size: Size of the input
            hidden_size: Size of the hidden state
            activation: Activation function to use
            use_bias: Whether to use bias
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.activation = ops.get_activation(activation)
        self.use_bias = use_bias
    
    @property
    def state_size(self) -> Union[int, List[int]]:
        """Return the size of the cell state."""
        return self.hidden_size
    
    @property
    def output_size(self) -> int:
        """Return the output size."""
        return self.hidden_size
    
    def forward(self, inputs, state=None, **kwargs):
        """
        Forward pass of the cell.
        
        Args:
            inputs: Input tensor
            state: State tensor (default: None)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, new_state)
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def reset_state(self, batch_size=1):
        """
        Reset the cell state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        if isinstance(self.state_size, int):
            return tensor.zeros((batch_size, self.state_size))
        else:
            # Handle list state sizes
            return [tensor.zeros((batch_size, size)) for size in self.state_size]