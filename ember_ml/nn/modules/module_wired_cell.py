"""
Module Wired Cell abstract base class.

This module provides the ModuleWiredCell abstract base class, which defines
the interface for all wired cell types in ember_ml.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
import ember_ml.nn.tensor as tensor
from ember_ml.nn.modules import Module
from ember_ml.nn.wirings import Wiring

class ModuleWiredCell(Module):
    """
    Abstract base class for wired cell types.
    
    This class defines the interface for all wired cell types, which use
    wiring configurations to define connectivity patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        wiring: Wiring,
        mode: str = "default",
        **kwargs
    ):
        """
        Initialize a ModuleWiredCell.
        
        Args:
            input_size: Size of the input
            wiring: Wiring configuration
            mode: Mode of operation
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Store the wiring configuration
        self.wiring = wiring
        
        # Build the wiring if needed
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        
        # Store parameters
        self.input_size = input_size
        self.mode = mode
    
    @property
    def state_size(self):
        """Return the size of the cell state."""
        return self.wiring.units
    @property
    def layer_sizes(self):
        """Return the sizes of each layer."""
        # Default implementation for backward compatibility
        return [self.wiring.units]
    
    @property
    def num_layers(self):
        """Return the number of layers."""
        # Default implementation for backward compatibility
        return 1
        return self.wiring.num_layers
    
    @property
    def sensory_size(self):
        """Return the sensory size."""
        return self.wiring.input_dim
    
    @property
    def motor_size(self):
        """Return the motor size."""
        return self.wiring.output_dim
    
    @property
    def output_size(self):
        """Return the output size."""
        return self.motor_size
    
    @property
    def synapse_count(self):
        """Return the number of synapses."""
        return ops.sum(ops.abs(self.wiring.adjacency_matrix))
    
    @property
    def sensory_synapse_count(self):
        """Return the number of sensory synapses."""
        return ops.sum(ops.abs(self.wiring.sensory_adjacency_matrix)) if self.wiring.sensory_adjacency_matrix is not None else 0
    
    def forward(self, input, hx, timespans=None):
        """
        Forward pass of the cell.
        
        Args:
            input: Input tensor
            hx: Hidden state tensor
            timespans: Time spans for continuous-time dynamics
            
        Returns:
            Tuple of (output, new_state)
        """
        raise NotImplementedError("Subclasses must implement forward")