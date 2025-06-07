"""
Random Wiring for neural circuit policies.

This module provides a random wiring configuration for neural
circuit policies, where connections are randomly generated.
"""

from typing import Optional, Tuple

from ember_ml import ops
from ember_ml.nn.tensor import EmberTensor, int32, random_uniform, cast
# Use explicit path for clarity now it's moved
from ember_ml.nn.modules.wiring.neuron_map import NeuronMap
from ember_ml.nn import tensor

class RandomMap(NeuronMap): # Name is already correct
    """
    Random wiring configuration.
    
    In a random wiring, connections between neurons are randomly generated
    based on the specified sparsity level.
    """
    
    def __init__(
        self, 
        units: int, 
        output_dim: Optional[int] = None, 
        input_dim: Optional[int] = None,
        sparsity_level: float = 0.5, 
        seed: Optional[int] = None
    ):
        """
        Initialize a random wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(units, output_dim, input_dim, sparsity_level, seed)
    
    def build(self, input_dim=None) -> Tuple[EmberTensor, EmberTensor, EmberTensor]:
        """
        Build the random wiring configuration.
        
        Args:
            input_dim: Input dimension (optional)
            
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set input_dim if provided
        if input_dim is not None:
            self.set_input_dim(input_dim)
            
        # Set random seed for reproducibility
        if self.seed is not None:
            tensor.set_seed(self.seed)
        
        # Create random masks
        input_mask = cast(
            random_uniform((self.input_dim,)) >= self.sparsity_level,
            dtype=int32
        )
        recurrent_mask = cast(
            random_uniform((self.units, self.units)) >= self.sparsity_level,
            dtype=int32
        )
        output_mask = cast(
            random_uniform((self.units,)) >= self.sparsity_level,
            dtype=int32
        )

        self._built = True # Mark map as built
        return input_mask, recurrent_mask, output_mask