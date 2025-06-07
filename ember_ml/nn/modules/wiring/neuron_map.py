"""
Base Wiring class for neural circuit policies.

This module provides the base Wiring class that defines the interface
for all wiring configurations.
"""

from typing import Optional, Tuple, Dict, Any
from ember_ml import ops
from ember_ml.ops import stats  # Import stats module for sum operation
from ember_ml.nn.tensor import EmberTensor, int32, convert_to_tensor
from ember_ml.nn.tensor.common import zeros, copy

class NeuronMap: # Renamed from Wiring
    """
    Base class for all wiring configurations.
    
    Wiring configurations define the connectivity patterns between neurons
    in neural circuit policies. They specify which neurons are connected to
    which other neurons, and with what weights.
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
        Initialize a wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        self.units = units
        self.output_dim = output_dim if output_dim is not None else units
        # Store input_dim if provided, otherwise leave it None until build()
        self.input_dim = input_dim
        self.sparsity_level = sparsity_level
        self.seed = seed
        
        # Initialize masks
        self._input_mask: Optional[EmberTensor] = None
        self._recurrent_mask: Optional[EmberTensor] = None
        self._output_mask: Optional[EmberTensor] = None
        
        # Initialize adjacency matrices
        self.adjacency_matrix = zeros([units, units], dtype=int32)
        self.sensory_adjacency_matrix = None
        self._built = False # Track build status
        
    def build(self, input_dim=None) -> Tuple[EmberTensor, EmberTensor, EmberTensor]:
        """
        Build the wiring configuration.
        
        Args:
            input_dim: Input dimension (optional)
        
        This method should be overridden by all subclasses to implement
        the specific wiring pattern.
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        if input_dim is not None:
            # Always use the provided input_dim
            self.set_input_dim(input_dim)
        
        raise NotImplementedError("Subclasses must implement build method")
    
    def set_input_dim(self, input_dim):
        """
        Set the input dimension.
        
        Args:
            input_dim: Input dimension
        """
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = zeros([input_dim, self.units], dtype=int32)
    
    def is_built(self):
        """
        Check if the wiring is built.
        
        Returns:
            True if the wiring is built, False otherwise
        """
        return self._built
    
    def get_input_mask(self) -> Optional[EmberTensor]:
        """
        Get the input mask.
        
        The input mask determines which input dimensions are connected to
        which neurons in the circuit.
        
        Returns:
            Input mask as an EmberTensor
        """
        if self._input_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        return convert_to_tensor(self._input_mask) if not isinstance(self._input_mask, EmberTensor) else self._input_mask
    def get_recurrent_mask(self) -> Optional[EmberTensor]:
        """
        Get the recurrent mask.
        
        The recurrent mask determines which neurons in the circuit are
        connected to which other neurons.
        
        Returns:
            Recurrent mask as a numpy array
        """
        if self._recurrent_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        # Return the mask directly
        return self._recurrent_mask
    
    def get_output_mask(self) -> Optional[EmberTensor]:
        """
        Get the output mask.
        
        The output mask determines which neurons in the circuit contribute
        to which output dimensions.
        
        Returns:
            Output mask as a numpy array
        """
        if self._output_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        # Return the mask directly
        return self._output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        return {
            "units": self.units,
            "output_dim": self.output_dim,
            # Ensure we save the actual input_dim, even if it's None initially
            "input_dim": self.input_dim,
            "sparsity_level": self.sparsity_level,
            "seed": self.seed
        }
    
    def erev_initializer(self, shape=None, dtype=None):
        """
        Initialize the reversal potential for the synapses.
        
        Args:
            shape: Shape of the output tensor (ignored)
            dtype: Data type of the output tensor (ignored)
            
        Returns:
            Adjacency matrix
        """
        return copy(self.adjacency_matrix)
    
    def sensory_erev_initializer(self, shape=None, dtype=None):
        """
        Initialize the reversal potential for the sensory synapses.
        
        Args:
            shape: Shape of the output tensor (ignored)
            dtype: Data type of the output tensor (ignored)
            
        Returns:
            Sensory adjacency matrix
        """
        if self.sensory_adjacency_matrix is not None:
            return copy(self.sensory_adjacency_matrix)
        return None
    
    @property
    def synapse_count(self):
        """Counts the number of synapses between internal neurons of the model"""
        return stats.sum(ops.abs(self.adjacency_matrix))
    
    @property
    def sensory_synapse_count(self):
        """Counts the number of synapses from the inputs (sensory neurons) to the internal neurons of the model"""
        return stats.sum(ops.abs(self.sensory_adjacency_matrix)) if self.sensory_adjacency_matrix is not None else 0
    
    def add_synapse(self, src, dest, polarity):
        """
        Add a synapse between two neurons.
        
        Args:
            src: Source neuron index
            dest: Destination neuron index
            polarity: Polarity of the synapse (-1 or 1)
        """
        if src < 0 or src >= self.units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity
    
    def add_sensory_synapse(self, src, dest, polarity):
        """
        Add a sensory synapse between an input and a neuron.
        
        Args:
            src: Source input index
            dest: Destination neuron index
            polarity: Polarity of the synapse (-1 or 1)
        """
        if self.input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if not polarity in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        
        # Initialize sensory_adjacency_matrix if it's None
        if self.sensory_adjacency_matrix is None:
            self.set_input_dim(self.input_dim)
        
        if self.sensory_adjacency_matrix is None:
            raise ValueError("Failed to initialize sensory_adjacency_matrix")
            
        self.sensory_adjacency_matrix[src, dest] = polarity
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NeuronMap': # Updated return type hint
        """
        Create a wiring configuration from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Wiring configuration
        """
        return cls(**config)