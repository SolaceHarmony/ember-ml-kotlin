"""
Neural Circuit Policy (NCP) Wiring.

This module provides a wiring configuration for neural circuit policies,
which divides neurons into sensory, inter, and motor neurons.
"""

from typing import Optional, Tuple, List, Dict, Any

from ember_ml import ops
from ember_ml.nn import tensor

# Already imports NeuronMap correctly
from ember_ml.nn.modules.wiring.neuron_map import NeuronMap # Explicit path
from ember_ml.nn.tensor import EmberTensor, int32, zeros, ones, random_uniform

class NCPMap(NeuronMap): # Name is already correct
    """
    Neural Circuit Policy (NCP) wiring configuration.
    
    In an NCP wiring, neurons are divided into three groups:
    - Sensory neurons: Receive input from the environment
    - Inter neurons: Process information internally
    - Command neurons: Coordinate motor responses
    - Motor neurons: Produce output to the environment
    
    The connectivity pattern between these groups is defined by the
    sparsity level and can be customized.
    
    This class also includes cell-specific parameters that define the dynamics
    of the neural network, making it a complete blueprint for both structure
    and behavior.
    """
    
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_neurons: int = 0,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        # Add cell-specific parameters
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        mode: str = "default",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        mixed_memory: bool = False,
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        # Keep existing sparsity parameters
        sensory_to_inter_sparsity: Optional[float] = None,
        sensory_to_motor_sparsity: Optional[float] = None,
        inter_to_inter_sparsity: Optional[float] = None,
        inter_to_motor_sparsity: Optional[float] = None,
        motor_to_motor_sparsity: Optional[float] = None,
        motor_to_inter_sparsity: Optional[float] = None,
        units: Optional[int] = None,  # Added for compatibility with from_config
        output_dim: Optional[int] = None,  # Added for compatibility with from_config
        input_dim: Optional[int] = None,  # Added for compatibility with from_config
    ):
        """
        Initialize an NCP wiring configuration.
        
        Args:
            inter_neurons: Number of inter neurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_neurons: Number of sensory neurons (default: 0)
            sparsity_level: Default sparsity level for all connections (default: 0.5)
            seed: Random seed for reproducibility
            
            # Cell-specific parameters
            time_scale_factor: Factor to scale the time constant (default: 1.0)
            activation: Activation function for the output (default: "tanh")
            recurrent_activation: Activation function for the recurrent step (default: "sigmoid")
            mode: Mode of operation (default: "default")
            use_bias: Whether to use bias (default: True)
            kernel_initializer: Initializer for the kernel weights (default: "glorot_uniform")
            recurrent_initializer: Initializer for the recurrent weights (default: "orthogonal")
            bias_initializer: Initializer for the bias (default: "zeros")
            mixed_memory: Whether to use mixed memory (default: False)
            ode_unfolds: Number of ODE solver unfoldings (default: 6)
            epsilon: Small constant to avoid division by zero (default: 1e-8)
            implicit_param_constraints: Whether to use implicit parameter constraints (default: False)
            input_mapping: Type of input mapping ('affine', 'linear', or None) (default: "affine")
            output_mapping: Type of output mapping ('affine', 'linear', or None) (default: "affine")
            
            # Sparsity parameters
            sensory_to_inter_sparsity: Sparsity level for sensory to inter connections
            sensory_to_motor_sparsity: Sparsity level for sensory to motor connections
            inter_to_inter_sparsity: Sparsity level for inter to inter connections
            inter_to_motor_sparsity: Sparsity level for inter to motor connections
            motor_to_motor_sparsity: Sparsity level for motor to motor connections
            motor_to_inter_sparsity: Sparsity level for motor to inter connections
            
            # Compatibility parameters
            units: Total number of units (optional, for compatibility)
            output_dim: Output dimension (optional, for compatibility)
            input_dim: Input dimension (optional, for compatibility)
        """
        # If units is provided, use it, otherwise calculate it
        if units is None:
            units = inter_neurons + command_neurons + motor_neurons + sensory_neurons
        
        # If output_dim is provided, use it, otherwise use motor_neurons
        if output_dim is None:
            output_dim = motor_neurons
        
        # If input_dim is provided, use it, otherwise use sensory_neurons if > 0
        if input_dim is None:
            input_dim = sensory_neurons if sensory_neurons > 0 else None
        
        super().__init__(units, output_dim, input_dim, sparsity_level, seed)
        
        # Store NCP-specific structural parameters
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_neurons = sensory_neurons
        
        # Store cell-specific parameters
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.mode = mode
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.mixed_memory = mixed_memory
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.implicit_param_constraints = implicit_param_constraints
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        
        # Custom sparsity levels
        self.sensory_to_inter_sparsity = sensory_to_inter_sparsity or sparsity_level
        self.sensory_to_motor_sparsity = sensory_to_motor_sparsity or sparsity_level
        self.inter_to_inter_sparsity = inter_to_inter_sparsity or sparsity_level
        self.inter_to_motor_sparsity = inter_to_motor_sparsity or sparsity_level
        self.motor_to_motor_sparsity = motor_to_motor_sparsity or sparsity_level
        self.motor_to_inter_sparsity = motor_to_inter_sparsity or sparsity_level
    
    def build(self, input_dim=None) -> Tuple[EmberTensor, EmberTensor, EmberTensor]:
        """
        Build the NCP wiring configuration.
        
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
        
        # Create masks
        input_mask = ones((self.input_dim,), dtype=int32)
        
        # Define neuron group indices based on diagram and original source structure
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = sensory_end + self.inter_neurons
        command_start = inter_end # Add command indices
        command_end = inter_end + self.command_neurons
        motor_start = command_end # Adjust motor indices
        motor_end = command_end + self.motor_neurons
        
        # Create output mask (only motor neurons contribute to output)
        # Initialize with zeros
        output_mask = zeros((self.units,), dtype=int32)
        
        # Set motor neurons to 1
        # Create a range of indices for motor neurons
        motor_indices = tensor.arange(motor_start, motor_end)
        
        # Create a tensor of ones with the same shape as motor_indices
        motor_values = ones((motor_end - motor_start,), dtype=int32)
        
        # Use scatter update to set motor neurons to 1
        output_mask = tensor.tensor_scatter_nd_update(
            output_mask,
            tensor.reshape(motor_indices, (-1, 1)),
            motor_values
        )
        
        # Initialize recurrent mask with zeros
        recurrent_mask = zeros((self.units, self.units), dtype=int32)
        
        # Helper function to create random connections between neuron groups
        def create_random_connections(from_start, from_end, to_start, to_end, sparsity):
            if from_end <= from_start or to_end <= to_start:
                return  # Skip if either group is empty
            
            # Create indices for the from and to neurons
            from_size = from_end - from_start
            to_size = to_end - to_start
            
            # Create a random mask for connections
            random_mask = random_uniform((from_size, to_size))
            connection_mask = ops.greater_equal(random_mask, sparsity)
            
            # Create indices for the connections
            from_indices = tensor.reshape(tensor.arange(from_start, from_end), (-1, 1, 1))
            to_indices = tensor.reshape(tensor.arange(to_start, to_end), (1, -1, 1))
            
            # Combine indices where connection_mask is True
            mask_indices = tensor.nonzero(connection_mask)
            if tensor.shape(mask_indices)[0] > 0:
                from_idx = from_indices[mask_indices[:, 0], 0, 0] 
                to_idx = to_indices[0, mask_indices[:, 1], 0]
                
                # Create update indices and values
                update_indices = tensor.stack([from_idx, to_idx], axis=1)
                update_values = ones((tensor.shape(update_indices)[0],), dtype=int32)
                
                # Update the recurrent mask
                nonlocal recurrent_mask
                recurrent_mask = tensor.tensor_scatter_nd_update(
                    recurrent_mask, 
                    update_indices, 
                    update_values
                )
        
        # Create connections between neuron groups
        # Sensory to inter connections
        if self.sensory_neurons > 0 and self.inter_neurons > 0:
            create_random_connections(
                sensory_start, sensory_end, 
                inter_start, inter_end, 
                self.sensory_to_inter_sparsity
            )
        
        # Sensory to command connections
        if self.sensory_neurons > 0 and self.command_neurons > 0:
            create_random_connections(
                sensory_start, sensory_end, 
                command_start, command_end, 
                self.sensory_to_inter_sparsity
            )
        
        # Inter to inter connections
        if self.inter_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                inter_start, inter_end, 
                self.inter_to_inter_sparsity
            )
        
        # Inter to command connections
        if self.inter_neurons > 0 and self.command_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                command_start, command_end, 
                self.inter_to_inter_sparsity
            )
        
        # Inter to motor connections
        if self.inter_neurons > 0 and self.motor_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                motor_start, motor_end, 
                self.inter_to_motor_sparsity
            )
        
        # Command to motor connections
        if self.command_neurons > 0 and self.motor_neurons > 0:
            create_random_connections(
                command_start, command_end, 
                motor_start, motor_end, 
                self.inter_to_motor_sparsity
            )
        
        self._built = True # Mark map as built
        return input_mask, recurrent_mask, output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            # Structural parameters
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons,
            "motor_neurons": self.motor_neurons,
            "sensory_neurons": self.sensory_neurons,
            
            # Cell-specific parameters
            "time_scale_factor": self.time_scale_factor,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "mode": self.mode,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "mixed_memory": self.mixed_memory,
            "ode_unfolds": self.ode_unfolds,
            "epsilon": self.epsilon,
            "implicit_param_constraints": self.implicit_param_constraints,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            
            # Sparsity parameters
            "sensory_to_inter_sparsity": self.sensory_to_inter_sparsity,
            "sensory_to_motor_sparsity": self.sensory_to_motor_sparsity,
            "inter_to_inter_sparsity": self.inter_to_inter_sparsity,
            "inter_to_motor_sparsity": self.inter_to_motor_sparsity,
            "motor_to_motor_sparsity": self.motor_to_motor_sparsity,
            "motor_to_inter_sparsity": self.motor_to_inter_sparsity
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCPMap':
        """
        Create an NCP wiring configuration from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP wiring configuration
        """
        # Handle the case where 'units' is in the config but not needed by __init__
        # We'll keep it in the config for compatibility with the parent class
        return cls(**config)
    
    def get_neuron_groups(self) -> Dict[str, List[int]]:
        """
        Get the indices of neurons in each group.
        
        Returns:
            Dictionary mapping group names to lists of neuron indices
        """
        # Define start/end indices consistent with build method
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = inter_start + self.inter_neurons
        command_start = inter_end
        command_end = command_start + self.command_neurons
        motor_start = command_end
        motor_end = self.units # self.units now includes command_neurons

        # Generate index lists
        sensory_idx = list(range(sensory_start, sensory_end))
        inter_idx = list(range(inter_start, inter_end))
        command_idx = list(range(command_start, command_end)) # Add command indices
        motor_idx = list(range(motor_start, motor_end)) # Adjust motor indices

        return {
            "sensory": sensory_idx,
            "inter": inter_idx,
            "command": command_idx, # Add command group
            "motor": motor_idx
        }