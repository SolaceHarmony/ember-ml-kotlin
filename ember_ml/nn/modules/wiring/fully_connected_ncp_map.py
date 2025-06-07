"""
Fully Connected NCP Map for neural circuit policies.

This module provides a fully connected wiring configuration for neural
circuit policies, where all neurons are connected to all other neurons,
with the enhanced cell-specific parameters from NCPMap.
"""

from typing import Optional, Tuple, Dict, Any

from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring.ncp_map import NCPMap

class FullyConnectedNCPMap(NCPMap):
    """
    Fully connected NCP wiring configuration.
    
    In a fully connected wiring, all neurons are connected to all other neurons,
    and all inputs are connected to all neurons. This class inherits from NCPMap
    to provide the enhanced cell-specific parameters.
    """
    
    def __init__(
        self,
        units: int,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        sparsity_level: float = 0.0,
        seed: Optional[int] = None,
        # Cell-specific parameters
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
        output_mapping: str = "affine"
    ):
        """
        Initialize a fully connected NCP wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.0)
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
        """
        # Calculate inter_neurons, command_neurons, and motor_neurons
        # For fully connected, we'll use a simple distribution:
        # - 50% inter neurons
        # - 25% command neurons
        # - 25% motor neurons
        inter_neurons = units // 2
        command_neurons = units // 4
        motor_neurons = units - inter_neurons - command_neurons
        sensory_neurons = input_dim if input_dim is not None else 0
        
        # Call NCPMap constructor with the calculated neuron distribution
        super().__init__(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=motor_neurons,
            sensory_neurons=sensory_neurons,
            sparsity_level=sparsity_level,
            seed=seed,
            time_scale_factor=time_scale_factor,
            activation=activation,
            recurrent_activation=recurrent_activation,
            mode=mode,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            mixed_memory=mixed_memory,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            units=units,
            output_dim=output_dim,
            input_dim=input_dim
        )
    
    def build(self, input_dim=None) -> Tuple:
        """
        Build the fully connected wiring configuration.
        
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
        recurrent_mask = tensor.ones((self.units, self.units), dtype=tensor.int32)
        input_mask = tensor.ones((self.input_dim,), dtype=tensor.int32)
        
        # Create output mask (only motor neurons contribute to output)
        output_mask = tensor.zeros((self.units,), dtype=tensor.int32)
        
        # Set motor neurons to 1
        motor_start = self.sensory_neurons + self.inter_neurons + self.command_neurons
        motor_end = min(motor_start + self.motor_neurons, self.units)  # Ensure we don't go out of bounds
        
        # Create a list of zeros with the correct length
        output_mask_list = [0] * self.units
        
        # Set motor neurons to 1, ensuring we don't go out of bounds
        for i in range(motor_start, motor_end):
            if i < self.units:  # Extra safety check
                output_mask_list[i] = 1
        
        # Create a new tensor with the updated values
        output_mask = tensor.convert_to_tensor(output_mask_list, dtype=tensor.int32)
        
        # Apply sparsity if needed
        if self.sparsity_level > 0.0:
            # Create sparse masks
            input_mask = tensor.cast(
                tensor.random_uniform((self.input_dim,)) >= self.sparsity_level,
                tensor.int32
            )
            recurrent_mask = tensor.cast(
                tensor.random_uniform((self.units, self.units)) >= self.sparsity_level,
                tensor.int32
            )
        
        self._built = True # Mark map as built
        return input_mask, recurrent_mask, output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the FullyConnectedNCPMap."""
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FullyConnectedNCPMap':
        """
        Create a FullyConnectedNCPMap from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            FullyConnectedNCPMap instance
        """
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        
        # Extract the parameters that FullyConnectedNCPMap.__init__ accepts
        units = config_copy.pop('units', None)
        if units is None:
            # Calculate units from neuron distribution if not provided
            inter_neurons = config_copy.pop('inter_neurons', 0)
            command_neurons = config_copy.pop('command_neurons', 0)
            motor_neurons = config_copy.pop('motor_neurons', 0)
            units = inter_neurons + command_neurons + motor_neurons
        
        # Extract the parameters that FullyConnectedNCPMap.__init__ accepts
        accepted_params = {
            'output_dim': config_copy.pop('output_dim', None),
            'input_dim': config_copy.pop('input_dim', None),
            'sparsity_level': config_copy.pop('sparsity_level', 0.0),
            'seed': config_copy.pop('seed', None),
            'time_scale_factor': config_copy.pop('time_scale_factor', 1.0),
            'activation': config_copy.pop('activation', 'tanh'),
            'recurrent_activation': config_copy.pop('recurrent_activation', 'sigmoid'),
            'mode': config_copy.pop('mode', 'default'),
            'use_bias': config_copy.pop('use_bias', True),
            'kernel_initializer': config_copy.pop('kernel_initializer', 'glorot_uniform'),
            'recurrent_initializer': config_copy.pop('recurrent_initializer', 'orthogonal'),
            'bias_initializer': config_copy.pop('bias_initializer', 'zeros'),
            'mixed_memory': config_copy.pop('mixed_memory', False),
            'ode_unfolds': config_copy.pop('ode_unfolds', 6),
            'epsilon': config_copy.pop('epsilon', 1e-8),
            'implicit_param_constraints': config_copy.pop('implicit_param_constraints', False),
            'input_mapping': config_copy.pop('input_mapping', 'affine'),
            'output_mapping': config_copy.pop('output_mapping', 'affine')
        }
        
        # Filter out None values
        accepted_params = {k: v for k, v in accepted_params.items() if v is not None}
        
        # Create the instance with the filtered parameters
        return cls(units=units, **accepted_params)