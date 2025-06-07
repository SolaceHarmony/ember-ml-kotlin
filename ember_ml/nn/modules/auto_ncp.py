"""
Auto Neural Circuit Policy (AutoNCP) module.

This module provides the AutoNCP class, which is a convenience wrapper
around the NCP class that automatically configures the wiring.
"""

from typing import Optional, Dict, Any

from ember_ml.nn.modules.ncp import NCP
from ember_ml.nn.modules.wiring import NCPMap # Import from wiring subpackage

class AutoNCP(NCP):
    """
    Auto Neural Circuit Policy (AutoNCP) module.
    
    This module is a convenience wrapper around the NCP class that
    automatically configures the wiring based on the number of units
    and outputs.
    """
    
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        dtype: Optional[Any] = None,
    ):
        """
        Initialize an AutoNCP module.
        
        Args:
            units: Number of units in the circuit
            output_size: Number of output dimensions
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
            activation: Activation function to use
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias weights
            dtype: Data type for the weights
        """
        if output_size >= units - 2:
            raise ValueError(
                f"Output size must be less than the number of units-2 (given {units} units, {output_size} output size)"
            )
        if sparsity_level < 0.1 or sparsity_level > 1.0:
            raise ValueError(
                f"Sparsity level must be between 0.0 and 0.9 (given {sparsity_level})"
            )
        
        # Calculate the number of inter and command neurons
        density_level = 1.0 - sparsity_level
        inter_and_command_neurons = units - output_size
        command_neurons = max(int(0.4 * inter_and_command_neurons), 1)
        inter_neurons = inter_and_command_neurons - command_neurons
        
        # Calculate the fanout and fanin parameters
        sensory_fanout = max(int(inter_neurons * density_level), 1)
        inter_fanout = max(int(command_neurons * density_level), 1)
        recurrent_command_synapses = max(int(command_neurons * density_level * 2), 1)
        motor_fanin = max(int(command_neurons * density_level), 1)
        
        # Create the wiring
        wiring = NCPMap( # Use renamed NCPMap
            inter_neurons=inter_neurons,
            command_neurons=command_neurons, # Pass calculated command_neurons
            motor_neurons=output_size,
            sensory_neurons=0,  # No sensory neurons in AutoNCP
            sparsity_level=sparsity_level,
            seed=seed,
            # input_dim will be inferred later during build
            units=units,  # Set units explicitly
        )
        
        # Initialize the NCP module
        super().__init__(
            neuron_map=wiring,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype,
        )
        
        # The parameters are already stored in the neuron_map (wiring)
        # We don't need to duplicate them as direct attributes
        # These can be accessed via self.neuron_map.units, self.neuron_map.output_dim, etc.
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the AutoNCP module.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        # Get values from neuron_map instead of direct attributes
        config.update({
            "units": self.neuron_map.units,
            "output_size": self.neuron_map.output_dim,
            "sparsity_level": self.neuron_map.sparsity_level,
            "seed": self.neuron_map.seed,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoNCP': # Return type is correct (AutoNCP Layer)
        """
        Create an AutoNCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AutoNCP module
        """
        # Remove the wiring-related parameters and neuron_map from the config
        config.pop("wiring", None)
        config.pop("wiring_class", None)
        config.pop("neuron_map", None)
        
        # Remove input_size to avoid duplicate parameter errors
        config.pop("input_size", None)
        
        # Create the AutoNCP module
        return cls(**config)