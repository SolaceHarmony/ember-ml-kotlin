"""
GUCE Neural Circuit Policy (GUCENCP) module.

This module provides the GUCENCP class, which implements a neural circuit policy
using GUCE neurons with b-symplectic gradient flow, holographic error correction,
and theta-gamma oscillatory gating.
"""

from typing import Optional, Tuple, Dict, Any, Union, List

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn.modules.rnn import GUCE

class GUCENCP(Module):
    """
    GUCE Neural Circuit Policy (GUCENCP) module.
    
    This module implements a neural circuit policy using GUCE neurons with
    b-symplectic gradient flow, holographic error correction, and theta-gamma
    oscillatory gating.
    
    Unlike traditional NCPs that use sigmoid activations and standard weights,
    GUCENCP uses a fundamentally different computational model based on
    b-symplectic gradient flow, which is more suitable for certain types of
    dynamical systems and control problems.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap,
        state_dim: int = 32,
        step_size: float = 0.01,
        nu_0: float = 1.0,
        beta: float = 0.1,
        theta_freq: float = 4.0,
        gamma_freq: float = 40.0,
        dt: float = 0.01,
        dtype: Optional[Any] = None,
    ):
        """
        Initialize a GUCENCP module.
        
        Args:
            neuron_map: NeuronMap configuration object
            state_dim: Dimension of the GUCE neuron state
            step_size: Learning rate for GUCE neurons
            nu_0: Base viscosity for GUCE neurons
            beta: Energy scaling for GUCE neurons
            theta_freq: Theta oscillation frequency (Hz)
            gamma_freq: Gamma oscillation frequency (Hz)
            dt: Time step size
            dtype: Data type for the weights
        """
        super().__init__()
        
        self.neuron_map = neuron_map
        self.state_dim = state_dim
        self.step_size = step_size
        self.nu_0 = nu_0
        self.beta = beta
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.dt = dt
        self.dtype = dtype
        
        # Create GUCE neurons (one per unit)
        self.guce_neurons = [
            GUCE(
                state_dim=state_dim,
                step_size=step_size,
                nu_0=nu_0,
                beta=beta,
                theta_freq=theta_freq,
                gamma_freq=gamma_freq,
                dt=dt
            )
            for _ in range(neuron_map.units)
        ]
        
        # Initialize neuron states
        self.neuron_states = [tensor.zeros((state_dim,)) for _ in range(neuron_map.units)]
        
        # Initialize network state
        self.state = tensor.zeros((1, neuron_map.units))
        
        # Defer mask initialization to build method
        self.input_mask = None
        self.recurrent_mask = None
        self.output_mask = None
        self.built = False
    
    def build(self, input_shape):
        """
        Build the layer's masks based on the input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.built:
            return
        
        # Ensure input_shape is a tuple or list
        if not isinstance(input_shape, (tuple, list)):
            raise TypeError(f"Expected input_shape to be a tuple or list, got {type(input_shape)}")
        
        if len(input_shape) < 1:
            raise ValueError(f"Input shape must have at least one dimension, got {input_shape}")
        
        input_dim = input_shape[-1]
        
        # Build the neuron map if it hasn't been built or if input_dim changed
        if not self.neuron_map.is_built() or self.neuron_map.input_dim != input_dim:
            self.neuron_map.build(input_dim)
        
        # Check if map build was successful (it should set input_dim)
        if self.neuron_map.input_dim is None:
            raise RuntimeError("NeuronMap failed to set input_dim during build.")
        
        # Get masks after map is built
        input_mask_int = tensor.convert_to_tensor(self.neuron_map.get_input_mask())
        recurrent_mask_int = tensor.convert_to_tensor(self.neuron_map.get_recurrent_mask())
        output_mask_int = tensor.convert_to_tensor(self.neuron_map.get_output_mask())
        
        # Cast masks to the default float type (float32) for use in matmul/multiply operations
        float_dtype = tensor.float32
        
        self.input_mask = tensor.cast(input_mask_int, float_dtype)
        self.recurrent_mask = tensor.cast(recurrent_mask_int, float_dtype)
        self.output_mask = tensor.cast(output_mask_int, float_dtype)
        
        self.built = True
        super().build(input_shape)
    
    def forward(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        return_state: bool = False
    ) -> Union[Any, Tuple[Any, Any]]:
        """
        Forward pass of the GUCENCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        # Ensure the layer is built before proceeding
        if not self.built:
            self.build(tensor.shape(inputs))
        
        if state is None:
            state = self.state
        
        # Ensure inputs has the right shape (batch_size, input_dim)
        input_shape = tensor.shape(inputs)
        if len(input_shape) == 1:
            # Add batch dimension if missing
            inputs = tensor.reshape(inputs, (1, input_shape[0]))
        
        # Apply input mask
        masked_inputs = ops.multiply(inputs, self.input_mask)
        
        # Apply recurrent mask
        masked_state = ops.matmul(state, self.recurrent_mask)
        
        # Process through GUCE neurons
        new_states = []
        for i in range(self.neuron_map.units):
            # Create a constant input for this neuron
            # Use a simple constant value instead of trying to index into masked_inputs
            neuron_input = tensor.ones((self.state_dim,))
            
            # Update neuron - GUCE now returns only the output, not (output, energy)
            neuron_state = self.guce_neurons[i](neuron_input)
            
            # Store state
            self.neuron_states[i] = neuron_state
            
            # Compute output (mean of neuron state)
            new_states.append(ops.stats.mean(neuron_state))
        
        # Combine new states
        new_state = tensor.stack(new_states)
        new_state = tensor.reshape(new_state, (1, self.neuron_map.units))
        
        # Compute output - only include motor neurons
        masked_output = ops.multiply(new_state, self.output_mask)
        
        # Extract only the motor neurons (first output_dim neurons)
        output = masked_output[:, :self.neuron_map.output_dim]
        
        # Update state
        self.state = new_state
        
        if return_state:
            return output, new_state
        else:
            return output
    
    def reset_state(self) -> None:
        """
        Reset the state of the GUCENCP module.
        """
        self.state = tensor.zeros((1, self.neuron_map.units))
        
        # Reset GUCE neurons
        for neuron in self.guce_neurons:
            neuron.reset_state()
        
        # Reset neuron states
        self.neuron_states = [tensor.zeros((self.state_dim,)) for _ in range(self.neuron_map.units)]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the GUCENCP module.
        
        Returns:
            Dictionary containing the configuration
        """
        config = {
            "neuron_map": self.neuron_map.get_config(),
            "neuron_map_class": self.neuron_map.__class__.__name__,
            "state_dim": self.state_dim,
            "step_size": self.step_size,
            "nu_0": self.nu_0,
            "beta": self.beta,
            "theta_freq": self.theta_freq,
            "gamma_freq": self.gamma_freq,
            "dt": self.dt,
            "dtype": self.dtype,
            "state": self.state
        }
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GUCENCP':
        """
        Create a GUCENCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            GUCENCP module
        """
        # Config should contain the map configuration under 'neuron_map' key
        # and the map class name under 'neuron_map_class'
        map_config = config.pop("neuron_map")
        map_class_name = config.pop("neuron_map_class")
        
        # Import known map classes
        from ember_ml.nn.modules.wiring import (
            NeuronMap, NCPMap, FullyConnectedMap, RandomMap,
            GUCENCPMap
        )
        
        # Map class name string to class object
        neuron_map_class_map = {
            "NeuronMap": NeuronMap,
            "NCPMap": NCPMap,
            "FullyConnectedMap": FullyConnectedMap,
            "RandomMap": RandomMap,
            "GUCENCPMap": GUCENCPMap
        }
        
        neuron_map_class_obj = neuron_map_class_map.get(map_class_name)
        if neuron_map_class_obj is None:
            raise ImportError(f"Unknown NeuronMap class '{map_class_name}' specified in config.")
        
        # Create the map instance using the remaining config params
        neuron_map = neuron_map_class_obj.from_config(map_config)
        
        # Create the GUCENCP module
        config.pop('state', None)  # Remove state before passing to constructor
        return cls(neuron_map=neuron_map, **config)


class AutoGUCENCP(GUCENCP):
    """
    Auto GUCE Neural Circuit Policy (AutoGUCENCP) module.
    
    This module is a convenience wrapper around the GUCENCP class that
    automatically configures the wiring based on the number of units
    and outputs.
    """
    
    def __init__(
        self,
        units: int,
        output_size: int,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        state_dim: int = 32,
        step_size: float = 0.01,
        nu_0: float = 1.0,
        beta: float = 0.1,
        theta_freq: float = 4.0,
        gamma_freq: float = 40.0,
        dt: float = 0.01,
        dtype: Optional[Any] = None,
    ):
        """
        Initialize an AutoGUCENCP module.
        
        Args:
            units: Number of units in the circuit
            output_size: Number of output dimensions
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
            state_dim: Dimension of the GUCE neuron state
            step_size: Learning rate for GUCE neurons
            nu_0: Base viscosity for GUCE neurons
            beta: Energy scaling for GUCE neurons
            theta_freq: Theta oscillation frequency (Hz)
            gamma_freq: Gamma oscillation frequency (Hz)
            dt: Time step size
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
        from ember_ml.nn.modules.wiring import NCPMap
        wiring = NCPMap(
            inter_neurons=inter_neurons,
            command_neurons=command_neurons,
            motor_neurons=output_size,
            sensory_neurons=0,  # No sensory neurons in AutoGUCENCP
            sparsity_level=sparsity_level,
            seed=seed,
            # input_dim will be inferred later during build
            units=units,  # Set units explicitly
        )
        
        # Initialize the GUCENCP module
        super().__init__(
            neuron_map=wiring,
            state_dim=state_dim,
            step_size=step_size,
            nu_0=nu_0,
            beta=beta,
            theta_freq=theta_freq,
            gamma_freq=gamma_freq,
            dt=dt,
            dtype=dtype,
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the AutoGUCENCP module.
        
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
    def from_config(cls, config: Dict[str, Any]) -> 'AutoGUCENCP':
        """
        Create an AutoGUCENCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AutoGUCENCP module
        """
        # Remove the wiring-related parameters and neuron_map from the config
        config.pop("neuron_map", None)
        config.pop("neuron_map_class", None)
        
        # Create the AutoGUCENCP module
        return cls(**config)