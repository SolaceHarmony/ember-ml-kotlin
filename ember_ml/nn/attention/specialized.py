"""
Specialized neuron implementations with role-specific behaviors.

This module provides the SpecializedNeuron class that implements different
functional roles such as memory, inhibition, and amplification using the
CfC (Closed-form Continuous-time) architecture.
"""

from typing import Union, List, Literal
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.rnn import CfC
from ember_ml.nn.modules.wiring import FullyConnectedMap


# Define valid role types
RoleType = Literal["default", "memory", "inhibition", "amplification"]


class SpecializedNeuron(Module):
    """
    A neuron with role-specific behavior modifications.
    
    This class implements different functional roles using CfC architecture:
    - memory: Slower decay for maintaining information
    - inhibition: Dampens input signals
    - amplification: Strengthens input signals
    - default: Standard CfC behavior
    
    Attributes:
        role (str): The neuron's functional role
        state (EmberTensor): Current activation state
        base_tau (float): Base time constant
        num_inputs (int): Number of input connections
    """
    
    # Role-specific parameters
    MEMORY_TAU_FACTOR = 1.5      # Slower decay for memory
    INHIBITION_FACTOR = 0.5      # Dampening factor for inhibition
    AMPLIFICATION_FACTOR = 1.5   # Strengthening factor for amplification
    
    def __init__(self,
                 tau: float = 1.0,
                 role: RoleType = "default",
                 num_inputs: int = 3):
        """
        Initialize a specialized neuron with a specific role.
        
        Args:
            tau: Base time constant for the neuron
            role: Functional role determining behavior
            num_inputs: Number of input connections
        """
        super().__init__()
        self.role = role
        self.base_tau = tau
        self.num_inputs = num_inputs
        # Initialize state as a scalar tensor
        self.state = tensor.convert_to_tensor(0.0, tensor.float32)
        
        # Create a neuron map for the CfC layer
        self.neuron_map = FullyConnectedMap(
            units=1,  # Single neuron
            output_dim=1,
            input_dim=num_inputs
        )
        
        # Create a CfC layer with appropriate time scale factor
        time_scale_factor = tau
        if role == "memory":
            time_scale_factor = ops.multiply(
                tensor.convert_to_tensor(tau, tensor.float32),
                tensor.convert_to_tensor(self.MEMORY_TAU_FACTOR, tensor.float32)
            )
            
        # Convert FullyConnectedMap to NCPMap for CfC
        from ember_ml.nn.modules.wiring import NCPMap
        
        # Create an NCPMap with equivalent structure
        self.ncp_map = NCPMap(
            inter_neurons=0,
            command_neurons=0,
            motor_neurons=1,
            sensory_neurons=num_inputs
        )
        
        self.cfc = CfC(
            neuron_map=self.ncp_map,
            time_scale_factor=time_scale_factor,
            return_sequences=False,
            return_state=True
        )
        
        # Initialize weights
        self.weights = tensor.random_uniform((num_inputs,), -0.5, 0.5)
    
    def update(self,
              inputs: Union[List[float], EmberTensor],
              dt: float = 0.1,
              tau_mod: float = 1.0,
              feedback: float = 0.0) -> float:
        """
        Update the neuron's state based on its role and inputs.
        
        Args:
            inputs: Input signals to the neuron
            dt: Time step size for integration
            tau_mod: Modifier for the time constant
            feedback: Additional feedback input
            
        Returns:
            float: Updated neuron state
        """
        # Import tensor and ops here to avoid circular imports
        from ember_ml import ops
        from ember_ml.nn import tensor
        
        # Ensure inputs are tensor
        input_tensor = tensor.convert_to_tensor(inputs, tensor.float32)
        
        # Add batch and sequence dimensions
        input_tensor = tensor.expand_dims(input_tensor, 0)  # Add batch dimension
        input_tensor = tensor.expand_dims(input_tensor, 0)  # Add sequence dimension
        
        # Add feedback to input if provided
        if feedback != 0.0:
            feedback_tensor = tensor.convert_to_tensor(feedback, tensor.float32)
            # Create a tensor of the same shape as input_tensor but with feedback value
            feedback_tensor = tensor.full_like(input_tensor, feedback_tensor)
            input_tensor = ops.add(input_tensor, feedback_tensor)
        
        # Create time delta tensor
        time_delta = tensor.convert_to_tensor(dt, tensor.float32)
        time_delta = tensor.expand_dims(time_delta, 0)  # Add batch dimension
        time_delta = tensor.expand_dims(time_delta, 0)  # Add sequence dimension
        
        # Apply role-specific modifications to the input
        if self.role == "inhibition":
            # Inhibition role: dampen signals
            inhibition_factor = tensor.convert_to_tensor(self.INHIBITION_FACTOR, tensor.float32)
            input_tensor = ops.multiply(ops.negative(input_tensor), inhibition_factor)
        elif self.role == "amplification":
            # Amplification role: strengthen signals
            amp_factor = tensor.convert_to_tensor(self.AMPLIFICATION_FACTOR, tensor.float32)
            input_tensor = ops.multiply(input_tensor, amp_factor)
        
        # Create initial state for CfC
        # CfC expects a list of two tensors: [h_state, t_state]
        h0 = tensor.zeros((1, 1))  # (batch_size, units)
        t0 = tensor.zeros((1, 1))  # (batch_size, units)
        initial_state = [h0, t0]
        
        # Update state using CfC
        _, self.state = self.cfc(input_tensor, initial_state=initial_state, time_deltas=time_delta)
        
        # Extract scalar value from state
        h_state = self.state[0]  # h state
        state_value = tensor.item(h_state[0, 0])  # Extract scalar value
        
        return state_value
    
    def reset(self) -> None:
        """Reset the neuron's state to zero."""
        # Reset state to a scalar tensor
        self.state = tensor.convert_to_tensor(0.0, tensor.float32)
    
    def get_weights(self) -> EmberTensor:
        """
        Get the current weight vector.
        
        Returns:
            EmberTensor: Copy of the weight vector
        """
        return tensor.copy(self.weights)
    
    def set_weights(self, weights: EmberTensor) -> None:
        """
        Set new weights for the neuron.
        
        Args:
            weights: New weight values to use
            
        Raises:
            ValueError: If weights length doesn't match num_inputs
        """
        weights_shape = tensor.shape(weights)
        expected_shape = (self.num_inputs,)
        
        if weights_shape != expected_shape:
            raise ValueError(
                f"Weights must have shape {expected_shape}, "
                f"got {weights_shape}"
            )
        self.weights = tensor.copy(weights)
    
    @property
    def role_name(self) -> str:
        """
        Get a human-readable name for the neuron's role.
        
        Returns:
            str: Capitalized role name
        """
        return self.role.capitalize()
    
    def __repr__(self) -> str:
        """
        Get a string representation of the neuron.
        
        Returns:
            str: Neuron description including role and tau
        """
        return (f"SpecializedNeuron(role={self.role}, "
                f"tau={self.base_tau:.2f}, "
                f"num_inputs={self.num_inputs})")