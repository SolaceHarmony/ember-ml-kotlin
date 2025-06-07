"""
Continuous-Time Recurrent Neural Network (CTRNN) Implementation

This module provides an implementation of CTRNN layers,
which are a type of recurrent neural network that operates in continuous time.
This implementation directly uses NeuronMap for both structure and dynamics.
"""

from typing import Dict, Any, Optional, List, Tuple, Union

from ember_ml import ops
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation

class CTRNN(Module):
    """
    Continuous-Time Recurrent Neural Network (CTRNN) layer.
    
    This layer implements a classic CTRNN where the state evolves according to
    a differential equation with time-varying parameters. The implementation
    directly uses NeuronMap for both structure and dynamics.
    
    The CTRNN implements the following dynamics:
        dh/dt = (-h + σ(W*x + U*h + b)) / τ
    where:
        - h is the hidden state
        - x is the input
        - W is the input weight matrix
        - U is the recurrent weight matrix
        - b is the bias vector
        - σ is the activation function
        - τ is the time constant
    """
    
    def __init__(
        self,
        neuron_map: NCPMap,
        global_feedback: bool = False,
        cell_clip: Optional[float] = None,
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
        **kwargs
    ):
        """
        Initialize the CTRNN layer.
        
        Args:
            neuron_map: NCPMap instance defining both structure and dynamics
            global_feedback: Whether to use global feedback
            cell_clip: Optional value to clip cell outputs
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
        
        # Store the neuron map and layer parameters
        self.neuron_map = neuron_map
        self.global_feedback = global_feedback
        self.cell_clip = cell_clip
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.batch_first = batch_first
        
        # Set input_size from neuron_map.input_dim if the map is already built
        # Otherwise, it will be set during the first forward pass
        self.input_size = getattr(neuron_map, 'input_dim', None)
        
        # Initialize parameters
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.tau = None
        self.built = False
    
    def build(self, input_shape):
        """Build the CTRNN layer."""
        # Get input dimension
        if len(input_shape) == 3:  # (batch, time, features) or (time, batch, features)
            feature_dim = 2
            input_dim = input_shape[feature_dim]
        else:
            input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Set input_size
        self.input_size = self.neuron_map.input_dim
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize weights for input connections
        self.kernel = Parameter(glorot_uniform((self.input_size, units)))
        
        # Initialize weights for recurrent connections
        self.recurrent_kernel = Parameter(orthogonal((units, units)))
        
        # Initialize bias
        self.bias = Parameter(tensor.zeros((units,)))
        
        # Initialize time constant (tau)
        self.tau = Parameter(0.1 * tensor.ones((units,)))
        
        # Mark as built
        self.built = True
    
    def _update_state(self, inputs, state, elapsed_time):
        """Update the state using CTRNN dynamics."""
        # Get parameters from neuron_map
        epsilon = self.neuron_map.epsilon
        
        # Get activation function
        activation_fn = get_activation(self.neuron_map.activation)
        
        # Project input
        net = ops.matmul(inputs, self.kernel)
        
        # Add recurrent connection
        net = ops.add(net, ops.matmul(state, self.recurrent_kernel))
        
        # Add bias
        net = ops.add(net, self.bias)
        
        # Apply activation to get target state
        target_state = activation_fn(net)
        
        # Update state using continuous-time dynamics
        d_state = ops.divide(
            ops.subtract(target_state, state),
            ops.add(self.tau, epsilon)
        )
        
        # Apply time scaling
        new_state = ops.add(state, ops.multiply(elapsed_time, d_state))
        
        # Apply cell clipping if specified
        if self.cell_clip is not None:
            new_state = ops.clip(new_state, -self.cell_clip, self.cell_clip)
        
        return new_state
    
    @property
    def state_size(self):
        return self.neuron_map.units
    
    @property
    def output_size(self):
        return self.neuron_map.output_dim
    
    def forward(self, inputs, initial_state=None, timespans=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            timespans: Time spans for continuous-time dynamics (default: 1.0)
            
        Returns:
            Layer output and final state if return_state is True, otherwise just the layer output
        """
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get device and batch information
        is_batched = len(tensor.shape(inputs)) == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # Handle non-batched inputs
        if not is_batched:
            inputs = tensor.expand_dims(inputs, batch_dim)
            if timespans is not None:
                timespans = tensor.expand_dims(timespans, batch_dim)
        
        # Get batch size and sequence length
        input_shape = tensor.shape(inputs)
        batch_size = input_shape[batch_dim]
        seq_length = input_shape[seq_dim]
        
        # Initialize state if not provided
        if initial_state is None:
            state = tensor.zeros((batch_size, self.state_size))
        else:
            state = initial_state
            
            # Handle non-batched states
            if is_batched and len(tensor.shape(state)) != 2:
                raise ValueError(
                    f"For batched inputs, initial_state should be 2D but got {len(tensor.shape(state))}D"
                )
            elif not is_batched and len(tensor.shape(state)) != 1:
                # Add batch dimension for non-batched states
                state = tensor.expand_dims(state, 0)
        
        # Process sequence
        output_sequence = []
        for t in range(seq_length):
            # Get input for current time step
            if self.batch_first:
                current_input = inputs[:, t]
                ts = 1.0 if timespans is None else timespans[:, t]
            else:
                current_input = inputs[t]
                ts = 1.0 if timespans is None else timespans[t]
            
            # Update state using CTRNN dynamics
            state = self._update_state(current_input, state, ts)
            
            # Apply activation to state for output
            output = get_activation(self.neuron_map.activation)(state)
            
            # Store output
            if self.return_sequences:
                output_sequence.append(output)
            else:
                # Only store the last output
                output_sequence = [output]
        
        # Prepare output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            outputs = tensor.stack(output_sequence, axis=stack_dim)
        else:
            # If not returning sequences, use the last output
            outputs = output_sequence[-1] if output_sequence else None
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = tensor.squeeze(outputs, batch_dim)
            state = tensor.squeeze(state, 0)
        
        if self.return_state:
            return outputs, state
        else:
            return outputs
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return tensor.zeros((batch_size, self.state_size))
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the CTRNN layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "global_feedback": self.global_feedback,
            "cell_clip": self.cell_clip,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "batch_first": self.batch_first
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CTRNN':
        """Creates a CTRNN layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)