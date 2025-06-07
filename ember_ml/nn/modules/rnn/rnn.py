"""
Recurrent Neural Network (RNN) Layer

This module provides an implementation of the RNN layer,
which wraps an RNNCell to create a recurrent layer.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules.activations import get_activation
# Import Dropout module from its new location
from ember_ml.nn.modules.activations import Dropout
class RNN(Module):
    """
    Recurrent Neural Network (RNN) layer.
    
    This layer wraps an RNNCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        activation: str = "tanh",
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = True,
        return_state: bool = False,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        **kwargs
    ):
        """
        Initialize the RNN layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of RNN layers
            activation: Activation function to use
            bias: Whether to use bias
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            dropout: Dropout probability (applied between layers)
            bidirectional: Whether to use a bidirectional RNN
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # RNNCell parameters
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        
        # Create RNN cells for each layer and direction
        self.forward_cells = []
        self.backward_cells = []
        
        # Input size for the first layer is the input size
        # For subsequent layers, it's the hidden size (or 2x hidden size if bidirectional)
        layer_input_size = input_size
        
        for layer in range(num_layers):
            # Initialize parameters for this layer
            self._initialize_layer_parameters(layer, layer_input_size)
            
            # Update input size for the next layer
            if self.bidirectional:
                layer_input_size = ops.multiply(hidden_size, 2)
            else:
                layer_input_size = hidden_size
                
    def _initialize_layer_parameters(self, layer, input_size):
        """Initialize parameters for a specific layer."""
        # Input weights
        self.input_kernels = getattr(self, 'input_kernels', [])
        self.recurrent_kernels = getattr(self, 'recurrent_kernels', [])
        self.biases = getattr(self, 'biases', [])
        
        # Backward direction weights
        self.backward_input_kernels = getattr(self, 'backward_input_kernels', [])
        self.backward_recurrent_kernels = getattr(self, 'backward_recurrent_kernels', [])
        self.backward_biases = getattr(self, 'backward_biases', [])
        
        # Forward direction weights
        kernel_shape = (input_size, self.hidden_size)
        recurrent_shape = (self.hidden_size, self.hidden_size)
        
        # Initialize forward weights
        self.input_kernels.append(Parameter(glorot_uniform(kernel_shape)))
        self.recurrent_kernels.append(Parameter(orthogonal(recurrent_shape)))
        
        if self.use_bias:
            bias_shape = (self.hidden_size,)
            self.biases.append(Parameter(tensor.zeros(bias_shape)))
        
        # Initialize backward weights if bidirectional
        if self.bidirectional:
            self.backward_input_kernels.append(Parameter(glorot_uniform(kernel_shape)))
            self.backward_recurrent_kernels.append(Parameter(orthogonal(recurrent_shape)))
            
            if self.use_bias:
                self.backward_biases.append(Parameter(tensor.zeros(bias_shape)))
    
    def forward(self, inputs, initial_state=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            
        Returns:
            Layer output and final state
        """
        # Get batch information
        is_batched = len(tensor.shape(inputs)) == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        
        # Handle non-batched inputs
        if not is_batched:
            inputs = tensor.expand_dims(inputs, batch_dim)
        
        # Get batch size and sequence length
        input_shape = tensor.shape(inputs)
        batch_size = input_shape[batch_dim]
        seq_length = input_shape[seq_dim]
        
        # Initialize states if not provided
        if initial_state is None:
            # Create initial states for each layer and direction
            h_states = []
            
            for layer in range(self.num_layers):
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
                
                if self.bidirectional:
                    h_states.append(tensor.zeros((batch_size, self.hidden_size)))
        else:
            # Unpack provided initial states
            h_states = initial_state
        
        # Process sequence through each layer
        layer_outputs = inputs
        final_h_states = []
        
        for layer in range(self.num_layers):
            # Get parameters for this layer
            input_kernel = self.input_kernels[layer]
            recurrent_kernel = self.recurrent_kernels[layer]
            bias = self.biases[layer] if self.use_bias else None
            
            if self.bidirectional:
                backward_input_kernel = self.backward_input_kernels[layer]
                backward_recurrent_kernel = self.backward_recurrent_kernels[layer]
                backward_bias = self.backward_biases[layer] if self.use_bias else None
            
            # Get initial states for this layer
            layer_idx = layer * (2 if self.bidirectional else 1)
            forward_h = h_states[layer_idx]
            
            if self.bidirectional:
                backward_h = h_states[layer_idx + 1]
            
            # Process sequence for this layer
            forward_outputs = []
            backward_outputs = []
            
            # Forward direction
            for t in range(seq_length):
                # Get input for current time step
                if self.batch_first:
                    current_input = layer_outputs[:, t]
                else:
                    current_input = layer_outputs[t]
                
                # RNN forward pass implementation
                activation_fn = get_activation(self.activation)
                
                # Project input
                z = ops.matmul(current_input, input_kernel)
                z = ops.add(z, ops.matmul(forward_h, recurrent_kernel))
                if self.use_bias:
                    z = ops.add(z, bias)
                
                # Apply activation
                forward_h = activation_fn(z)
                forward_outputs.append(forward_h)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                for t in range(seq_length - 1, -1, -1):
                    # Get input for current time step
                    if self.batch_first:
                        current_input = layer_outputs[:, t]
                    else:
                        current_input = layer_outputs[t]
                    
                    # RNN backward pass implementation
                    activation_fn = get_activation(self.activation)
                    
                    # Project input
                    z = ops.matmul(current_input, backward_input_kernel)
                    z = ops.add(z, ops.matmul(backward_h, backward_recurrent_kernel))
                    if self.use_bias:
                        z = ops.add(z, backward_bias)
                    
                    # Apply activation
                    backward_h = activation_fn(z)
                    backward_outputs.insert(0, backward_h)
            
            # Combine outputs
            if self.bidirectional:
                combined_outputs = []
                for t in range(seq_length):
                    combined = tensor.concatenate([forward_outputs[t], backward_outputs[t]], axis=-1)
                    combined_outputs.append(combined)
            else:
                combined_outputs = forward_outputs
            
            # Stack outputs for this layer
            if self.batch_first:
                layer_outputs = tensor.stack(combined_outputs, axis=1)
            else:
                layer_outputs = tensor.stack(combined_outputs, axis=0)
            
            # Apply dropout (except for the last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                # Instantiate and apply Dropout module
                # Need to handle the training state appropriately
                # Assuming a 'training=True' context for dropout during this layer's forward pass
                # A more robust solution would pass a training flag down
                dropout_layer = Dropout(rate=self.dropout)
                # Pass training=True, assuming this forward call is during training
                layer_outputs = dropout_layer(layer_outputs, training=True)
            
            # Store final states for this layer
            final_h_states.append(forward_h)
            
            if self.bidirectional:
                final_h_states.append(backward_h)
        
        # Prepare output
        if not self.return_sequences:
            if self.batch_first:
                outputs = layer_outputs[:, -1]
            else:
                outputs = layer_outputs[-1]
        else:
            outputs = layer_outputs
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = tensor.squeeze(outputs, batch_dim)
        
        # Prepare final state
        # Return final state as a list containing the stacked tensor(s)
        # for consistency with other RNN layers' state format
        final_state = [tensor.stack(final_h_states)]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs, final_state
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
        h_states = []
        
        for layer in range(self.num_layers):
            h_states.append(tensor.zeros((batch_size, self.hidden_size)))
            
            if self.bidirectional:
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
        
        return tensor.stack(h_states)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the RNN layer."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "activation": self.activation, # Save activation name
            "bias": self.bias,
            "batch_first": self.batch_first,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer
        })
        # Cell is reconstructed in __init__ based on these args
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RNN':
        """Creates an RNN layer from its configuration."""
        # BaseModule.from_config handles calling cls(**config)
        return super(RNN, cls).from_config(config)