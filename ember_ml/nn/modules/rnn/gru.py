"""
Gated Recurrent Unit (GRU) Layer

This module provides an implementation of the GRU layer,
which wraps a GRUCell to create a recurrent layer.
"""

from typing import Dict, Any

from ember_ml import ops
from ember_ml.nn.modules import Module
# Removed GRUCell import
from ember_ml.nn import tensor

class GRU(Module):
    """
    Gated Recurrent Unit (GRU) RNN layer.
    
    This layer wraps a GRUCell to create a recurrent layer.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = True,
        return_state: bool = False,
        use_bias: bool = True,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        **kwargs
    ):
        """
        Initialize the GRU layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of GRU layers
            bias: Whether to use bias
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            dropout: Dropout probability (applied between layers)
            bidirectional: Whether to use a bidirectional GRU
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # GRUCell parameters
        self.use_bias = use_bias
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        
        # Initialize parameters for GRU cells
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize parameters for GRU cells."""
        from ember_ml.nn.initializers import glorot_uniform, orthogonal
        
        # Input size for the first layer is the input size
        # For subsequent layers, it's the hidden size (or 2x hidden size if bidirectional)
        layer_input_size = self.input_size
        
        # Initialize parameters for each layer
        self.kernel_z = []
        self.kernel_r = []
        self.kernel_h = []
        self.recurrent_kernel_z = []
        self.recurrent_kernel_r = []
        self.recurrent_kernel_h = []
        self.bias_z = []
        self.bias_r = []
        self.bias_h = []
        self.recurrent_bias_z = []
        self.recurrent_bias_r = []
        self.recurrent_bias_h = []
        
        # For bidirectional GRU
        if self.bidirectional:
            self.backward_kernel_z = []
            self.backward_kernel_r = []
            self.backward_kernel_h = []
            self.backward_recurrent_kernel_z = []
            self.backward_recurrent_kernel_r = []
            self.backward_recurrent_kernel_h = []
            self.backward_bias_z = []
            self.backward_bias_r = []
            self.backward_bias_h = []
            self.backward_recurrent_bias_z = []
            self.backward_recurrent_bias_r = []
            self.backward_recurrent_bias_h = []
        
        for layer in range(self.num_layers):
            # Forward direction parameters
            self.kernel_z.append(glorot_uniform((layer_input_size, self.hidden_size)))
            self.kernel_r.append(glorot_uniform((layer_input_size, self.hidden_size)))
            self.kernel_h.append(glorot_uniform((layer_input_size, self.hidden_size)))
            
            self.recurrent_kernel_z.append(orthogonal((self.hidden_size, self.hidden_size)))
            self.recurrent_kernel_r.append(orthogonal((self.hidden_size, self.hidden_size)))
            self.recurrent_kernel_h.append(orthogonal((self.hidden_size, self.hidden_size)))
            
            if self.use_bias:
                self.bias_z.append(tensor.zeros((self.hidden_size,)))
                self.bias_r.append(tensor.zeros((self.hidden_size,)))
                self.bias_h.append(tensor.zeros((self.hidden_size,)))
                
                self.recurrent_bias_z.append(tensor.zeros((self.hidden_size,)))
                self.recurrent_bias_r.append(tensor.zeros((self.hidden_size,)))
                self.recurrent_bias_h.append(tensor.zeros((self.hidden_size,)))
            
            # Backward direction parameters (if bidirectional)
            if self.bidirectional:
                self.backward_kernel_z.append(glorot_uniform((layer_input_size, self.hidden_size)))
                self.backward_kernel_r.append(glorot_uniform((layer_input_size, self.hidden_size)))
                self.backward_kernel_h.append(glorot_uniform((layer_input_size, self.hidden_size)))
                
                self.backward_recurrent_kernel_z.append(orthogonal((self.hidden_size, self.hidden_size)))
                self.backward_recurrent_kernel_r.append(orthogonal((self.hidden_size, self.hidden_size)))
                self.backward_recurrent_kernel_h.append(orthogonal((self.hidden_size, self.hidden_size)))
                
                if self.use_bias:
                    self.backward_bias_z.append(tensor.zeros((self.hidden_size,)))
                    self.backward_bias_r.append(tensor.zeros((self.hidden_size,)))
                    self.backward_bias_h.append(tensor.zeros((self.hidden_size,)))
                    
                    self.backward_recurrent_bias_z.append(tensor.zeros((self.hidden_size,)))
                    self.backward_recurrent_bias_r.append(tensor.zeros((self.hidden_size,)))
                    self.backward_recurrent_bias_h.append(tensor.zeros((self.hidden_size,)))
            
            # Update input size for the next layer
            if self.bidirectional:
                layer_input_size = ops.multiply(self.hidden_size, 2)
            else:
                layer_input_size = self.hidden_size
    
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
            # Create GRU cells for this layer
            # Since we removed GRUCell, we'll implement the GRU logic directly here
            # This is a temporary solution until we refactor the GRU layer completely
            
            # Get initial states for this layer
            if self.bidirectional:
                layer_idx = ops.multiply(layer, 2)
            else:
                layer_idx = layer
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
                
                # GRU forward pass implementation
                # Input projection
                x_z = ops.matmul(current_input, self.kernel_z[layer])
                x_r = ops.matmul(current_input, self.kernel_r[layer])
                x_h = ops.matmul(current_input, self.kernel_h[layer])
                
                # Recurrent projection
                h_z = ops.matmul(forward_h, self.recurrent_kernel_z[layer])
                h_r = ops.matmul(forward_h, self.recurrent_kernel_r[layer])
                h_h = ops.matmul(forward_h, self.recurrent_kernel_h[layer])
                
                # Add bias if needed
                if self.use_bias:
                    x_z = ops.add(x_z, self.bias_z[layer])
                    x_r = ops.add(x_r, self.bias_r[layer])
                    x_h = ops.add(x_h, self.bias_h[layer])
                    h_z = ops.add(h_z, self.recurrent_bias_z[layer])
                    h_r = ops.add(h_r, self.recurrent_bias_r[layer])
                    h_h = ops.add(h_h, self.recurrent_bias_h[layer])
                
                # Compute gates using recurrent_activation
                from ember_ml.nn.modules.activations import get_activation
                rec_activation_fn = get_activation(self.recurrent_activation)
                z = rec_activation_fn(ops.add(x_z, h_z))  # Update gate
                r = rec_activation_fn(ops.add(x_r, h_r))  # Reset gate
                
                # Compute candidate hidden state using activation
                activation_fn = get_activation(self.activation)
                h_tilde = activation_fn(ops.add(x_h, ops.multiply(r, h_h)))
                
                # Compute new hidden state
                one_minus_z = ops.subtract(tensor.ones_like(z), z)
                scaled_h_tilde = ops.multiply(one_minus_z, h_tilde)
                scaled_prev_h = ops.multiply(z, forward_h)
                forward_h = ops.add(scaled_prev_h, scaled_h_tilde)
                forward_outputs.append(forward_h)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                # Create a list of indices in reverse order
                indices = []
                for i in range(seq_length):
                    idx = ops.subtract(ops.subtract(seq_length, 1), i)
                    indices.append(idx)
                
                # Process each index
                for t in indices:
                    # Get input for current time step
                    if self.batch_first:
                        current_input = layer_outputs[:, t]
                    else:
                        current_input = layer_outputs[t]
                    
                    # GRU backward pass implementation
                    # Input projection
                    x_z = ops.matmul(current_input, self.backward_kernel_z[layer])
                    x_r = ops.matmul(current_input, self.backward_kernel_r[layer])
                    x_h = ops.matmul(current_input, self.backward_kernel_h[layer])
                    
                    # Recurrent projection
                    h_z = ops.matmul(backward_h, self.backward_recurrent_kernel_z[layer])
                    h_r = ops.matmul(backward_h, self.backward_recurrent_kernel_r[layer])
                    h_h = ops.matmul(backward_h, self.backward_recurrent_kernel_h[layer])
                    
                    # Add bias if needed
                    if self.use_bias:
                        x_z = ops.add(x_z, self.backward_bias_z[layer])
                        x_r = ops.add(x_r, self.backward_bias_r[layer])
                        x_h = ops.add(x_h, self.backward_bias_h[layer])
                        h_z = ops.add(h_z, self.backward_recurrent_bias_z[layer])
                        h_r = ops.add(h_r, self.backward_recurrent_bias_r[layer])
                        h_h = ops.add(h_h, self.backward_recurrent_bias_h[layer])
                    
                    # Compute gates using recurrent_activation
                    from ember_ml.nn.modules.activations import get_activation
                    rec_activation_fn = get_activation(self.recurrent_activation)
                    z = rec_activation_fn(ops.add(x_z, h_z))  # Update gate
                    r = rec_activation_fn(ops.add(x_r, h_r))  # Reset gate
                    
                    # Compute candidate hidden state using activation
                    activation_fn = get_activation(self.activation)
                    h_tilde = activation_fn(ops.add(x_h, ops.multiply(r, h_h)))
                    
                    # Compute new hidden state
                    one_minus_z = ops.subtract(tensor.ones_like(z), z)
                    scaled_h_tilde = ops.multiply(one_minus_z, h_tilde)
                    scaled_prev_h = ops.multiply(z, backward_h)
                    backward_h = ops.add(scaled_prev_h, scaled_h_tilde)
                    backward_outputs.insert(0, backward_h)
            
            # Combine outputs
            if self.bidirectional:
                combined_outputs = []
                for t in range(seq_length):
                    combined = ops.concat([forward_outputs[t], backward_outputs[t]], axis=-1)
                    combined_outputs.append(combined)
            else:
                combined_outputs = forward_outputs
            
            # Stack outputs for this layer
            if self.batch_first:
                layer_outputs = tensor.stack(combined_outputs, axis=1)
            else:
                layer_outputs = tensor.stack(combined_outputs, axis=0)
            
            # Apply dropout (except for the last layer)
            is_last_layer = ops.equal(layer, ops.subtract(self.num_layers, 1))
            if not is_last_layer and self.dropout > 0:
                layer_outputs = ops.dropout(layer_outputs, self.dropout)
            
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
        """Returns the configuration of the GRU layer."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bias": self.bias,
            "batch_first": self.batch_first,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer
        })
        # Cell is reconstructed in __init__ based on these args
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Creates a GRU layer from its configuration."""
        # Create instance directly from config
        return cls(**config)