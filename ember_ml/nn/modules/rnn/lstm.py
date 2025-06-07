"""
Long Short-Term Memory (LSTM) Layer

This module provides an implementation of the LSTM layer,
which wraps an LSTMCell to create a recurrent layer.
"""

from typing import Dict, Any

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules.activations import get_activation

class LSTM(Module):
    """
    Long Short-Term Memory (LSTM) RNN layer.
    
    This layer wraps an LSTMCell to create a recurrent layer.
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
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        **kwargs
    ):
        """
        Initialize the LSTM layer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            bias: Whether to use bias
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            dropout: Dropout probability (applied between layers)
            bidirectional: Whether to use a bidirectional LSTM
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
        
        # LSTMCell parameters
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        
        # Create lists for storing parameters
        self.forward_cells: list = []
        self.backward_cells: list = []
        
        # Input size for the first layer is the input size
        # For subsequent layers, it's the hidden size (or 2x hidden size if bidirectional)
        layer_input_size = input_size
        
        for layer in range(num_layers):
            # Initialize parameters for this layer
            self._initialize_layer_parameters(layer, layer_input_size)
    
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
        kernel_shape = (input_size, ops.multiply(self.hidden_size, 4))
        recurrent_shape = (self.hidden_size, ops.multiply(self.hidden_size, 4))
        
        # Initialize forward weights
        self.input_kernels.append(Parameter(glorot_uniform(kernel_shape)))
        self.recurrent_kernels.append(Parameter(orthogonal(recurrent_shape)))
        
        if self.use_bias:
            bias_shape = (ops.multiply(self.hidden_size, 4),)
            bias_data = tensor.zeros(bias_shape)
            
            # Initialize forget gate bias to 1.0 for better gradient flow
            # Use a simpler approach that works with all backends
            bias_data_list = bias_data.tolist() if hasattr(bias_data, 'tolist') else bias_data
            
            # Set the forget gate bias (second quarter of the bias) to 1.0
            hidden_size_int = int(self.hidden_size) if not isinstance(self.hidden_size, int) else self.hidden_size
            for i in range(hidden_size_int, hidden_size_int * 2):
                bias_data_list[i] = 1.0
                
            bias_data = tensor.convert_to_tensor(bias_data_list)
            forget_gate_bias = tensor.ones((self.hidden_size,))
            
            
            self.biases.append(Parameter(bias_data))
        
        # Initialize backward weights if bidirectional
        if self.bidirectional:
            self.backward_input_kernels.append(Parameter(glorot_uniform(kernel_shape)))
            self.backward_recurrent_kernels.append(Parameter(orthogonal(recurrent_shape)))
            
            if self.use_bias:
                backward_bias_data = tensor.zeros(bias_shape)
                
                # Initialize forget gate bias to 1.0 for better gradient flow
                # Use a simpler approach that works with all backends
                backward_bias_data_list = backward_bias_data.tolist() if hasattr(backward_bias_data, 'tolist') else backward_bias_data
                
                # Set the forget gate bias (second quarter of the bias) to 1.0
                for i in range(hidden_size_int, hidden_size_int * 2):
                    backward_bias_data_list[i] = 1.0
                    
                backward_bias_data = tensor.convert_to_tensor(backward_bias_data_list)
                
                self.backward_biases.append(Parameter(backward_bias_data))
            
    
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
            c_states = []
            
            for layer in range(self.num_layers):
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
                c_states.append(tensor.zeros((batch_size, self.hidden_size)))
                
                if self.bidirectional:
                    h_states.append(tensor.zeros((batch_size, self.hidden_size)))
                    c_states.append(tensor.zeros((batch_size, self.hidden_size)))
        else:
            # Unpack provided initial states
            h_states, c_states = initial_state
        
        # Process sequence through each layer
        layer_outputs = inputs
        final_h_states = []
        final_c_states = []
        
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
            if self.bidirectional:
                layer_idx = ops.multiply(layer, 2)
            else:
                layer_idx = layer
            forward_h = h_states[layer_idx]
            forward_c = c_states[layer_idx]
            
            if self.bidirectional:
                backward_h = h_states[layer_idx + 1]
                backward_c = c_states[layer_idx + 1]
            
            # Process sequence for this layer
            forward_outputs = []
            backward_outputs = []
            
            # Forward direction
            for t in list(range(seq_length)):
                # Get input for current time step
                if self.batch_first:
                    current_input = layer_outputs[:, t]
                else:
                    current_input = layer_outputs[t]
                
                # LSTM forward pass implementation
                # Project input
                z = ops.matmul(current_input, input_kernel)
                z = ops.add(z, ops.matmul(forward_h, recurrent_kernel))
                if self.use_bias:
                    z = ops.add(z, bias)
                
                # Split into gates
                z_chunks = tensor.split_tensor(z, 4, axis=-1)
                z_i, z_f, z_o, z_c = z_chunks
                
                # Apply activations
                activation_fn = get_activation("tanh")
                rec_activation_fn = get_activation("sigmoid")
                
                i = rec_activation_fn(z_i)  # Input gate
                f = rec_activation_fn(z_f)  # Forget gate
                o = rec_activation_fn(z_o)  # Output gate
                c = activation_fn(z_c)      # Cell input
                
                # Update cell state
                forward_c = ops.add(ops.multiply(f, forward_c), ops.multiply(i, c))
                
                # Update hidden state
                forward_h = ops.multiply(o, activation_fn(forward_c))
                forward_outputs.append(forward_h)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                # Process in reverse order
                reversed_indices = []
                for i in range(seq_length):
                    idx = ops.subtract(ops.subtract(seq_length, 1), i)
                    reversed_indices.append(idx)
                
                # Process each index
                for t in reversed_indices:
                    # Get input for current time step
                    if self.batch_first:
                        current_input = layer_outputs[:, t]
                    else:
                        current_input = layer_outputs[t]
                    
                    # LSTM backward pass implementation
                    # Project input
                    z = ops.matmul(current_input, backward_input_kernel)
                    z = ops.add(z, ops.matmul(backward_h, backward_recurrent_kernel))
                    if self.use_bias:
                        z = ops.add(z, backward_bias)
                    
                    # Split into gates
                    z_chunks = tensor.split_tensor(z, 4, axis=-1)
                    z_i, z_f, z_o, z_c = z_chunks
                    
                    # Apply activations
                    i = rec_activation_fn(z_i)  # Input gate
                    f = rec_activation_fn(z_f)  # Forget gate
                    o = rec_activation_fn(z_o)  # Output gate
                    c = activation_fn(z_c)      # Cell input
                    
                    # Update cell state
                    backward_c = ops.add(ops.multiply(f, backward_c), ops.multiply(i, c))
                    
                    # Update hidden state
                    backward_h = ops.multiply(o, activation_fn(backward_c))
                    backward_outputs.insert(0, backward_h)
            
            # Combine outputs
            if self.bidirectional:
                combined_outputs = []
                for t in list(range(seq_length)):
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
            is_last_layer = ops.equal(layer, ops.subtract(self.num_layers, 1))
            if not is_last_layer and self.dropout > 0:
                layer_outputs = ops.dropout(layer_outputs, self.dropout)
            
            # Store final states for this layer
            final_h_states.append(forward_h)
            final_c_states.append(forward_c)
            
            if self.bidirectional:
                final_h_states.append(backward_h)
                final_c_states.append(backward_c)
        
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
        final_state = (tensor.stack(final_h_states), tensor.stack(final_c_states))
        
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
        c_states = []
        
        for layer in range(self.num_layers):
            h_states.append(tensor.zeros((batch_size, self.hidden_size)))
            c_states.append(tensor.zeros((batch_size, self.hidden_size)))
            
            if self.bidirectional:
                h_states.append(tensor.zeros((batch_size, self.hidden_size)))
                c_states.append(tensor.zeros((batch_size, self.hidden_size)))
        
        return (tensor.stack(h_states), tensor.stack(c_states))

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the LSTM layer."""
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
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer
        })
        # Note: We don't save the cell configs directly, as they are reconstructed
        # based on the layer's parameters in __init__.
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Creates an LSTM layer from its configuration."""
        # Create instance directly from config
        return cls(**config)