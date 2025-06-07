"""
Stride-Aware Layer

This module provides an implementation of the StrideAware layer,
which wraps a StrideAwareCell to create a recurrent layer.
"""

# Removed unused imports

# Removed numpy import
from ember_ml import ops
from ember_ml.nn.modules import Module
# Removed stride_aware_cell import
from ember_ml.nn import tensor

class StrideAware(Module):
    """
    Stride-Aware RNN layer.
    
    This layer wraps a StrideAwareCell to create a recurrent layer
    that can handle different timescales.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        return_sequences: bool = True,
        batch_first: bool = True,
        activation: str = "tanh",
        bias: bool = True,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs
    ):
        """
        Initialize the StrideAware layer.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            stride_length: Length of the stride this layer handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            return_sequences: Whether to return the full sequence or just the last output
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            activation: Activation function to use
            bias: Whether to use bias
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.return_sequences = return_sequences
        self.batch_first = batch_first
        
        # StrideAwareCell parameters
        self.activation = activation
        self.bias = bias
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        # Initialize parameters directly instead of creating a cell
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for the StrideAware layer."""
        from ember_ml.nn.initializers import glorot_uniform
        
        # Input weights
        self.input_kernel = tensor.Parameter(glorot_uniform((self.input_size, self.hidden_size)))
        
        # Hidden weights
        self.hidden_kernel = tensor.Parameter(glorot_uniform((self.hidden_size, self.hidden_size)))
        
        # Output weights
        self.output_kernel = tensor.Parameter(glorot_uniform((self.hidden_size, self.hidden_size)))
        
        # Bias terms
        if self.use_bias:
            self.input_bias = tensor.Parameter(tensor.zeros((self.hidden_size,)))
            self.hidden_bias = tensor.Parameter(tensor.zeros((self.hidden_size,)))
            self.output_bias = tensor.Parameter(tensor.zeros((self.hidden_size,)))
        
        # Time constant
        self.tau = tensor.Parameter(tensor.ones((self.hidden_size,)))
    
    def forward(self, inputs, initial_state=None, elapsed_time=1.0):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state of the RNN
            elapsed_time: Time elapsed since last update (default: 1.0)
            
        Returns:
            Layer output and final state
        """
        # Get device and batch information
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
        
        # Initialize state if not provided
        if initial_state is None:
            state = tensor.zeros((batch_size, self.hidden_size))
        else:
            state = initial_state
            
            # Handle non-batched states
            if is_batched and len(tensor.shape(state)) != 2:
                raise ValueError(
                    f"For batched inputs, initial_state should be 2D but got {len(tensor.shape(state))}D"
                )
            elif not is_batched and len(tensor.shape(state)) != 1:
                raise ValueError(
                    f"For non-batched inputs, initial_state should be 1D but got {len(tensor.shape(state))}D"
                )
                
                # Add batch dimension for non-batched states
                state = tensor.expand_dims(state, 0)
        
        # Process sequence
        output_sequence = []
        for t in range(seq_length):
            # Get input for current time step
            if self.batch_first:
                current_input = inputs[:, t]
            else:
                current_input = inputs[t]
            
            # Apply stride-specific temporal scaling
            effective_time = ops.multiply(elapsed_time, ops.multiply(self.stride_length, self.time_scale_factor))
            
            # Compute input contribution
            input_signal = ops.matmul(current_input, self.input_kernel)
            if self.use_bias:
                input_signal = ops.add(input_signal, self.input_bias)
            
            # Compute hidden contribution
            hidden_signal = ops.matmul(state, self.hidden_kernel)
            if self.use_bias:
                hidden_signal = ops.add(hidden_signal, self.hidden_bias)
            
            # Update state using time-scaled dynamics
            scale_factor = ops.divide(1.0, ops.multiply(self.tau, effective_time))
            combined_signal = ops.add(input_signal, hidden_signal)
            diff_signal = ops.subtract(combined_signal, state)
            scaled_diff = ops.multiply(scale_factor, diff_signal)
            state = ops.add(state, scaled_diff)
            
            # Compute output
            output = ops.matmul(state, self.output_kernel)
            if self.use_bias:
                output = ops.add(output, self.output_bias)
            
            # Apply activation function
            from ember_ml.nn.modules.activations import get_activation
            activation_fn = get_activation(self.activation)
            output = activation_fn(output)
            
            # Store output
            output_sequence.append(output)
        
        # Prepare output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            outputs = tensor.stack(output_sequence, axis=stack_dim)
        else:
            # If not returning sequences, use the last output
            outputs = output_sequence[-1]
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = tensor.squeeze(outputs, batch_dim)
            state = tensor.squeeze(state, 0)
        
        return outputs, state
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return tensor.zeros((batch_size, self.hidden_size))