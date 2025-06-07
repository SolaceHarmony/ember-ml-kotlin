"""
Spatially Embedded Closed-form Continuous-time (seCfC) Neural Network.

This module provides an implementation of the seCfC neural network,
which integrates spatial embedding with continuous-time dynamics.
"""

# (Removed unused typing imports)
# (Removed unused numpy import)

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.wiring import EnhancedNeuronMap
from ember_ml.nn.modules.activations import get_activation
from ember_ml.nn.initializers import glorot_uniform, orthogonal

class seCfC(Module):
    """
    Spatially Embedded Closed-form Continuous-time (seCfC) neural network.
    
    This class implements a spatially embedded CfC neural network
    that integrates spatial constraints with continuous-time dynamics.
    """
    
    def __init__(
        self,
        neuron_map: EnhancedNeuronMap,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        mode: str = "default",
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        mixed_memory: bool = False,
        regularization_strength: float = 0.01,
        **kwargs
    ):
        """
        Initialize the seCfC neural network.
        
        Args:
            neuron_map: Enhanced neuron map defining connectivity and dynamics
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            go_backwards: Whether to process the sequence backwards
            regularization_strength: Strength of spatial regularization
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, EnhancedNeuronMap):
            raise TypeError("neuron_map must be an EnhancedNeuronMap instance")
        
        # Store the neuron map
        self.neuron_map = neuron_map
        
        # Store layer-specific parameters
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.regularization_strength = regularization_strength

        # CfC layer parameters
        self.mode = mode
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.mixed_memory = mixed_memory
        
        # Initialize parameters
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False
        
        # Get dynamic properties from neuron map
        dynamic_props = neuron_map.get_dynamic_properties()
        self.neuron_type = dynamic_props["neuron_type"]
        self.neuron_params = dynamic_props["neuron_params"]
        
        # Get spatial properties from neuron map
        spatial_props = neuron_map.get_spatial_properties()
        self.distance_matrix = spatial_props["distance_matrix"]
        self.communicability_matrix = spatial_props["communicability_matrix"]
    
    def build(self, input_shape):
        """Build the seCfC layer."""
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize parameters
        self.kernel = Parameter(tensor.zeros((input_dim, units * 4)))
        self.recurrent_kernel = Parameter(tensor.zeros((units, units * 4)))
        
        # Initialize bias if needed
        if self.use_bias:
            self.bias = Parameter(tensor.zeros((units * 4,)))
        
        # Initialize weights
        if self.kernel_initializer == "glorot_uniform":
            self.kernel.data = glorot_uniform((input_dim, units * 4))

        if self.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = orthogonal((units, units * 4))

        # Initialize learnable time_scale parameter
        ones = tensor.ones((units,))  # shape: (units,)
        scale_val = tensor.convert_to_tensor(self.time_scale_factor, dtype=ones.dtype)
        self.time_scale = Parameter(ops.multiply(ones, scale_val))

        # Convert spatial matrices to tensors
        self.distance_tensor = tensor.convert_to_tensor(self.distance_matrix)
        self.communicability_tensor = tensor.convert_to_tensor(self.communicability_matrix)

        # Mark as built
        self.built = True
    
    def forward(self, inputs, initial_state=None, time_deltas=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor (batch, time, features)
            initial_state: Initial state(s) for the cell
            time_deltas: Time deltas between inputs (optional)
            
        Returns:
            Layer output(s)
        """
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get input shape
        input_shape = tensor.shape(inputs)
        if len(input_shape) != 3:
            raise ValueError("Input tensor must be 3D (batch, time, features)")
        batch_size, time_steps, _ = input_shape
        
        # Create initial state if not provided
        if initial_state is None:
            h0 = tensor.zeros((batch_size, self.neuron_map.units))
            t0 = tensor.zeros((batch_size, self.neuron_map.units))
            initial_state = [h0, t0]
        
        # Process sequence
        outputs = []
        states = initial_state
        
        # Get parameters from neuron map
        time_scale_factor = self.neuron_params.get("time_scale_factor", 1.0)
        activation = self.neuron_params.get("activation", "tanh")
        recurrent_activation = self.neuron_params.get("recurrent_activation", "sigmoid")
        
        # Get activation functions
        activation_fn = get_activation(activation)
        rec_activation_fn = get_activation(recurrent_activation)
        
        # Get masks from neuron map
        input_mask = self.neuron_map.get_input_mask()
        recurrent_mask = self.neuron_map.get_recurrent_mask()
        output_mask = self.neuron_map.get_output_mask()
        
        # Convert masks to tensors
        # Convert masks to tensors with float32 dtype to ensure compatibility with matmul
        input_mask = tensor.convert_to_tensor(input_mask, dtype=tensor.float32)
        recurrent_mask = tensor.convert_to_tensor(recurrent_mask, dtype=tensor.float32)
        output_mask = tensor.convert_to_tensor(output_mask, dtype=tensor.float32)
        
        # Process sequence in reverse if go_backwards is True
        time_indices = range(time_steps - 1, -1, -1) if self.go_backwards else range(time_steps)
        
        # Process each time step
        for t in time_indices:
            # Get current input
            x_t = inputs[:, t]
            
            # Get time delta for this step if provided
            ts = 1.0
            if time_deltas is not None:
                ts = time_deltas[:, t]
            
            # Apply input mask
            masked_inputs = ops.multiply(x_t, input_mask)
            
            # Project input
            z = ops.matmul(masked_inputs, self.kernel)
            
            # Apply recurrent mask
            masked_state = ops.matmul(states[0], recurrent_mask)
            
            # Add recurrent contribution
            z = ops.add(z, ops.matmul(masked_state, self.recurrent_kernel))
            
            # Add bias if needed
            if hasattr(self, 'bias') and self.bias is not None:
                z = ops.add(z, self.bias)
            
            # Split into gates
            z_chunks = tensor.split_tensor(z, 4, axis=-1)
            z_i, z_f, z_o, z_c = z_chunks
            
            # Apply activations
            i = rec_activation_fn(z_i)  # Input gate
            f = rec_activation_fn(z_f)  # Forget gate
            o = rec_activation_fn(z_o)  # Output gate
            c = activation_fn(z_c)      # Cell input
            
            # Apply time scaling
            decay = ops.exp(ops.divide(
                ops.subtract(tensor.zeros_like(ts), ts),
                time_scale_factor
            ))
            
            # Update state
            t_next = ops.add(ops.multiply(f, states[1]), ops.multiply(i, c))
            h_next = ops.multiply(o, activation_fn(ops.add(
                ops.multiply(decay, states[0]),
                ops.multiply(ops.subtract(tensor.ones_like(decay), decay), t_next)
            )))
            
            # Apply output mask
            masked_output = ops.multiply(h_next, output_mask)
            
            # Get only the output neurons (the last output_dim neurons)
            output_dim = self.neuron_map.output_dim
            output_neurons = masked_output[:, -output_dim:]
            
            # Store output and update state
            outputs.append(output_neurons)
            states = [h_next, t_next]
        
        # If processing backwards, reverse the outputs sequence
        if self.go_backwards:
            outputs.reverse()
        
        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1)
        else:
            outputs_tensor = outputs[-1]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs_tensor, states
        else:
            return outputs_tensor
    
    def reset_state(self, batch_size=1):
        """
        Reset the layer state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        h0 = tensor.zeros((batch_size, self.neuron_map.units))
        t0 = tensor.zeros((batch_size, self.neuron_map.units))
        return [h0, t0]
    
    def get_regularization_loss(self):
        """
        Get the regularization loss for the layer.
        
        Returns:
            Regularization loss
        """
        if not self.built:
            return 0.0
        
        # Get the weight matrix
        weight_matrix = self.recurrent_kernel.data
        
        # Apply spatial regularization
        # L1 regularization weighted by distance and communicability
        abs_weights = ops.abs(weight_matrix)
        
        # Get the shape of the weight matrix
        weight_shape = tensor.shape(abs_weights)
        units = weight_shape[0]
        units_times_4 = weight_shape[1]
        
        # Since we can't reshape directly, we'll use a different approach
        # We'll slice the weight matrix into 4 parts and sum them
        slice_size = units
        weights_1 = abs_weights[:, :slice_size]
        weights_2 = abs_weights[:, slice_size:slice_size*2]
        weights_3 = abs_weights[:, slice_size*2:slice_size*3]
        weights_4 = abs_weights[:, slice_size*3:slice_size*4]
        
        # Sum the 4 parts to get a (units, units) matrix
        summed_weights = ops.add(weights_1, weights_2)
        summed_weights = ops.add(summed_weights, weights_3)
        summed_weights = ops.add(summed_weights, weights_4)
        
        # Apply distance and communicability weighting
        weighted_weights = ops.multiply(summed_weights, self.distance_tensor)
        weighted_weights = ops.multiply(weighted_weights, self.communicability_tensor)
        
        # Sum the weighted weights
        reg_loss = stats.sum(weighted_weights)
        
        # Scale by regularization strength
        reg_loss = ops.multiply(reg_loss, self.regularization_strength)
        
        return reg_loss
    
    def get_config(self):
        """
        Get the configuration of the seCfC layer.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards,
            "regularization_strength": self.regularization_strength
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Create a seCfC layer from a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            seCfC layer instance
        """
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import EnhancedNCPMap
        neuron_map = EnhancedNCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)