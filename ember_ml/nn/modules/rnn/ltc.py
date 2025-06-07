"""
Liquid Time-Constant (LTC) Layer - Pure Wired Implementation

This module provides an implementation of the LTC layer that directly uses
NeuronMap for both structure and dynamics, without relying on separate cell objects.
"""

from typing import Dict, Any

from ember_ml import ops
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation

class LTC(Module):
    """
    Liquid Time-Constant (LTC) RNN layer - Pure Wired Implementation.
    
    This layer directly uses NeuronMap for both structure and dynamics,
    without relying on separate cell objects.
    """
    
    def __init__(
        self,
        neuron_map: NCPMap,
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
        **kwargs
    ):
        """
        Initialize the LTC layer.
        
        Args:
            neuron_map: NCPMap instance defining both structure and dynamics
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            batch_first: Whether the batch or time dimension is the first (0-th) dimension
            mixed_memory: Whether to augment the RNN by a memory-cell to help learn long-term dependencies
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
        
        # Store the neuron map and layer parameters
        self.neuron_map = neuron_map
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.batch_first = batch_first
        self.mixed_memory = mixed_memory
        
        # LTCCell parameters
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.implicit_param_constraints = implicit_param_constraints
        self.make_positive_fn = get_activation("softplus") if implicit_param_constraints else lambda x: x
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        
        # Set input_size from neuron_map.input_dim if the map is already built
        # Otherwise, it will be set during the first forward pass
        self.input_size = getattr(neuron_map, 'input_dim', None)
        
        # Initialize parameters
        self.gleak = None
        self.vleak = None
        self.cm = None
        self.sigma = None
        self.mu = None
        self.w = None
        self.erev = None
        self.sensory_sigma = None
        self.sensory_mu = None
        self.sensory_w = None
        self.sensory_erev = None
        self.input_kernel = None
        self.input_bias = None
        self.output_kernel = None
        self.output_bias = None
        self.built = False
        
        # Create memory cell if using mixed memory
        # If input_size is not available yet, memory cell creation will be deferred
        self.memory_cell = None
        if self.mixed_memory and self.input_size is not None:
            self.memory_cell = self._create_memory_cell(self.input_size, self.state_size)
    
    def build(self, input_shape):
        """Build the LTC layer."""
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
        
        # Initialize parameters
        self.gleak = Parameter(tensor.ones((units,)))
        self.vleak = Parameter(tensor.zeros((units,)))
        self.cm = Parameter(tensor.ones((units,)))
        
        # Get recurrent mask from neuron map
        recurrent_mask = self.neuron_map.get_recurrent_mask()
        
        # Initialize weights for recurrent connections
        self.sigma = Parameter(tensor.ones((units,)))
        self.mu = Parameter(tensor.zeros((units,)))
        self.w = Parameter(glorot_uniform((units, units)))
        self.erev = Parameter(tensor.zeros((units, units)))
        
        # Initialize weights for input connections
        # Check if input_mapping exists, default to "affine" if not (for compatibility with maps like FullyConnectedMap)
        input_mapping = getattr(self.neuron_map, 'input_mapping', 'affine')
        if input_mapping in ["affine", "linear"]:
            self.sensory_sigma = Parameter(tensor.ones((self.input_size,)))
            self.sensory_mu = Parameter(tensor.zeros((self.input_size,)))
            self.sensory_w = Parameter(glorot_uniform((self.input_size, units)))
            self.sensory_erev = Parameter(tensor.zeros((self.input_size, units)))
            
            # Initialize input projection
            self.input_kernel = Parameter(glorot_uniform((self.input_size, units)))
            if input_mapping == "affine": # Use the retrieved or default input_mapping
                self.input_bias = Parameter(tensor.zeros((units,)))

        # Initialize output projection
        # Check if output_mapping exists, default to "affine" if not
        output_mapping = getattr(self.neuron_map, 'output_mapping', 'affine')
        if output_mapping in ["affine", "linear"]:
            output_dim = self.neuron_map.output_dim
            self.output_kernel = Parameter(glorot_uniform((units, output_dim)))
            if output_mapping == "affine": # Use the retrieved or default output_mapping
                self.output_bias = Parameter(tensor.zeros((output_dim,)))

        # Create memory cell if using mixed memory and it wasn't created during init
        if self.mixed_memory and self.memory_cell is None:
            self.memory_cell = self._create_memory_cell(self.input_size, self.state_size)
        
        # Mark as built
        self.built = True
    
    def _create_memory_cell(self, input_size, state_size):
        """Create a memory cell for mixed memory mode."""
        # Simple memory cell implementation
        class MemoryCell(Module):
            def __init__(self, input_size, state_size):
                super().__init__()
                self.input_size = input_size
                self.state_size = state_size
                
                # Input gate
                self.input_kernel = glorot_uniform((input_size, state_size))
                self.input_recurrent_kernel = orthogonal((state_size, state_size))
                self.input_bias = tensor.zeros((state_size,))
                
                # Forget gate
                self.forget_kernel = glorot_uniform((input_size, state_size))
                self.forget_recurrent_kernel = orthogonal((state_size, state_size))
                self.forget_bias = tensor.ones((state_size,))  # Initialize with 1s for better gradient flow
                
                # Cell gate
                self.cell_kernel = glorot_uniform((input_size, state_size))
                self.cell_recurrent_kernel = orthogonal((state_size, state_size))
                self.cell_bias = tensor.zeros((state_size,))
                
                # Output gate
                self.output_kernel = glorot_uniform((input_size, state_size))
                self.output_recurrent_kernel = orthogonal((state_size, state_size))
                self.output_bias = tensor.zeros((state_size,))
            
            def forward(self, inputs, states):
                h_prev, c_prev = states
                
                # Input gate
                i_term1 = ops.matmul(inputs, self.input_kernel)
                i_term2 = ops.matmul(h_prev, self.input_recurrent_kernel)
                i_sum = ops.add(ops.add(i_term1, i_term2), self.input_bias)
                i = ops.sigmoid(i_sum)
                
                # Forget gate
                f_term1 = ops.matmul(inputs, self.forget_kernel)
                f_term2 = ops.matmul(h_prev, self.forget_recurrent_kernel)
                f_sum = ops.add(ops.add(f_term1, f_term2), self.forget_bias)
                f = ops.sigmoid(f_sum)
                
                # Cell gate
                g_term1 = ops.matmul(inputs, self.cell_kernel)
                g_term2 = ops.matmul(h_prev, self.cell_recurrent_kernel)
                g_sum = ops.add(ops.add(g_term1, g_term2), self.cell_bias)
                g = ops.tanh(g_sum)
                
                # Output gate
                o_term1 = ops.matmul(inputs, self.output_kernel)
                o_term2 = ops.matmul(h_prev, self.output_recurrent_kernel)
                o_sum = ops.add(ops.add(o_term1, o_term2), self.output_bias)
                o = ops.sigmoid(o_sum)
                
                # Update cell state
                fc = ops.multiply(f, c_prev)
                ig = ops.multiply(i, g)
                c = ops.add(fc, ig)
                
                # Update hidden state
                tanh_c = ops.tanh(c)
                h = ops.multiply(o, tanh_c)
                
                return h, (h, c)
        
        return MemoryCell(input_size, state_size)
    
    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODE for the LTC dynamics."""
        # Use parameters from the layer
        ode_unfolds = self.ode_unfolds
        epsilon = self.epsilon
        implicit_param_constraints = self.implicit_param_constraints
        
        # Apply constraints to parameters if needed
        if implicit_param_constraints:
            gleak_abs = ops.abs(self.gleak)
            gleak = ops.add(gleak_abs, epsilon)
            cm_abs = ops.abs(self.cm)
            cm = ops.add(cm_abs, epsilon)
        else:
            gleak = self.gleak
            cm = self.cm
        
        # Get recurrent mask from neuron map
        recurrent_mask = self.neuron_map.get_recurrent_mask()
        
        # Apply mask to weights
        masked_w = ops.multiply(self.w, recurrent_mask)
        
        # Initialize state for ODE solver
        v_pre = state
        
        # Get batch size and units
        batch_size = tensor.shape(v_pre)[0]
        units = tensor.shape(v_pre)[1]
        
        # Solve ODE using multiple unfoldings
        for i in range(ode_unfolds):
            # Calculate activation
            # Use getattr to safely access activation, default to 'tanh' if not present on map
            activation_name = getattr(self.neuron_map, 'activation', 'tanh')
            activation = get_activation(activation_name)(v_pre)

            # Calculate synaptic current
            syn_current = ops.matmul(activation, masked_w)
            
            # Reshape erev for broadcasting with v_pre
            # erev shape: (units, units), v_pre shape: (batch_size, units)
            # We need to reshape erev to (1, units, units) for proper broadcasting
            erev_expanded = tensor.reshape(self.erev, (1, units, units))
            v_pre_expanded = tensor.reshape(v_pre, (batch_size, units, 1))
            
            # Calculate (erev - v_pre) with proper broadcasting
            # This will result in a tensor of shape (batch_size, units, units)
            erev_minus_v = ops.subtract(erev_expanded, v_pre_expanded)
            
            # Reshape syn_current for element-wise multiplication
            syn_current = tensor.reshape(syn_current, (batch_size, 1, units))
            
            # Perform element-wise multiplication and sum along the last dimension
            # This will result in a tensor of shape (batch_size, units)
            syn_current = stats.sum(ops.multiply(syn_current, erev_minus_v), axis=2)
            
            # Calculate sensory current
            sensory_current = tensor.zeros_like(syn_current)
            # Check if input_mapping exists, default to "affine" if not
            input_mapping = getattr(self.neuron_map, 'input_mapping', 'affine')
            if input_mapping in ["affine", "linear"]:
                # Apply activation to inputs
                # Use getattr for activation as well, defaulting to tanh
                activation_name = getattr(self.neuron_map, 'activation', 'tanh')
                sensory_activation = get_activation(activation_name)(inputs)

                # Ensure inputs and weights have compatible shapes for matrix multiplication
                # inputs shape: (batch_size, input_size)
                # self.sensory_w shape: (input_size, units)
                # We need to transpose self.sensory_w to make the dimensions compatible
                sensory_w_transposed = tensor.transpose(self.sensory_w)
                sensory_current = ops.matmul(sensory_activation, sensory_w_transposed)
                
                # Reshape sensory_erev for broadcasting with v_pre
                # sensory_erev shape: (input_size, units), v_pre shape: (batch_size, units)
                # We need to reshape to (1, input_size, units) for broadcasting with v_pre_expanded
                sensory_erev_expanded = tensor.reshape(self.sensory_erev, (1, self.input_size, self.neuron_map.units))
                
                # Reshape v_pre for broadcasting with sensory_erev
                # v_pre shape: (batch_size, units)
                # We need to reshape to (batch_size, 1, units) for broadcasting with sensory_erev_expanded
                v_pre_expanded = tensor.reshape(v_pre, (batch_size, 1, self.neuron_map.units))
                
                # Calculate (sensory_erev - v_pre) with proper broadcasting
                # Result shape: (batch_size, input_size, units)
                sensory_erev_minus_v = ops.subtract(sensory_erev_expanded, v_pre_expanded)
                
                # Reshape sensory_current for element-wise multiplication
                # sensory_current shape: (batch_size, units)
                # We need to reshape to (batch_size, input_size, 1) for broadcasting
                sensory_current = tensor.reshape(sensory_current, (batch_size, self.input_size, 1))
                
                # Perform element-wise multiplication and sum along the input_size dimension
                # Result shape: (batch_size, units)
                sensory_current = stats.sum(ops.multiply(sensory_current, sensory_erev_minus_v), axis=1)
            
            # Calculate leak current
            leak_current = ops.multiply(gleak, ops.subtract(self.vleak, v_pre))
            
            # Calculate total current
            total_current = ops.add(leak_current, ops.add(syn_current, sensory_current))
            
            # Update state
            delta_v = ops.divide(ops.multiply(elapsed_time, total_current), cm)
            v_pre = ops.add(v_pre, delta_v)
        
        return v_pre
    
    def _map_inputs(self, inputs):
        """Map inputs using the specified input mapping."""
        if self.input_mapping == "affine":
            return ops.add(ops.matmul(inputs, self.input_kernel), self.input_bias)
        elif self.input_mapping == "linear":
            return ops.matmul(inputs, self.input_kernel)
        else:
            return inputs
    
    def _map_outputs(self, state):
        """Map outputs using the specified output mapping."""
        if self.output_mapping == "affine":
            return ops.add(ops.matmul(state, self.output_kernel), self.output_bias)
        elif self.output_mapping == "linear":
            return ops.matmul(state, self.output_kernel)
        else:
            return state
    
    @property
    def state_size(self):
        return self.neuron_map.units
    
    @property
    def sensory_size(self):
        return self.neuron_map.input_dim
    
    @property
    def motor_size(self):
        return self.neuron_map.output_dim
    
    @property
    def output_size(self):
        return self.motor_size
    
    @property
    def synapse_count(self):
        # Use ops/tensor for calculations, avoid numpy
        # Ensure adjacency_matrix is a tensor first
        adj_matrix_tensor = tensor.convert_to_tensor(self.neuron_map.adjacency_matrix)
        return stats.sum(tensor.abs(adj_matrix_tensor))
    
    @property
    def sensory_synapse_count(self):
        # Use ops/tensor for calculations, avoid numpy
        sensory_matrix_tensor = tensor.convert_to_tensor(self.neuron_map.sensory_adjacency_matrix)
        # sum result might be a 0-dim tensor, convert to float if necessary
        sum_val = stats.sum(tensor.abs(sensory_matrix_tensor))
        # Convert to scalar without using float() cast
        return tensor.item(sum_val)
    
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
        
        # Initialize states if not provided
        if initial_state is None:
            h_state = tensor.zeros((batch_size, self.state_size))
            c_state = tensor.zeros((batch_size, self.state_size)) if self.mixed_memory else None
        else:
            if self.mixed_memory and not isinstance(initial_state, (list, tuple)):
                raise ValueError(
                    "When using mixed_memory=True, initial_state must be a tuple (h0, c0)"
                )
            h_state, c_state = initial_state if self.mixed_memory else (initial_state, None)
            
            # Handle non-batched states
            if is_batched and len(tensor.shape(h_state)) != 2:
                raise ValueError(
                    f"For batched inputs, initial_state should be 2D but got {len(tensor.shape(h_state))}D"
                )
            elif not is_batched and len(tensor.shape(h_state)) != 1:
                # Add batch dimension for non-batched states
                h_state = tensor.expand_dims(h_state, 0)
                c_state = tensor.expand_dims(c_state, 0) if c_state is not None else None
        
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
            
            # Apply memory cell if using mixed memory
            if self.mixed_memory:
                h_state, (h_state, c_state) = self.memory_cell(current_input, (h_state, c_state))
            
            # Map inputs
            mapped_input = self._map_inputs(current_input)
            
            # Apply LTC dynamics
            h_state = self._ode_solver(mapped_input, h_state, ts)
            
            # Map outputs
            output = self._map_outputs(h_state)
            
            # Store output if returning sequences
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
        
        # Prepare final state
        final_state = (h_state, c_state) if self.mixed_memory else h_state
        
        # Handle non-batched outputs
        if not is_batched:
            outputs = tensor.squeeze(outputs, batch_dim)
            if self.mixed_memory:
                final_state = (tensor.squeeze(h_state, 0), tensor.squeeze(c_state, 0))
            else:
                final_state = tensor.squeeze(h_state, 0)
        
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
        h_state = tensor.zeros((batch_size, self.state_size))
        if self.mixed_memory:
            c_state = tensor.zeros((batch_size, self.state_size))
            return (h_state, c_state)
        else:
            return h_state
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the LTC layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "batch_first": self.batch_first,
            "mixed_memory": self.mixed_memory,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "ode_unfolds": self.ode_unfolds,
            "epsilon": self.epsilon,
            "implicit_param_constraints": self.implicit_param_constraints
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LTC':
        """Creates an LTC layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)