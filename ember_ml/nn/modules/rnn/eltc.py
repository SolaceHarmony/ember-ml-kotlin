"""
Enhanced Liquid Time-Constant (ELTC) Neural Network

This module provides an implementation of ELTC layers,
which extend the LTC with configurable ODE solvers and enhanced dynamics.
This implementation directly uses NeuronMap for both structure and dynamics.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.initializers import glorot_uniform, orthogonal
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation

class ODESolver(Enum):
    """ODE solver types for continuous-time neural networks.
    
    Available solvers:
    - SEMI_IMPLICIT: Semi-implicit Euler method, good balance of stability and speed
    - EXPLICIT: Explicit Euler method, fastest but less stable
    - RUNGE_KUTTA: 4th order Runge-Kutta method, most accurate but computationally intensive
    """
    SEMI_IMPLICIT = "semi_implicit"
    EXPLICIT = "explicit"
    RUNGE_KUTTA = "rk4"

class ELTC(Module):
    """
    Enhanced Liquid Time-Constant (ELTC) RNN layer.
    
    This layer extends the LTC with configurable ODE solvers and enhanced dynamics.
    It directly uses NeuronMap for both structure and dynamics.
    
    The ELTC implements the following ODE:
        dy/dt = σ(Wx + Uh + b) - y
    where:
        - y is the cell state
        - x is the input
        - W is the input weight matrix
        - U is the recurrent weight matrix
        - b is the bias vector
        - σ is the activation function
    
    The ODE is solved using one of three methods:
    1. Semi-implicit Euler: Provides good stability and accuracy balance
    2. Explicit Euler: Fastest but may be unstable for stiff equations
    3. Runge-Kutta (RK4): Most accurate but computationally expensive
    """
    
    def __init__(
        self,
        neuron_map: NCPMap,
        solver: Union[str, ODESolver] = ODESolver.RUNGE_KUTTA,
        ode_unfolds: int = 6,
        sparsity: float = 0.5,
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
        mixed_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the ELTC layer.
        
        Args:
            neuron_map: NCPMap instance defining both structure and dynamics
            solver: ODE solver type (ODESolver enum or string)
                - "semi_implicit": Semi-implicit Euler method
                - "explicit": Explicit Euler method
                - "rk4": 4th order Runge-Kutta method
            ode_unfolds: Number of ODE solver steps per time step
            sparsity: Sparsity level for adjacency matrices (0.0 to 1.0)
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
        self.solver = solver if isinstance(solver, ODESolver) else ODESolver(solver)
        self.ode_unfolds = ode_unfolds
        self.sparsity = sparsity
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.batch_first = batch_first
        self.mixed_memory = mixed_memory
        
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
        """Build the ELTC layer."""
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
        
        # Apply sparsity to recurrent connections
        # Create a binary mask using Bernoulli distribution
        sparsity_mask = tensor.random_bernoulli(
            shape=(units, units),
            p=1.0 - self.sparsity
        )
        self.w.data = ops.multiply(self.w.data, sparsity_mask)
        
        # Initialize weights for input connections
        if self.neuron_map.input_mapping in ["affine", "linear"]:
            self.sensory_sigma = Parameter(tensor.ones((self.input_size,)))
            self.sensory_mu = Parameter(tensor.zeros((self.input_size,)))
            self.sensory_w = Parameter(glorot_uniform((self.input_size, units)))
            self.sensory_erev = Parameter(tensor.zeros((self.input_size, units)))
            
            # Initialize input projection
            self.input_kernel = Parameter(glorot_uniform((self.input_size, units)))
            if self.neuron_map.input_mapping == "affine":
                self.input_bias = Parameter(tensor.zeros((units,)))
        
        # Initialize output projection
        if self.neuron_map.output_mapping in ["affine", "linear"]:
            output_dim = self.neuron_map.output_dim
            self.output_kernel = Parameter(glorot_uniform((units, output_dim)))
            if self.neuron_map.output_mapping == "affine":
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
                i = ops.sigmoid(
                    ops.matmul(inputs, self.input_kernel) +
                    ops.matmul(h_prev, self.input_recurrent_kernel) +
                    self.input_bias
                )
                
                # Forget gate
                f = ops.sigmoid(
                    ops.matmul(inputs, self.forget_kernel) +
                    ops.matmul(h_prev, self.forget_recurrent_kernel) +
                    self.forget_bias
                )
                
                # Cell gate
                g = ops.tanh(
                    ops.matmul(inputs, self.cell_kernel) +
                    ops.matmul(h_prev, self.cell_recurrent_kernel) +
                    self.cell_bias
                )
                
                # Output gate
                o = ops.sigmoid(
                    ops.matmul(inputs, self.output_kernel) +
                    ops.matmul(h_prev, self.output_recurrent_kernel) +
                    self.output_bias
                )
                
                # Update cell state
                c = f * c_prev + i * g
                
                # Update hidden state
                h = o * ops.tanh(c)
                
                return h, (h, c)
        
        return MemoryCell(input_size, state_size)
    
    def _explicit_euler_solve(self, f, y, dt):
        """Explicit Euler method for ODE solving."""
        return y + dt * f(None, y)
    
    def _semi_implicit_solve(self, f, y, dt):
        """Semi-implicit Euler method for ODE solving."""
        # First get the derivative at the current point
        k1 = f(None, y)
        
        # Then use this to estimate the next point
        y_pred = y + dt * k1
        
        # Get the derivative at the predicted point
        k2 = f(None, y_pred)
        
        # Average the derivatives and take a step
        return y + dt * (k1 + k2) / 2
    
    def _rk4_solve(self, f, y, t, dt):
        """4th order Runge-Kutta method for ODE solving."""
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt*k1/2)
        k3 = f(t + dt/2, y + dt*k2/2)
        k4 = f(t + dt, y + dt*k3)
        
        return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODE for the ELTC dynamics using the selected solver."""
        # Get parameters from neuron_map
        epsilon = self.neuron_map.epsilon
        implicit_param_constraints = self.neuron_map.implicit_param_constraints
        
        # Apply constraints to parameters if needed
        if implicit_param_constraints:
            gleak = ops.abs(self.gleak) + epsilon
            cm = ops.abs(self.cm) + epsilon
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
        
        # Define ODE function
        def f(t, y):
            # Calculate activation
            activation = get_activation(self.neuron_map.activation)(y)
            
            # Calculate synaptic current
            syn_current = ops.matmul(activation, masked_w)
            
            # Reshape erev for broadcasting with y
            erev_expanded = tensor.reshape(self.erev, (1, units, units))
            y_expanded = tensor.reshape(y, (batch_size, units, 1))
            
            # Calculate (erev - y) with proper broadcasting
            erev_minus_v = ops.subtract(erev_expanded, y_expanded)
            
            # Reshape syn_current for element-wise multiplication
            syn_current = tensor.reshape(syn_current, (batch_size, 1, units))
            
            # Perform element-wise multiplication and sum along the last dimension
            syn_current = stats.sum(ops.multiply(syn_current, erev_minus_v), axis=2)
            
            # Calculate sensory current
            sensory_current = 0
            if self.neuron_map.input_mapping in ["affine", "linear"]:
                # Apply activation to inputs
                sensory_activation = get_activation(self.neuron_map.activation)(inputs)
                
                # Ensure inputs and weights have compatible shapes for matrix multiplication
                sensory_w_transposed = tensor.transpose(self.sensory_w)
                sensory_current = ops.matmul(sensory_activation, sensory_w_transposed)
                
                # Reshape sensory_erev for broadcasting with y
                sensory_erev_expanded = tensor.reshape(self.sensory_erev, (1, self.input_size, self.neuron_map.units))
                
                # Reshape y for broadcasting with sensory_erev
                y_expanded = tensor.reshape(y, (batch_size, 1, self.neuron_map.units))
                
                # Calculate (sensory_erev - y) with proper broadcasting
                sensory_erev_minus_v = ops.subtract(sensory_erev_expanded, y_expanded)
                
                # Reshape sensory_current for element-wise multiplication
                sensory_current = tensor.reshape(sensory_current, (batch_size, self.input_size, 1))
                
                # Perform element-wise multiplication and sum along the input_size dimension
                sensory_current = stats.sum(ops.multiply(sensory_current, sensory_erev_minus_v), axis=1)
            
            # Calculate leak current
            leak_current = ops.multiply(gleak, ops.subtract(self.vleak, y))
            
            # Calculate total current
            total_current = ops.add(leak_current, ops.add(syn_current, sensory_current))
            
            # Return derivative
            return ops.divide(total_current, cm)
        
        # Solve ODE using selected solver
        dt = elapsed_time / self.ode_unfolds
        
        for _ in range(self.ode_unfolds):
            if self.solver == ODESolver.SEMI_IMPLICIT:
                v_pre = self._semi_implicit_solve(f, v_pre, dt)
            elif self.solver == ODESolver.EXPLICIT:
                v_pre = self._explicit_euler_solve(f, v_pre, dt)
            elif self.solver == ODESolver.RUNGE_KUTTA:
                v_pre = self._rk4_solve(f, v_pre, 0, dt)
            else:
                raise ValueError(f"Unsupported solver type: {self.solver}")
        
        return v_pre
    
    def _map_inputs(self, inputs):
        """Map inputs using the specified input mapping."""
        if self.neuron_map.input_mapping == "affine":
            return ops.add(ops.matmul(inputs, self.input_kernel), self.input_bias)
        elif self.neuron_map.input_mapping == "linear":
            return ops.matmul(inputs, self.input_kernel)
        else:
            return inputs
    
    def _map_outputs(self, state):
        """Map outputs using the specified output mapping."""
        if self.neuron_map.output_mapping == "affine":
            return ops.add(ops.matmul(state, self.output_kernel), self.output_bias)
        elif self.neuron_map.output_mapping == "linear":
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
        # Use item() to get Python scalar
        return float(tensor.item(sum_val))
    
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
            
            # Apply ELTC dynamics with selected ODE solver
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
        """Returns the configuration of the ELTC layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "solver": self.solver.value,
            "ode_unfolds": self.ode_unfolds,
            "sparsity": self.sparsity,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "batch_first": self.batch_first,
            "mixed_memory": self.mixed_memory
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ELTC':
        """Creates an ELTC layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)