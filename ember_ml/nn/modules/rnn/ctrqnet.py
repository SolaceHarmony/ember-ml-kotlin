"""
Continuous-Time Recurrent Quantum Neural Network (CTRQNet)

This module provides an implementation of Continuous-Time Recurrent Quantum Neural Networks (CTRQNets),
which extend LQNets with continuous-time dynamics and enhanced quantum-inspired features.
"""

from typing import Dict, Any, Optional, List, Tuple, Union

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap
from ember_ml.nn import tensor
from ember_ml.nn.modules.activations import get_activation

class CTRQNet(Module):
    """
    Continuous-Time Recurrent Quantum Neural Network (CTRQNet).
    
    This layer implements a quantum-inspired recurrent neural network with continuous-time
    dynamics. It extends LQNets with enhanced quantum-inspired features and
    temporal processing capabilities.
    """
    
    def __init__(
        self,
        neuron_map: NeuronMap,
        nu_0: float = 1.0,
        beta: float = 0.1,
        noise_scale: float = 0.1,
        time_scale_factor: float = 1.0,
        use_harmonic_embedding: bool = True,
        return_sequences: bool = True,
        return_state: bool = False,
        batch_first: bool = True,
        **kwargs
    ):
        """
        Initialize the CTRQNet layer.
        
        Args:
            neuron_map: NeuronMap instance defining connectivity
            nu_0: Base viscosity parameter
            beta: Energy scaling parameter
            noise_scale: Scale of the stochastic noise
            time_scale_factor: Factor to scale the time constant
            use_harmonic_embedding: Whether to use harmonic embedding
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
        self.nu_0 = nu_0
        self.beta = beta
        self.noise_scale = noise_scale
        self.time_scale_factor = time_scale_factor
        self.use_harmonic_embedding = use_harmonic_embedding
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.batch_first = batch_first
        
        # Initialize parameters
        self.W_symplectic = None
        self.A_matrix = None
        self.B_matrix = None
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.time_scale = None
        self.amplitudes = None
        self.frequencies = None
        self.phases = None
        
        # Mark as not built yet
        self.built = False
    
    def build(self, input_shape):
        """Build the CTRQNet layer."""
        # Get input dimension
        if len(input_shape) == 3:  # (batch, time, features) or (time, batch, features)
            feature_dim = 2 if self.batch_first else 1
            input_dim = input_shape[feature_dim]
        else:
            input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize skew-symmetric matrix for b-symplectic structure
        self.W_symplectic = Parameter(self._skew_symmetric_weights(units))
        
        # Initialize matrices for stochastic-quantum mapping
        # A_matrix maps from hidden state to drift term
        self.A_matrix = Parameter(tensor.random_normal((units, units)))
        # B_matrix maps from noise to diffusion term
        self.B_matrix = Parameter(tensor.random_normal((units, units)))
        
        # Initialize input weights
        self.kernel = Parameter(tensor.zeros((input_dim, units)))
        
        # Initialize recurrent weights using neuron map's adjacency matrix
        recurrent_mask = self.neuron_map.adjacency_matrix
        self.recurrent_kernel = Parameter(ops.multiply(
            tensor.random_normal((units, units)),
            recurrent_mask
        ))
        
        # Initialize bias
        self.bias = Parameter(tensor.zeros((units,)))
        
        # Initialize time scale parameter
        self.time_scale = Parameter(tensor.ones((units,)) * self.time_scale_factor)
        
        # Initialize harmonic embedding parameters if needed
        if self.use_harmonic_embedding:
            self.amplitudes = Parameter(tensor.random_uniform((input_dim, units), minval=0.5, maxval=1.5))
            self.frequencies = Parameter(tensor.random_uniform((input_dim, units), minval=0.1, maxval=2.0))
            self.phases = Parameter(tensor.random_uniform((input_dim, units), minval=0.0, maxval=2*3.14159))
        
        # Mark as built
        self.built = True
    
    def _skew_symmetric_weights(self, dim):
        """
        Create a random skew-symmetric matrix.
        
        Args:
            dim: Dimension of the matrix
            
        Returns:
            Skew-symmetric matrix
        """
        # Create a random matrix
        W = tensor.random_normal((dim, dim))
        
        # Make it skew-symmetric: W = 0.5 * (W - W^T)
        W_T = tensor.transpose(W)
        return ops.multiply(0.5, ops.subtract(W, W_T))
    
    def compute_viscosity(self, state):
        """
        Compute Boltzmann-modulated viscosity based on state energy.
        
        Args:
            state: The state tensor
            
        Returns:
            Viscosity tensor
        """
        # Calculate energy as sum of squared state elements
        energy = stats.sum(ops.square(state), axis=-1, keepdims=True)
        
        # Calculate viscosity using Boltzmann factor
        return ops.multiply(self.nu_0, ops.exp(ops.multiply(-self.beta, energy)))
    
    def apply_b_symplectic(self, h_c, h_s):
        """
        Apply b-Poisson bracket to preserve geometric structure.
        
        Args:
            h_c: Classical component of the state
            h_s: Stochastic component of the state
            
        Returns:
            Updated classical and stochastic components
        """
        # Compute squared magnitude for b-Poisson bracket
        h_squared = ops.add(ops.square(h_c), ops.square(h_s))
        
        # Apply b-Poisson bracket
        dh_c = ops.multiply(h_squared, ops.matmul(h_s, self.W_symplectic))
        dh_s = ops.multiply(h_squared, ops.matmul(h_c, ops.negative(self.W_symplectic)))
        
        # Update states
        h_c_new = ops.add(h_c, dh_c)
        h_s_new = ops.add(h_s, dh_s)
        
        return h_c_new, h_s_new
    
    def apply_stochastic_quantum(self, h_c, h_s, dt):
        """
        Apply stochastic-quantum mapping to emulate quantum effects.
        
        Args:
            h_c: Classical component of the state
            h_s: Stochastic component of the state
            dt: Time step
            
        Returns:
            Updated classical and stochastic components
        """
        # Apply drift term to classical component: A * h_c
        drift_c = ops.matmul(h_c, self.A_matrix)
        
        # Apply drift term to stochastic component: A * h_s
        drift_s = ops.matmul(h_s, self.A_matrix)
        
        # Generate correlated noise for classical component
        noise_c = tensor.random_normal(tensor.shape(h_c), stddev=self.noise_scale)
        
        # Generate correlated noise for stochastic component
        noise_s = tensor.random_normal(tensor.shape(h_s), stddev=self.noise_scale)
        
        # Apply diffusion term to classical component: B * noise_c
        diffusion_c = ops.matmul(noise_c, self.B_matrix)
        
        # Apply diffusion term to stochastic component: B * noise_s
        diffusion_s = ops.matmul(noise_s, self.B_matrix)
        
        # Update classical component
        h_c_new = ops.add(h_c, ops.add(
            ops.multiply(drift_c, dt),
            ops.multiply(diffusion_c, ops.sqrt(dt))
        ))
        
        # Update stochastic component
        h_s_new = ops.add(h_s, ops.add(
            ops.multiply(drift_s, dt),
            ops.multiply(diffusion_s, ops.sqrt(dt))
        ))
        
        return h_c_new, h_s_new
    
    def apply_harmonic_embedding(self, x, t):
        """
        Apply harmonic embedding to convert inputs into time-evolving waveforms.
        
        Args:
            x: Input tensor
            t: Time tensor
            
        Returns:
            Time-evolving waveform embedding
        """
        if not self.use_harmonic_embedding:
            return ops.add(ops.matmul(x, self.kernel), self.bias)
        
        # Reshape t for broadcasting
        t_expanded = tensor.reshape(t, (-1, 1, 1))
        
        # Compute phase: 2π * f_j * t + φ_j
        phase = ops.add(
            ops.multiply(
                ops.multiply(2.0 * ops.pi, self.frequencies),
                t_expanded
            ),
            self.phases
        )
        
        # Compute sine wave
        sine_wave = ops.sin(phase)
        
        # Apply amplitude
        waveform = ops.multiply(self.amplitudes, sine_wave)
        
        # Modulate waveform by input
        x_expanded = tensor.reshape(x, (-1, tensor.shape(x)[-1], 1))
        modulated_waveform = ops.multiply(x_expanded, waveform)
        
        # Sum across input dimension
        output = stats.sum(modulated_waveform, axis=1)
        
        return output
    
    def forward(self, inputs, initial_state=None, time_deltas=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_length, features) if batch_first=True,
                   or (seq_length, batch_size, features) if batch_first=False
            initial_state: Initial state tuple (h_c, h_s, t)
            time_deltas: Time deltas between inputs (optional)
            
        Returns:
            Layer output and final state if return_state is True, otherwise just the layer output
        """
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get input shape
        input_shape = tensor.shape(inputs)
        if len(input_shape) != 3:
            raise ValueError("Input tensor must be 3D (batch, time, features)")
        
        # Get batch size and sequence length
        if self.batch_first:
            batch_size, seq_length, _ = input_shape
        else:
            seq_length, batch_size, _ = input_shape
        
        # Initialize state if not provided
        if initial_state is None:
            h_c = tensor.zeros((batch_size, self.neuron_map.units))
            h_s = tensor.zeros((batch_size, self.neuron_map.units))
            t = tensor.zeros((batch_size, 1))
            state = (h_c, h_s, t)
        else:
            state = initial_state
        
        # Process sequence
        outputs = []
        
        # Process sequence in forward or backward order
        time_indices = range(seq_length)
        
        # Process each time step
        for t_idx in time_indices:
            # Get current input
            if self.batch_first:
                x_t = inputs[:, t_idx]
            else:
                x_t = inputs[t_idx]
            
            # Get time delta for this step if provided
            dt = 1.0
            if time_deltas is not None:
                if self.batch_first:
                    dt = time_deltas[:, t_idx]
                else:
                    dt = time_deltas[t_idx]
            
            # Unpack state
            h_c, h_s, t = state
            
            # Update time
            t_new = ops.add(t, dt)
            
            # Apply harmonic embedding or standard projection
            if self.use_harmonic_embedding:
                x_projected = self.apply_harmonic_embedding(x_t, t)
            else:
                x_projected = ops.add(ops.matmul(x_t, self.kernel), self.bias)
            
            # Apply stochastic-quantum mapping
            h_c_sq, h_s_sq = self.apply_stochastic_quantum(h_c, h_s, dt)
            
            # Apply b-symplectic integrator
            h_c_bs, h_s_bs = self.apply_b_symplectic(h_c_sq, h_s_sq)
            
            # Compute viscosity
            viscosity = self.compute_viscosity(h_c_bs)
            
            # Compute time decay factor
            decay = ops.exp(ops.divide(-dt, self.time_scale))
            
            # Compute gradient: dL/dΨ ≈ W @ Ψ + inputs
            gradient_c = ops.add(ops.matmul(h_c_bs, self.recurrent_kernel), x_projected)
            gradient_s = ops.matmul(h_s_bs, self.recurrent_kernel)
            
            # Update state with viscosity and time decay
            step_c = ops.multiply(viscosity, gradient_c)
            step_s = ops.multiply(viscosity, gradient_s)
            
            # Apply time decay to hidden state
            h_c_decayed = ops.multiply(decay, h_c_bs)
            h_s_decayed = ops.multiply(decay, h_s_bs)
            
            # Add gradient step
            h_c_new = ops.add(h_c_decayed, ops.multiply(ops.subtract(tensor.ones_like(decay), decay), step_c))
            h_s_new = ops.add(h_s_decayed, ops.multiply(ops.subtract(tensor.ones_like(decay), decay), step_s))
            
            # Combine classical and stochastic components for output
            output = ops.add(h_c_new, h_s_new)
            
            # Store output
            outputs.append(output)
            
            # Update state
            state = (h_c_new, h_s_new, t_new)
        
        # Stack outputs
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            outputs_tensor = tensor.stack(outputs, axis=stack_dim)
        else:
            outputs_tensor = outputs[-1]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs_tensor, state
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
        h_c = tensor.zeros((batch_size, self.neuron_map.units))
        h_s = tensor.zeros((batch_size, self.neuron_map.units))
        t = tensor.zeros((batch_size, 1))
        return (h_c, h_s, t)
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the CTRQNet layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "nu_0": self.nu_0,
            "beta": self.beta,
            "noise_scale": self.noise_scale,
            "time_scale_factor": self.time_scale_factor,
            "use_harmonic_embedding": self.use_harmonic_embedding,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "batch_first": self.batch_first
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CTRQNet':
        """Creates a CTRQNet layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)