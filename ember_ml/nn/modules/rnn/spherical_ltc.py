"""
Spherical variant of LTC neurons operating on unit sphere manifold.
"""

from typing import Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass
from ember_ml.ops import linearalg
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn.geometric import GeometricNeuron, normalize_sphere
from ember_ml.nn.modules.rnn.blocky import BaseChain
@dataclass
class SphericalLTCConfig:
    """Configuration for Spherical LTC neurons."""
    
    tau: float = 1.0
    gleak: float = 0.5
    dt: float = 0.01

def log_map_sphere(p, q):
    """
    Compute logarithmic map from p to q on unit sphere.

    Args:
        p: Base point on sphere
        q: Target point on sphere

    Returns:
        Tangent vector at p pointing toward q
    """
    p_n = normalize_sphere(p)
    q_n = normalize_sphere(q)
    
    # Compute geodesic distance
    dot_prod = ops.clip(ops.dot(p_n, q_n), -1.0, 1.0)
    angle = ops.arccos(dot_prod)
    
    if ops.less(angle, 1e-12):
        return tensor.zeros_like(p)
        
    # Compute direction in tangent space
    perp = ops.subtract(q_n, ops.multiply(dot_prod, p_n))
    perp_norm = linearalg.norm(perp)
    
    if ops.less(perp_norm, 1e-12):
        return tensor.zeros_like(p)
        
    return ops.multiply(ops.divide(perp, perp_norm), angle)

def exp_map_sphere(p, v):
    """
    Compute exponential map of v at p on unit sphere.

    Args:
        p: Base point on sphere
        v: Tangent vector at p

    Returns:
        Point on sphere reached by geodesic in direction v
    """
    norm_v = linearalg.norm(v)
    if ops.less(norm_v, 1e-12):
        return p
    
    p_n = normalize_sphere(p)
    dir_v = ops.divide(v, norm_v)
    
    # Remove component along p
    dir_v = ops.subtract(dir_v, ops.multiply(ops.dot(dir_v, p_n), p_n))
    dir_v = normalize_sphere(dir_v)
    
    # Compute point on sphere
    result = ops.add(
        ops.multiply(ops.cos(norm_v), p_n),
        ops.multiply(ops.sin(norm_v), dir_v)
    )
    return normalize_sphere(result)

class SphericalLTCNeuron(GeometricNeuron):
    """LTC neuron implementation on unit sphere."""
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 dim: int = 3):
        """
        Initialize spherical LTC neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau: Time constant
            dt: Time step for numerical integration
            gleak: Leak conductance
            dim: Dimension of sphere (default 3 for S²)
        """
        # Set dim before calling super().__init__
        self.dim = dim
        self.gleak = gleak
        self.baseline = tensor.zeros(dim)
        # Use tensor_scatter_update to set the last element to 1.0 (North pole as rest state)
        self.baseline = tensor.tensor_scatter_nd_update(
            self.baseline, 
            [ops.subtract(dim, 1)], 
            [1.0]
        )
        
        # Call super().__init__ after setting dim
        super().__init__(neuron_id, tau, dt, dim)
        
    def _initialize_manifold_state(self):
        """Initialize state as random point on sphere."""
        state = tensor.random_normal(shape=(self.dim,))
        return normalize_sphere(state)
        
    def _manifold_update(self, 
                        current_state,
                        target_state,
                        **kwargs):
        """
        Update state according to spherical geometry.
        
        Args:
            current_state: Current state on sphere
            target_state: Target state on sphere
            **kwargs: Additional parameters
            
        Returns:
            Updated state on sphere
        """
        # Compute LTC update in tangent space
        v = log_map_sphere(current_state, target_state)
        v_scaled = ops.multiply(ops.divide(self.dt, self.tau), v)
        
        # Update state via exponential map
        updated_state = exp_map_sphere(current_state, v_scaled)
        
        # Apply leak toward baseline
        if ops.greater(self.gleak, 0):
            v_leak = log_map_sphere(updated_state, self.baseline)
            leak_scale = ops.multiply(self.gleak, self.dt)
            updated_state = exp_map_sphere(updated_state, ops.multiply(leak_scale, v_leak))
            
        return updated_state
        
    def save_state(self) -> Dict[str, Any]:
        """Save neuron state and parameters."""
        state_dict = super().save_state()
        state_dict.update({
            'gleak': self.gleak,
            'baseline': tensor.to_numpy(self.baseline).tolist()
        })
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load neuron state and parameters."""
        super().load_state(state_dict)
        self.gleak = state_dict['gleak']
        self.baseline = tensor.convert_to_tensor(state_dict['baseline'])

class SphericalLTCChain(BaseChain):
    """Chain of LTC neurons operating on unit sphere."""
    
    def __init__(self,
                 num_neurons: int,
                 base_tau_or_config: Optional[Union[float, SphericalLTCConfig]] = None,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 dim: int = 3):
        """
        Initialize spherical LTC chain.

        Args:
            num_neurons: Number of neurons in chain
            base_tau_or_config: Base time constant or SphericalLTCConfig object
            dt: Time step
            gleak: Leak conductance
            dim: Dimension of sphere
        """
        self.dim = dim
        
        # Handle config object
        if isinstance(base_tau_or_config, SphericalLTCConfig):
            config = base_tau_or_config
            base_tau = config.tau
            dt = config.dt
            gleak = config.gleak
        else:
            base_tau = 1.0 if base_tau_or_config is None else float(base_tau_or_config)
        
        # Create a neuron factory class
        class SphericalLTCFactory(SphericalLTCNeuron):
            def __new__(cls, neuron_id, tau, dt):
                return SphericalLTCNeuron(
                    neuron_id=neuron_id,
                    tau=tau,
                    dt=dt,
                    gleak=gleak,
                    dim=dim
                )

        super().__init__(
            num_neurons=num_neurons,
            neuron_class=SphericalLTCFactory,
            base_tau=base_tau,
            dt=dt
        )
    
    def __call__(self, input_batch):
        """
        Process input batch through the chain.
        
        Args:
            input_batch: Input batch tensor (batch_size, dim)
            
        Returns:
            Tuple of (states tensor, metadata dict)
        """
        # Get batch size from input shape
        batch_size = tensor.shape(input_batch)[0]
        
        # Process each batch element
        all_states = []
        for i in range(batch_size):
            # Extract single batch element using slice
            input_i = tensor.slice_tensor(input_batch, [i, 0], [1, self.dim])
            # Update chain with this input
            states = self.update(input_i)
            all_states.append(states)
            
        # Stack all states
        states_tensor = tensor.stack(all_states)
        
        # Return states and empty metadata
        return states_tensor, {}
    
    def reset_states(self, batch_size: int = 1) -> None:
        """Reset chain states for batch processing.
        
        Args:
            batch_size: Batch size for parallel simulation
        """
        self.reset()
        # No additional batch handling needed for now
    
    def get_forgetting_times(self, states, threshold: float = 0.1):
        """Compute forgetting times for each neuron.
        
        Args:
            states: State history tensor (batch_size, num_steps, num_neurons, 3)
            threshold: Threshold for considering a state as forgotten
            
        Returns:
            Dictionary mapping neuron indices to forgetting times
        """
        # Get tensor dimensions
        shape = tensor.shape(states)
        batch_size = shape[0]
        num_steps = shape[1]
        num_neurons = shape[2]
        
        # Use first batch element for analysis
        states_0 = tensor.slice_tensor(states, [0, 0, 0, 0], [1, num_steps, num_neurons, self.dim])
        states_0 = tensor.squeeze(states_0, axis=0)
        
        # Compute x-axis projections (first component)
        x_proj = tensor.slice_tensor(states_0, [0, 0, 0], [num_steps, num_neurons, 1])
        x_proj = tensor.squeeze(x_proj, axis=2)
        
        # Find pattern end index (assuming pattern is in first half)
        pattern_end = ops.floor_divide(num_steps, 2)
        
        # Compute forgetting times
        forgetting_times = {}
        
        for i in range(num_neurons):
            # Get neuron trace
            trace = tensor.slice_tensor(x_proj, [0, i], [num_steps, 1])
            trace = tensor.squeeze(trace, axis=1)
            
            # Get pattern end value
            pattern_value = tensor.slice_tensor(trace, [ops.subtract(pattern_end, 1)], [1])
            pattern_value = tensor.squeeze(pattern_value)
            
            # Find when trace drops below threshold
            found_forgetting = False
            for t in range(tensor.to_numpy(pattern_end), tensor.to_numpy(num_steps)):
                trace_t = tensor.slice_tensor(trace, [t], [1])
                trace_t = tensor.squeeze(trace_t)
                
                if ops.greater(ops.abs(ops.subtract(trace_t, pattern_value)), threshold):
                    forgetting_times[i] = ops.multiply(ops.subtract(t, pattern_end), self.dt)
                    found_forgetting = True
                    break
            
            if not found_forgetting:
                # Pattern maintained throughout simulation
                forgetting_times[i] = None
        
        return forgetting_times
        
    def update(self, input_signals):
        """
        Update chain state based on inputs.

        Args:
            input_signals: Input array for the chain [num_neurons x dim]

        Returns:
            Updated states of all neurons [num_neurons x dim]
        """
        # Create zeros tensor for states
        states = tensor.zeros((self.num_neurons, self.dim))
        
        # Update first neuron with external input
        states_0 = self.neurons[0].update(tensor.slice_tensor(input_signals, [0, 0], [1, self.dim]))
        states = tensor.tensor_scatter_nd_update(states, [0], [states_0])
        
        # Update subsequent neurons using chain connections
        for i in range(1, self.num_neurons):
            # Each neuron receives state of previous neuron as input
            prev_idx = ops.subtract(i, 1)
            prev_state = tensor.slice_tensor(states, [prev_idx, 0], [1, self.dim])
            states_i = self.neurons[i].update(prev_state)
            states = tensor.tensor_scatter_nd_update(states, [i], [states_i])
            
        # Store chain state history
        self.state_history.append(tensor.copy(states))
        
        return states

def create_spherical_ltc_chain(num_neurons: int,
                             base_tau_or_config=None,
                             dt: float = 0.01,
                             gleak: float = 0.5,
                             dim: int = 3) -> SphericalLTCChain:
    """
    Factory function to create a spherical LTC chain.

    Args:
        num_neurons: Number of neurons in chain
        base_tau_or_config: Base time constant or SphericalLTCConfig object
        dt: Time step
        gleak: Leak conductance
        dim: Dimension of sphere

    Returns:
        Configured spherical LTC chain
    """
    return SphericalLTCChain(
        num_neurons=num_neurons,
        base_tau_or_config=base_tau_or_config,
        dt=dt,
        gleak=gleak,
        dim=dim
    )
