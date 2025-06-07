"""
Base geometric neural implementations for non-Euclidean manifolds.
"""

from typing import Dict, Any
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module

def normalize_sphere(vec):
    """
    Normalize a vector to the unit sphere.

    Args:
        vec: Input vector

    Returns:
        Normalized vector on unit sphere
    """
    norm = ops.linalg.norm(vec)
    if ops.less(norm, 1e-12):
        return vec
    return ops.divide(vec, norm)

class GeometricNeuron(Module):
    """Base class for geometry-aware neural processing."""
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 dim: int = 3):
        """
        Initialize geometric neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau: Time constant
            dt: Time step for numerical integration
            dim: Dimension of the manifold
        """
        super().__init__(neuron_id, tau, dt)
        self.dim = dim
        self.manifold_state = self._initialize_manifold_state()
    
    def _initialize_state(self):
        """Initialize neuron state."""
        return self._initialize_manifold_state()
        
    def _initialize_manifold_state(self):
        """Initialize state on the manifold. Override in subclasses."""
        return tensor.zeros(self.dim)
        
    def _manifold_update(self, 
                        current_state,
                        target_state,
                        **kwargs):
        """
        Update state according to manifold geometry.
        
        Args:
            current_state: Current state on manifold
            target_state: Target state on manifold
            **kwargs: Additional parameters
            
        Returns:
            Updated state on manifold
        """
        raise NotImplementedError(
            "Manifold update must be implemented by subclass"
        )
        
    def update(self, 
               input_signal,
               **kwargs):
        """
        Update neuron state using manifold geometry.
        
        Args:
            input_signal: Input state on manifold
            **kwargs: Additional parameters
            
        Returns:
            Updated state on manifold
        """
        self.manifold_state = self._manifold_update(
            self.manifold_state,
            input_signal,
            **kwargs
        )
        self.state = self.manifold_state
        self.history.append(tensor.copy(self.state))
        return self.state
        
    def save_state(self) -> Dict[str, Any]:
        """Save neuron state and parameters."""
        state_dict = super().save_state()
        state_dict.update({
            'dim': self.dim,
            'manifold_state': tensor.to_numpy(self.manifold_state).tolist()
        })
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load neuron state and parameters."""
        super().load_state(state_dict)
        self.dim = state_dict['dim']
        self.manifold_state = tensor.convert_to_tensor(state_dict['manifold_state'])
