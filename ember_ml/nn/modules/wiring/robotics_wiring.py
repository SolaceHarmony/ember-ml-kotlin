"""
Robotics Wiring Pattern for Control and State Estimation

This module provides a specialized wiring pattern for robotics applications,
implementing sensor processing, state estimation, and control layers.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn import tensor

class RoboticsWiring(NeuronMap):
    """
    Custom wiring for robotics applications.
    
    Architecture:
    - Sensor processing layer (sensor neurons)
    - State estimation layer (state neurons)
    - Control layer (control neurons)
    
    Features:
    - Direct sensor-to-motor connections for reflexes
    - Recurrent connections for state estimation
    - Multiple timescales for different control loops
    """
    
    def __init__(
        self,
        sensor_neurons: int,
        state_neurons: int,
        control_neurons: int,
        sensor_fanout: int = 4,
        state_recurrent: int = 3,
        control_fanin: int = 4,
        reflex_probability: float = 0.2,
        **kwargs
    ):
        """
        Initialize the robotics wiring pattern.
        
        Args:
            sensor_neurons: Number of sensor processing neurons
            state_neurons: Number of state estimation neurons
            control_neurons: Number of control output neurons
            sensor_fanout: Number of connections from each sensor neuron
            state_recurrent: Number of recurrent connections in state layer
            control_fanin: Number of connections to each control neuron
            reflex_probability: Probability of direct sensor-to-control connections
            **kwargs: Additional keyword arguments
        """
        # Calculate total units
        total_units = sensor_neurons + state_neurons + control_neurons
        
        # Initialize base class
        super().__init__(units=total_units, output_dim=control_neurons, **kwargs)
        
        # Store configuration
        self.sensor_neurons = sensor_neurons
        self.state_neurons = state_neurons
        self.control_neurons = control_neurons
        self.sensor_fanout = sensor_fanout
        self.state_recurrent = state_recurrent
        self.control_fanin = control_fanin
        self.reflex_probability = reflex_probability
        
        # Define neuron ranges
        self.control_range = range(control_neurons)
        self.state_range = range(
            control_neurons,
            control_neurons + state_neurons
        )
        self.sensor_range = range(
            control_neurons + state_neurons,
            total_units
        )
        
        # Mark as not built yet
        self.built = False
    
    def build(self, input_dim: Optional[int] = None):
        """
        Build the wiring pattern.
        
        Args:
            input_dim: Input dimension (optional)
        """
        if self.built:
            return
        
        # Set input dimension if provided
        if input_dim is not None:
            self.input_dim = input_dim
        
        # Initialize adjacency matrix if not already done
        if self.adjacency_matrix is None:
            self.adjacency_matrix = tensor.zeros((self.units, self.units))
        
        # Build connectivity
        self._build_sensor_connections()
        self._build_state_connections()
        self._build_control_connections()
        
        # Mark as built
        self.built = True
    
    def _build_sensor_connections(self):
        """Build connections from sensor layer."""
        # Connect sensors to state estimation
        for src in self.sensor_range:
            # Randomly select target state neurons
            for _ in range(self.sensor_fanout):
                dest = tensor.random_choice(
                    tensor.convert_to_tensor(list(self.state_range))
                )
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, src, dest, 1.0
                )
            
            # Direct sensor-to-motor connections (reflexes)
            if tensor.random_uniform(()) < self.reflex_probability:
                dest = tensor.random_choice(
                    tensor.convert_to_tensor(list(self.control_range))
                )
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, src, dest, 1.0
                )
    
    def _build_state_connections(self):
        """Build connections in state estimation layer."""
        # Recurrent connections for state memory
        for _ in range(self.state_recurrent):
            src = tensor.random_choice(
                tensor.convert_to_tensor(list(self.state_range))
            )
            dest = tensor.random_choice(
                tensor.convert_to_tensor(list(self.state_range))
            )
            self.adjacency_matrix = tensor.with_value(
                self.adjacency_matrix, src, dest, 1.0
            )
    
    def _build_control_connections(self):
        """Build connections to control layer."""
        # Connect state estimation to control
        for dest in self.control_range:
            # Randomly select source state neurons
            for _ in range(self.control_fanin):
                src = tensor.random_choice(
                    tensor.convert_to_tensor(list(self.state_range))
                )
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, src, dest, 1.0
                )
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the wiring."""
        config = super().get_config()
        config.update({
            "sensor_neurons": self.sensor_neurons,
            "state_neurons": self.state_neurons,
            "control_neurons": self.control_neurons,
            "sensor_fanout": self.sensor_fanout,
            "state_recurrent": self.state_recurrent,
            "control_fanin": self.control_fanin,
            "reflex_probability": self.reflex_probability
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RoboticsWiring':
        """Creates a wiring from its configuration."""
        return cls(**config)