"""
GUCE Neural Circuit Policy Map

This module provides a NeuronMap implementation that integrates GUCE neurons with
Neural Circuit Policy (NCP) connectivity structures.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn.modules.rnn import GUCE
from ember_ml.nn import tensor

class GUCENCPMap(NeuronMap):
    """
    Neural Circuit Policy map with GUCE neurons.
    
    This NeuronMap defines a structured connectivity pattern based on the
    Neural Circuit Policy (NCP) architecture, with sensory, inter, command,
    and motor neurons organized in a layered structure.
    
    Features:
    - Sensory, inter, and command neuron organization
    - Structured connectivity for control applications
    - Integration with GUCE neuron dynamics
    """
    
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_fanout: int = 4,
        inter_fanout: int = 2,
        recurrent_command_synapses: int = 4,
        motor_fanin: int = 6,
        state_dim: int = 32,
        step_size: float = 0.01,
        nu_0: float = 1.0,
        beta: float = 0.1,
        theta_freq: float = 4.0,
        gamma_freq: float = 40.0,
        dt: float = 0.01,
        **kwargs
    ):
        """
        Initialize the GUCE Neural Circuit Policy map.
        
        Args:
            inter_neurons: Number of inter neurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_fanout: Number of outgoing synapses from each sensory neuron
            inter_fanout: Number of outgoing synapses from each inter neuron
            recurrent_command_synapses: Number of recurrent synapses for each command neuron
            motor_fanin: Number of incoming synapses to each motor neuron
            state_dim: Dimension of the GUCE neuron state
            step_size: Learning rate for GUCE neurons
            nu_0: Base viscosity for GUCE neurons
            beta: Energy scaling for GUCE neurons
            theta_freq: Theta oscillation frequency (Hz)
            gamma_freq: Gamma oscillation frequency (Hz)
            dt: Time step size
            **kwargs: Additional keyword arguments
        """
        # Calculate total units
        self.sensory_neurons = kwargs.get('sensory_neurons', None)
        if self.sensory_neurons is None:
            self.sensory_neurons = motor_neurons  # Default: same as motor neurons
        
        total_units = self.sensory_neurons + inter_neurons + command_neurons + motor_neurons
        
        # Initialize base class
        super().__init__(units=total_units, output_dim=motor_neurons, **kwargs)
        
        # Store neuron counts
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        
        # Store connectivity parameters
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin
        
        # Store GUCE parameters
        self.state_dim = state_dim
        self.step_size = step_size
        self.nu_0 = nu_0
        self.beta = beta
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.dt = dt
        
        # Mark as not built yet
        self.built = False
    
    def build(self, input_dim: Optional[int] = None):
        """
        Build the NeuronMap.
        
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
        self._build_sensory_connections()
        self._build_inter_connections()
        self._build_command_connections()
        self._build_motor_connections()
        
        # Mark as built
        self.built = True
    
    def _build_sensory_connections(self):
        """Build connections from sensory neurons."""
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = inter_start + self.inter_neurons
        
        # Connect each sensory neuron to random inter neurons
        for i in range(sensory_start, sensory_end):
            # Select random inter neurons to connect to
            targets = tensor.random_choice(
                tensor.arange(inter_start, inter_end),
                self.sensory_fanout,
                replace=False
            )
            
            # Add connections
            for target in targets:
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, i, target, 1.0
                )
    
    def _build_inter_connections(self):
        """Build connections from inter neurons."""
        inter_start = self.sensory_neurons
        inter_end = inter_start + self.inter_neurons
        command_start = inter_end
        command_end = command_start + self.command_neurons
        
        # Connect each inter neuron to random command neurons
        for i in range(inter_start, inter_end):
            # Select random command neurons to connect to
            targets = tensor.random_choice(
                tensor.arange(command_start, command_end),
                self.inter_fanout,
                replace=False
            )
            
            # Add connections
            for target in targets:
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, i, target, 1.0
                )
    
    def _build_command_connections(self):
        """Build connections from command neurons."""
        command_start = self.sensory_neurons + self.inter_neurons
        command_end = command_start + self.command_neurons
        
        # Add recurrent connections within command neurons
        for i in range(command_start, command_end):
            # Select random command neurons to connect to (including self)
            targets = tensor.random_choice(
                tensor.arange(command_start, command_end),
                self.recurrent_command_synapses,
                replace=False
            )
            
            # Add connections
            for target in targets:
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, i, target, 1.0
                )
    
    def _build_motor_connections(self):
        """Build connections to motor neurons."""
        command_start = self.sensory_neurons + self.inter_neurons
        command_end = command_start + self.command_neurons
        motor_start = command_end
        motor_end = motor_start + self.motor_neurons
        
        # Connect command neurons to motor neurons
        for i in range(motor_start, motor_end):
            # Select random command neurons to connect from
            sources = tensor.random_choice(
                tensor.arange(command_start, command_end),
                self.motor_fanin,
                replace=False
            )
            
            # Add connections
            for source in sources:
                self.adjacency_matrix = tensor.with_value(
                    self.adjacency_matrix, source, i, 1.0
                )
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the NeuronMap."""
        config = super().get_config()
        config.update({
            "sensory_neurons": self.sensory_neurons,
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons,
            "motor_neurons": self.motor_neurons,
            "sensory_fanout": self.sensory_fanout,
            "inter_fanout": self.inter_fanout,
            "recurrent_command_synapses": self.recurrent_command_synapses,
            "motor_fanin": self.motor_fanin,
            "state_dim": self.state_dim,
            "step_size": self.step_size,
            "nu_0": self.nu_0,
            "beta": self.beta,
            "theta_freq": self.theta_freq,
            "gamma_freq": self.gamma_freq,
            "dt": self.dt
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GUCENCPMap':
        """Creates a NeuronMap from its configuration."""
        return cls(**config)