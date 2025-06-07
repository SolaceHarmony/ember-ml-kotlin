"""
Frequency Wiring Pattern for Signal Decomposition

This module provides a specialized wiring pattern for frequency analysis tasks,
implementing frequency-specific neurons and harmonic connections.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules.wiring import NeuronMap
from ember_ml.nn import tensor

class FrequencyWiring(NeuronMap):
    """
    Wiring pattern for frequency analysis.
    
    Features:
    - Frequency-specific neurons
    - Harmonic connections
    - Phase relationships
    
    This wiring pattern is designed for decomposing signals into frequency components,
    with dedicated neurons for each frequency band and connections that capture
    harmonic relationships between frequencies.
    """
    
    def __init__(
        self,
        input_size: int,
        freq_neurons: int = 32,
        num_freqs: int = 4,
        harmonic_connections: bool = True,
        **kwargs
    ):
        """
        Initialize the frequency wiring pattern.
        
        Args:
            input_size: Size of the input signal
            freq_neurons: Number of neurons per frequency
            num_freqs: Number of frequency components
            harmonic_connections: Whether to add harmonic connections
            **kwargs: Additional keyword arguments
        """
        # Calculate total units
        total_units = freq_neurons * num_freqs
        
        # Initialize base class
        super().__init__(units=total_units, output_dim=num_freqs, **kwargs)
        
        # Store configuration
        self.input_size = input_size
        self.freq_neurons = freq_neurons
        self.num_freqs = num_freqs
        self.harmonic_connections = harmonic_connections
        
        # Define frequency ranges
        self.freq_ranges = [
            range(
                i * freq_neurons,
                (i + 1) * freq_neurons
            )
            for i in range(num_freqs)
        ]
        
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
        else:
            self.input_dim = self.input_size
        
        # Initialize adjacency matrix if not already done
        if self.adjacency_matrix is None:
            self.adjacency_matrix = tensor.zeros((self.units, self.units))
        
        # Build connectivity
        self._build_freq_connections()
        if self.harmonic_connections:
            self._build_harmonic_connections()
        
        # Mark as built
        self.built = True
    
    def _build_freq_connections(self):
        """Build connections within each frequency band."""
        for freq_range in self.freq_ranges:
            # Dense connectivity within frequency band
            for src in freq_range:
                for dest in freq_range:
                    if src != dest:  # No self-connections
                        self.adjacency_matrix = tensor.with_value(
                            self.adjacency_matrix, src, dest, 1.0
                        )
    
    def _build_harmonic_connections(self):
        """Build connections between harmonically related frequencies."""
        # Connect fundamental frequency to harmonics
        fundamental_range = self.freq_ranges[0]
        
        for i in range(1, self.num_freqs):
            harmonic_range = self.freq_ranges[i]
            
            # Connect fundamental to harmonic (sparse connections)
            for src in fundamental_range:
                if tensor.random_uniform(()) < 0.3:  # 30% chance of connection
                    dest = tensor.random_choice(
                        tensor.convert_to_tensor(list(harmonic_range))
                    )
                    self.adjacency_matrix = tensor.with_value(
                        self.adjacency_matrix, src, dest, 1.0
                    )
            
            # Connect harmonic to fundamental (feedback)
            for src in harmonic_range:
                if tensor.random_uniform(()) < 0.2:  # 20% chance of connection
                    dest = tensor.random_choice(
                        tensor.convert_to_tensor(list(fundamental_range))
                    )
                    self.adjacency_matrix = tensor.with_value(
                        self.adjacency_matrix, src, dest, 1.0
                    )
            
            # Connect to other harmonics
            for j in range(i+1, self.num_freqs):
                other_harmonic_range = self.freq_ranges[j]
                
                # Sparse connections between harmonics
                for src in harmonic_range:
                    if tensor.random_uniform(()) < 0.1:  # 10% chance of connection
                        dest = tensor.random_choice(
                            tensor.convert_to_tensor(list(other_harmonic_range))
                        )
                        self.adjacency_matrix = tensor.with_value(
                            self.adjacency_matrix, src, dest, 1.0
                        )
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the wiring."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "freq_neurons": self.freq_neurons,
            "num_freqs": self.num_freqs,
            "harmonic_connections": self.harmonic_connections
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FrequencyWiring':
        """Creates a wiring from its configuration."""
        return cls(**config)