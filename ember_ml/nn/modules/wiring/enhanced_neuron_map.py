"""
Enhanced Neuron Map for Spatially Embedded Neural Networks.

This module provides an enhanced neuron map implementation that supports
arbitrary neuron types and dynamics, with a focus on spatial embedding.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from ember_ml.nn.tensor.types import TensorLike # Corrected import
import numpy as np
import scipy.spatial.distance

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter

class EnhancedNeuronMap:
    """
    Enhanced NeuronMap that supports arbitrary neuron types and dynamics.
    
    This class extends the basic NeuronMap with support for:
    1. Neuron-type specific parameters
    2. Dynamic properties that affect temporal processing
    3. Spatial constraints that affect connectivity
    """
    
    def __init__(
        self,
        units: int,
        output_dim: int,
        input_dim: Optional[int] = None,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        # Dynamic properties
        neuron_type: str = "cfc",
        neuron_params: Optional[Dict[str, Any]] = None,
        # Spatial properties
        coordinates_list: Optional[List[TensorLike]] = None,
        network_structure: Tuple[int, int, int] = (5, 5, 4),
        distance_metric: str = "euclidean",
        distance_power: float = 1.0,
    ):
        """
        Initialize an enhanced neuron map.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions
            input_dim: Number of input dimensions (default: units)
            sparsity_level: Sparsity level for the connections
            seed: Random seed for reproducibility
            
            # Dynamic properties
            neuron_type: Type of neuron to use ("cfc", "ltc", "ctgru", etc.)
            neuron_params: Parameters specific to the neuron type
            
            # Spatial properties
            coordinates_list: Optional list of coordinate arrays for neurons
            network_structure: 3D structure of the network if coordinates not provided
            distance_metric: Metric to use for distance calculations
            distance_power: Power to raise distances to (1.0 = linear, 2.0 = quadratic)
        """
        # Initialize base properties
        self.units = units
        self.output_dim = output_dim
        self.input_dim = input_dim if input_dim is not None else units
        self.sparsity_level = sparsity_level
        self.seed = seed
        
        # Initialize neuron-specific properties
        self.neuron_type = neuron_type
        self.neuron_params = neuron_params or {}
        
        # Initialize spatial properties
        self._initialize_spatial_properties(coordinates_list, network_structure, 
                                           distance_metric, distance_power)
        
        # Initialize masks and adjacency matrices
        self._input_mask = None
        self._recurrent_mask = None
        self._output_mask = None
        self.adjacency_matrix = tensor.zeros([units, units], dtype=tensor.int32)
        self.sensory_adjacency_matrix = None
        self._built = False
    
    def _initialize_spatial_properties(self, coordinates_list, network_structure, 
                                      distance_metric, distance_power):
        """Initialize spatial properties of the neuron map."""
        if coordinates_list is not None:
            self.coordinates = coordinates_list
        else:
            # Set up neurons per dimension using tensor operations
            nx = tensor.arange(network_structure[0])
            ny = tensor.arange(network_structure[1])
            nz = tensor.arange(network_structure[2])
            
            # Use tensor.meshgrid for coordinate grid
            [x, y, z] = tensor.meshgrid(nx, ny, nz)
            
            # Flatten and slice to get coordinates
            x_flat = tensor.reshape(x, [-1])[:self.units]
            y_flat = tensor.reshape(y, [-1])[:self.units]
            z_flat = tensor.reshape(z, [-1])[:self.units]
            
            self.coordinates = [
                tensor.to_numpy(x_flat),  # Convert to numpy for compatibility with existing code
                tensor.to_numpy(y_flat),
                tensor.to_numpy(z_flat)
            ]
        
        # Convert coordinates to tensor for processing
        coords_tensor = [tensor.convert_to_tensor(coord) for coord in self.coordinates]
        coords_stacked = tensor.stack(coords_tensor, axis=1)
        
        # Calculate the distance matrix using scipy temporarily
        # This will be replaced with a pure tensor implementation when available
        coords_np = tensor.to_numpy(coords_stacked)
        euclidean_vector = scipy.spatial.distance.pdist(coords_np, metric=distance_metric)
        euclidean = scipy.spatial.distance.squareform(euclidean_vector**distance_power)
        self.distance_matrix = tensor.convert_to_tensor(euclidean, dtype=tensor.float32)
        
        # Calculate communicability matrix (will be updated during build)
        self.communicability_matrix = tensor.ones_like(self.distance_matrix)
    
    def build(self, input_dim=None):
        """
        Build the neuron map.
        
        This method should be overridden by subclasses to implement
        specific connectivity patterns.
        """
        raise NotImplementedError("Subclasses must implement build method")
    
    def is_built(self):
        """
        Check if the neuron map is built.
        
        Returns:
            True if the neuron map is built, False otherwise
        """
        return self._built
    
    def set_input_dim(self, input_dim):
        """
        Set the input dimension.
        
        Args:
            input_dim: Input dimension
        """
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = tensor.zeros([input_dim, self.units], dtype=tensor.int32)
    
    def get_input_mask(self):
        """
        Get the input mask.
        
        Returns:
            Input mask tensor
        """
        if self._input_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        return self._input_mask
    
    def get_recurrent_mask(self):
        """
        Get the recurrent mask.
        
        Returns:
            Recurrent mask tensor
        """
        if self._recurrent_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        return self._recurrent_mask
    
    def get_output_mask(self):
        """
        Get the output mask.
        
        Returns:
            Output mask tensor
        """
        if self._output_mask is None:
            self._input_mask, self._recurrent_mask, self._output_mask = self.build()
        
        return self._output_mask
    
    def get_neuron_factory(self):
        """
        Get a factory function for creating neurons of the specified type.
        
        Returns:
            A function that creates neurons with the specified parameters
        """
        if self.neuron_type == "cfc":
            from ember_ml.nn.modules.rnn import CfCCell
            
            def factory(neuron_id, **kwargs):
                params = self.neuron_params.copy()
                params.update(kwargs)
                return CfCCell(
                    input_size=self.input_dim,
                    hidden_size=1,  # Single neuron
                    neuron_id=neuron_id,
                    **params
                )
            
            return factory
        elif self.neuron_type == "ltc":
            from ember_ml.nn.modules.rnn import LTCCell
            
            def factory(neuron_id, **kwargs):
                params = self.neuron_params.copy()
                params.update(kwargs)
                return LTCCell(
                    input_size=self.input_dim,
                    hidden_size=1,  # Single neuron
                    neuron_id=neuron_id,
                    **params
                )
            
            return factory
        else:
            raise ValueError(f"Unsupported neuron type: {self.neuron_type}")
    
    def get_dynamic_properties(self):
        """
        Get dynamic properties for the neuron map.
        
        Returns:
            Dictionary of dynamic properties
        """
        return {
            "neuron_type": self.neuron_type,
            "neuron_params": self.neuron_params
        }
    
    def get_spatial_properties(self):
        """
        Get spatial properties for the neuron map.
        
        Returns:
            Dictionary of spatial properties
        """
        return {
            "coordinates": self.coordinates,
            "distance_matrix": self.distance_matrix,
            "communicability_matrix": self.communicability_matrix
        }
    
    def get_config(self):
        """
        Get the configuration of the neuron map.
        
        Returns:
            Dictionary containing the configuration
        """
        config = {
            "units": self.units,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "sparsity_level": self.sparsity_level,
            "seed": self.seed,
            "neuron_type": self.neuron_type,
            "neuron_params": self.neuron_params,
            # We don't store the full coordinates/matrices in the config
            # but rather the parameters used to generate them
            "network_structure": self._get_network_structure()
        }
        return config
    
    def _get_network_structure(self):
        """Get the network structure from coordinates."""
        if not hasattr(self, 'coordinates') or self.coordinates is None:
            return (5, 5, 4)  # Default
        
        # Use Python's built-in max function directly on the coordinates
        x_dim = max(self.coordinates[0]) + 1
        y_dim = max(self.coordinates[1]) + 1
        z_dim = max(self.coordinates[2]) + 1
        
        return (x_dim, y_dim, z_dim)
    
    @classmethod
    def from_config(cls, config):
        """
        Create a neuron map from a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Neuron map instance
        """
        return cls(**config)