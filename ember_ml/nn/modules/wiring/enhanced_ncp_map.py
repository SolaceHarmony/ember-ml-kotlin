"""
Enhanced Neural Circuit Policy (NCP) Map.

This module provides an enhanced NCP map implementation that supports
arbitrary neuron types and dynamics, with a focus on spatial embedding.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from ember_ml.nn.tensor.types import TensorLike # Added import
import numpy as np
import scipy.spatial.distance

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring.enhanced_neuron_map import EnhancedNeuronMap

class EnhancedNCPMap(EnhancedNeuronMap):
    """
    Enhanced Neural Circuit Policy (NCP) map.
    
    This class implements a Neural Circuit Policy connectivity pattern
    with support for arbitrary neuron types and dynamics.
    """
    
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_neurons: int = 0,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        # Dynamic properties
        neuron_type: str = "cfc",
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        mode: str = "default",
        # Spatial properties
        coordinates_list: Optional[List[TensorLike]] = None,
        network_structure: Tuple[int, int, int] = (5, 5, 4),
        distance_metric: str = "euclidean",
        distance_power: float = 1.0,
        # Connectivity properties
        sensory_to_inter_sparsity: Optional[float] = None,
        sensory_to_motor_sparsity: Optional[float] = None,
        inter_to_inter_sparsity: Optional[float] = None,
        inter_to_motor_sparsity: Optional[float] = None,
        motor_to_motor_sparsity: Optional[float] = None,
        motor_to_inter_sparsity: Optional[float] = None,
    ):
        """
        Initialize an enhanced NCP map.
        
        Args:
            inter_neurons: Number of inter neurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_neurons: Number of sensory neurons
            sparsity_level: Default sparsity level for all connections
            seed: Random seed for reproducibility
            
            # Dynamic properties
            neuron_type: Type of neuron to use ("cfc", "ltc", "ctgru", etc.)
            time_scale_factor: Factor to scale the time constant
            activation: Activation function for the output
            recurrent_activation: Activation function for the recurrent step
            mode: Mode of operation
            
            # Spatial properties
            coordinates_list: Optional list of coordinate arrays for neurons
            network_structure: 3D structure of the network if coordinates not provided
            distance_metric: Metric to use for distance calculations
            distance_power: Power to raise distances to
            
            # Connectivity properties
            sensory_to_inter_sparsity: Sparsity level for sensory to inter connections
            sensory_to_motor_sparsity: Sparsity level for sensory to motor connections
            inter_to_inter_sparsity: Sparsity level for inter to inter connections
            inter_to_motor_sparsity: Sparsity level for inter to motor connections
            motor_to_motor_sparsity: Sparsity level for motor to motor connections
            motor_to_inter_sparsity: Sparsity level for motor to inter connections
        """
        # Calculate total units
        units = sensory_neurons + inter_neurons + command_neurons + motor_neurons
        
        # Create neuron-specific parameters
        neuron_params = {
            "time_scale_factor": time_scale_factor,
            "activation": activation,
            "recurrent_activation": recurrent_activation,
            "mode": mode
        }
        
        # Initialize base class
        super().__init__(
            units=units,
            output_dim=motor_neurons,
            input_dim=sensory_neurons if sensory_neurons > 0 else None,
            sparsity_level=sparsity_level,
            seed=seed,
            neuron_type=neuron_type,
            neuron_params=neuron_params,
            coordinates_list=coordinates_list,
            network_structure=network_structure,
            distance_metric=distance_metric,
            distance_power=distance_power
        )
        
        # Store NCP-specific parameters
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_neurons = sensory_neurons
        
        # Store connectivity parameters
        self.sensory_to_inter_sparsity = sensory_to_inter_sparsity or sparsity_level
        self.sensory_to_motor_sparsity = sensory_to_motor_sparsity or sparsity_level
        self.inter_to_inter_sparsity = inter_to_inter_sparsity or sparsity_level
        self.inter_to_motor_sparsity = inter_to_motor_sparsity or sparsity_level
        self.motor_to_motor_sparsity = motor_to_motor_sparsity or sparsity_level
        self.motor_to_inter_sparsity = motor_to_inter_sparsity or sparsity_level
    
    def build(self, input_dim=None):
        """
        Build the NCP map.
        
        Args:
            input_dim: Input dimension (optional)
            
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set input_dim if provided
        if input_dim is not None:
            self.set_input_dim(input_dim)
        
        # Set random seed for reproducibility
        if self.seed is not None:
            tensor.set_seed(self.seed)
        
        # Create masks
        input_mask = tensor.ones((self.input_dim,), dtype=tensor.int32)
        
        # Define neuron group indices
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = sensory_end + self.inter_neurons
        command_start = inter_end
        command_end = inter_end + self.command_neurons
        motor_start = command_end
        motor_end = command_end + self.motor_neurons
        
        # Create output mask (only motor neurons contribute to output)
        output_mask = tensor.zeros((self.units,), dtype=tensor.int32)
        motor_indices = tensor.arange(motor_start, motor_end)
        motor_values = tensor.ones((motor_end - motor_start,), dtype=tensor.int32)
        output_mask = tensor.tensor_scatter_nd_update(
            output_mask,
            tensor.reshape(motor_indices, (-1, 1)),
            motor_values
        )
        
        # Initialize recurrent mask with zeros
        recurrent_mask = tensor.zeros((self.units, self.units), dtype=tensor.int32)
        
        # Helper function to create random connections between neuron groups
        def create_random_connections(from_start, from_end, to_start, to_end, sparsity):
            if from_end <= from_start or to_end <= to_start:
                return  # Skip if either group is empty
            
            # Create indices for the from and to neurons
            from_size = from_end - from_start
            to_size = to_end - to_start
            
            # Create a random mask for connections
            random_mask = tensor.random_uniform((from_size, to_size))
            
            # Apply spatial constraints
            from_coords = [coord[from_start:from_end] for coord in self.coordinates]
            to_coords = [coord[to_start:to_end] for coord in self.coordinates]
            
            # Convert coordinates to tensors and stack them
            from_coords_tensor = [tensor.convert_to_tensor(coord) for coord in from_coords]
            to_coords_tensor = [tensor.convert_to_tensor(coord) for coord in to_coords]
            
            from_points_tensor = tensor.stack(from_coords_tensor, axis=1)
            to_points_tensor = tensor.stack(to_coords_tensor, axis=1)
            
            # Convert to numpy temporarily for cdist (will be replaced with tensor equivalent)
            from_points = tensor.to_numpy(from_points_tensor)
            to_points = tensor.to_numpy(to_points_tensor)
            
            # Calculate pairwise distances
            distances = scipy.spatial.distance.cdist(
                from_points, to_points, metric="euclidean")
            
            # Convert distances to tensor
            distances_tensor = tensor.convert_to_tensor(distances)
            
            # Normalize distances to [0, 1] range
            max_dist = stats.max(distances_tensor)
            if tensor.to_numpy(max_dist) > 0:
                distances_tensor = ops.divide(distances_tensor, max_dist)
            
            # Adjust sparsity based on distance
            adjusted_sparsity = sparsity + ops.multiply((1 - sparsity), distances_tensor)
            
            # Create connection mask
            connection_mask = ops.greater_equal(random_mask, adjusted_sparsity)
            
            # Create indices for the connections
            from_indices = tensor.reshape(tensor.arange(from_start, from_end), (-1, 1, 1))
            to_indices = tensor.reshape(tensor.arange(to_start, to_end), (1, -1, 1))
            
            # Combine indices where connection_mask is True
            mask_indices = tensor.nonzero(connection_mask)
            if tensor.shape(mask_indices)[0] > 0:
                from_idx = from_indices[mask_indices[:, 0], 0, 0] 
                to_idx = to_indices[0, mask_indices[:, 1], 0]
                
                # Create update indices and values
                update_indices = tensor.stack([from_idx, to_idx], axis=1)
                update_values = tensor.ones((tensor.shape(update_indices)[0],), dtype=tensor.int32)
                
                # Update the recurrent mask
                nonlocal recurrent_mask
                recurrent_mask = tensor.tensor_scatter_nd_update(
                    recurrent_mask, 
                    update_indices, 
                    update_values
                )
        
        # Create connections between neuron groups
        # Sensory to inter connections
        if self.sensory_neurons > 0 and self.inter_neurons > 0:
            create_random_connections(
                sensory_start, sensory_end, 
                inter_start, inter_end, 
                self.sensory_to_inter_sparsity
            )
        
        # Sensory to command connections
        if self.sensory_neurons > 0 and self.command_neurons > 0:
            create_random_connections(
                sensory_start, sensory_end, 
                command_start, command_end, 
                self.sensory_to_inter_sparsity
            )
        
        # Inter to inter connections
        if self.inter_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                inter_start, inter_end, 
                self.inter_to_inter_sparsity
            )
        
        # Inter to command connections
        if self.inter_neurons > 0 and self.command_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                command_start, command_end, 
                self.inter_to_inter_sparsity
            )
        
        # Inter to motor connections
        if self.inter_neurons > 0 and self.motor_neurons > 0:
            create_random_connections(
                inter_start, inter_end, 
                motor_start, motor_end, 
                self.inter_to_motor_sparsity
            )
        
        # Command to motor connections
        if self.command_neurons > 0 and self.motor_neurons > 0:
            create_random_connections(
                command_start, command_end, 
                motor_start, motor_end, 
                self.inter_to_motor_sparsity
            )
        
        # Update communicability matrix based on recurrent mask
        # We need to use numpy temporarily for matrix exponential
        # This will be replaced with a pure tensor implementation when available
        recurrent_np = tensor.to_numpy(recurrent_mask)
        
        # Calculate communicability (using matrix exponential)
        from scipy.linalg import expm
        
        # Normalize by degree using tensor operations
        row_sums = stats.sum(recurrent_mask, axis=1)
        
        # Add small epsilon to avoid division by zero
        row_sums_eps = ops.add(row_sums, tensor.convert_to_tensor(1e-8))
        
        # Create diagonal matrices using ops.linearalg.diag
        from ember_ml.ops.linearalg import diag
        D_tensor = diag(row_sums_eps)
        
        # Calculate inverse square root
        sqrt_row_sums = ops.sqrt(row_sums_eps)
        inv_sqrt_row_sums = ops.divide(tensor.ones_like(sqrt_row_sums), sqrt_row_sums)
        D_sqrt_inv_tensor = diag(inv_sqrt_row_sums)
        
        # Convert to numpy for matrix exponential (until we have a tensor version)
        D_sqrt_inv = tensor.to_numpy(D_sqrt_inv_tensor)
        
        # Matrix multiplication
        normalized = D_sqrt_inv @ recurrent_np @ D_sqrt_inv
        
        # Apply matrix exponential
        comm_matrix_np = expm(normalized)
        
        # Convert back to tensor
        self.communicability_matrix = tensor.convert_to_tensor(comm_matrix_np)
        
        self._built = True
        return input_mask, recurrent_mask, output_mask
    
    def get_neuron_groups(self):
        """
        Get the indices of neurons in each group.
        
        Returns:
            Dictionary mapping group names to lists of neuron indices
        """
        # Define start/end indices
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = inter_start + self.inter_neurons
        command_start = inter_end
        command_end = command_start + self.command_neurons
        motor_start = command_end
        motor_end = self.units
        
        # Generate index lists
        sensory_idx = list(range(sensory_start, sensory_end))
        inter_idx = list(range(inter_start, inter_end))
        command_idx = list(range(command_start, command_end))
        motor_idx = list(range(motor_start, motor_end))
        
        return {
            "sensory": sensory_idx,
            "inter": inter_idx,
            "command": command_idx,
            "motor": motor_idx
        }
    
    def get_config(self):
        """
        Get the configuration of the NCP map.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons,
            "motor_neurons": self.motor_neurons,
            "sensory_neurons": self.sensory_neurons,
            "sensory_to_inter_sparsity": self.sensory_to_inter_sparsity,
            "sensory_to_motor_sparsity": self.sensory_to_motor_sparsity,
            "inter_to_inter_sparsity": self.inter_to_inter_sparsity,
            "inter_to_motor_sparsity": self.inter_to_motor_sparsity,
            "motor_to_motor_sparsity": self.motor_to_motor_sparsity,
            "motor_to_inter_sparsity": self.motor_to_inter_sparsity
        })
        return config