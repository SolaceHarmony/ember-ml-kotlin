"""
Multi-Sphere model for wave-based neural processing.

This module provides a multi-sphere model for wave-based neural processing,
which represents data points on multiple hyperspheres for enhanced representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from ember_ml.nn import tensor # Added import
from ember_ml.nn.tensor.types import TensorLike # Added import
import math

class SphereProjection(nn.Module):
    """
    Projects data onto a hypersphere.
    """
    
    def __init__(self, input_dim: int, sphere_dim: int, radius: float = 1.0):
        """
        Initialize the sphere projection.
        
        Args:
            input_dim: Input dimension
            sphere_dim: Sphere dimension
            radius: Sphere radius
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.sphere_dim = sphere_dim
        self.radius = radius
        
        # Projection layer
        self.projection = nn.Linear(input_dim, sphere_dim)
        
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Projected tensor of shape (batch_size, sphere_dim)
        """
        # Project to sphere dimension
        projected = self.projection(x)
        
        # Normalize to sphere
        norm = torch.norm(projected, p=2, dim=1, keepdim=True)
        normalized = projected / (norm + 1e-8)
        
        # Scale by radius
        return normalized * self.radius

class MultiSphereProjection(nn.Module):
    """
    Projects data onto multiple hyperspheres.
    """
    
    def __init__(self, input_dim: int, sphere_dims: List[int], radii: List[float] = None):
        """
        Initialize the multi-sphere projection.
        
        Args:
            input_dim: Input dimension
            sphere_dims: List of sphere dimensions
            radii: List of sphere radii
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.sphere_dims = sphere_dims
        
        if radii is None:
            radii = [1.0] * len(sphere_dims)
        
        assert len(sphere_dims) == len(radii), "Number of sphere dimensions must match number of radii"
        
        self.radii = radii
        
        # Create sphere projections
        self.projections = nn.ModuleList([
            SphereProjection(input_dim, sphere_dim, radius)
            for sphere_dim, radius in zip(sphere_dims, radii)
        ])
        
    def forward(self, x: tensor.convert_to_tensor) -> List[tensor.convert_to_tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of projected tensors, each of shape (batch_size, sphere_dim_i)
        """
        return [projection(x) for projection in self.projections]

class MultiSphereEncoder(nn.Module):
    """
    Encodes data onto multiple hyperspheres.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], sphere_dims: List[int], 
                 radii: List[float] = None, activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.1):
        """
        Initialize the multi-sphere encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            sphere_dims: List of sphere dimensions
            radii: List of sphere radii
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sphere_dims = sphere_dims
        
        if radii is None:
            radii = [1.0] * len(sphere_dims)
        
        self.radii = radii
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Multi-sphere projection
        self.projection = MultiSphereProjection(prev_dim, sphere_dims, radii)
        
    def forward(self, x: tensor.convert_to_tensor) -> List[tensor.convert_to_tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of projected tensors, each of shape (batch_size, sphere_dim_i)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Project to multiple spheres
        return self.projection(encoded)

class MultiSphereDecoder(nn.Module):
    """
    Decodes data from multiple hyperspheres.
    """
    
    def __init__(self, sphere_dims: List[int], hidden_dims: List[int], output_dim: int, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the multi-sphere decoder.
        
        Args:
            sphere_dims: List of sphere dimensions
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.sphere_dims = sphere_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Calculate total sphere dimension
        self.total_sphere_dim = sum(sphere_dims)
        
        # Build decoder layers
        layers = []
        prev_dim = self.total_sphere_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, spheres: List[tensor.convert_to_tensor]) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            spheres: List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Concatenate sphere representations
        x = torch.cat(spheres, dim=1)
        
        # Decode
        return self.decoder(x)

class MultiSphereModel(nn.Module):
    """
    Multi-sphere model for wave-based neural processing.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], sphere_dims: List[int], 
                 output_dim: int, radii: List[float] = None, activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.1):
        """
        Initialize the multi-sphere model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            sphere_dims: List of sphere dimensions
            output_dim: Output dimension
            radii: List of sphere radii
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sphere_dims = sphere_dims
        self.output_dim = output_dim
        
        if radii is None:
            radii = [1.0] * len(sphere_dims)
        
        self.radii = radii
        
        # Encoder
        self.encoder = MultiSphereEncoder(input_dim, hidden_dims, sphere_dims, radii, activation, dropout)
        
        # Decoder
        self.decoder = MultiSphereDecoder(sphere_dims, hidden_dims[::-1], output_dim, activation, dropout)
        
    def forward(self, x: tensor.convert_to_tensor) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
            - 'output': Output tensor of shape (batch_size, output_dim)
            - 'spheres': List of sphere tensors, each of shape (batch_size, sphere_dim_i)
        """
        # Encode to multiple spheres
        spheres = self.encoder(x)
        
        # Decode from multiple spheres
        output = self.decoder(spheres)
        
        return {
            'output': output,
            'spheres': spheres
        }
    
    def encode(self, x: tensor.convert_to_tensor) -> List[tensor.convert_to_tensor]:
        """
        Encode input to multiple spheres.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            List of sphere tensors, each of shape (batch_size, sphere_dim_i)
        """
        return self.encoder(x)
    
    def decode(self, spheres: List[tensor.convert_to_tensor]) -> tensor.convert_to_tensor:
        """
        Decode from multiple spheres.
        
        Args:
            spheres: List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.decoder(spheres)

class SphericalHarmonics(nn.Module):
    """
    Spherical harmonics layer for multi-sphere models.
    """
    
    def __init__(self, sphere_dim: int, max_degree: int):
        """
        Initialize the spherical harmonics layer.
        
        Args:
            sphere_dim: Sphere dimension
            max_degree: Maximum degree of spherical harmonics
        """
        super().__init__()
        
        self.sphere_dim = sphere_dim
        self.max_degree = max_degree
        
        # Calculate number of harmonics
        self.num_harmonics = self._calculate_num_harmonics()
        
        # Weights for harmonics
        self.weights = nn.Parameter(tensor.convert_to_tensor(self.num_harmonics))
        self.reset_parameters()
        
    def _calculate_num_harmonics(self) -> int:
        """
        Calculate the number of spherical harmonics.
        
        Returns:
            Number of harmonics
        """
        # For S^n, the number of linearly independent spherical harmonics of degree l is
        # (2l + n - 1) * (l + n - 2)! / (l! * (n - 1)!)
        n = self.sphere_dim - 1  # Dimension of the sphere
        
        num_harmonics = 0
        for l in range(self.max_degree + 1):
            num_l = (2 * l + n - 1) * math.factorial(l + n - 2) // (math.factorial(l) * math.factorial(n - 1))
            num_harmonics += num_l
        
        return num_harmonics
    
    def reset_parameters(self):
        """
        Reset parameters.
        """
        nn.init.normal_(self.weights, mean=0.0, std=0.1)
    
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sphere_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_harmonics)
        """
        # This is a simplified implementation
        # In practice, computing spherical harmonics is more complex
        # and would involve specialized libraries
        
        # Normalize input to unit sphere
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Compute powers of coordinates as a simple approximation
        harmonics = []
        
        for degree in range(self.max_degree + 1):
            # For each degree, compute all possible products of coordinates
            # that have total degree equal to 'degree'
            
            # This is a simplified version
            if degree == 0:
                # Constant term
                harmonics.append(torch.ones(x.size(0), 1, device=x.device))
            else:
                # Powers of each coordinate
                for i in range(self.sphere_dim):
                    harmonics.append(x_norm[:, i:i+1] ** degree)
        
        # Concatenate all harmonics
        all_harmonics = torch.cat(harmonics, dim=1)
        
        # Apply weights
        # In practice, this would be more complex
        return all_harmonics * self.weights[:all_harmonics.size(1)]

class MultiSphereHarmonicModel(nn.Module):
    """
    Multi-sphere model with spherical harmonics for wave-based neural processing.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], sphere_dims: List[int], 
                 max_degrees: List[int], output_dim: int, radii: List[float] = None, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the multi-sphere harmonic model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            sphere_dims: List of sphere dimensions
            max_degrees: List of maximum degrees for spherical harmonics
            output_dim: Output dimension
            radii: List of sphere radii
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sphere_dims = sphere_dims
        self.max_degrees = max_degrees
        self.output_dim = output_dim
        
        assert len(sphere_dims) == len(max_degrees), "Number of sphere dimensions must match number of max degrees"
        
        if radii is None:
            radii = [1.0] * len(sphere_dims)
        
        self.radii = radii
        
        # Encoder
        self.encoder = MultiSphereEncoder(input_dim, hidden_dims, sphere_dims, radii, activation, dropout)
        
        # Spherical harmonics layers
        self.harmonic_layers = nn.ModuleList([
            SphericalHarmonics(sphere_dim, max_degree)
            for sphere_dim, max_degree in zip(sphere_dims, max_degrees)
        ])
        
        # Calculate total harmonic dimension
        self.harmonic_dims = [layer.num_harmonics for layer in self.harmonic_layers]
        self.total_harmonic_dim = sum(self.harmonic_dims)
        
        # Decoder
        decoder_layers = []
        prev_dim = self.total_harmonic_dim
        
        for hidden_dim in hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(activation)
            
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: tensor.convert_to_tensor) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
            - 'output': Output tensor of shape (batch_size, output_dim)
            - 'spheres': List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            - 'harmonics': List of harmonic tensors, each of shape (batch_size, harmonic_dim_i)
        """
        # Encode to multiple spheres
        spheres = self.encoder(x)
        
        # Compute spherical harmonics
        harmonics = [layer(sphere) for layer, sphere in zip(self.harmonic_layers, spheres)]
        
        # Concatenate harmonics
        all_harmonics = torch.cat(harmonics, dim=1)
        
        # Decode
        output = self.decoder(all_harmonics)
        
        return {
            'output': output,
            'spheres': spheres,
            'harmonics': harmonics
        }

class MultiSphereWaveModel(nn.Module):
    """
    Multi-sphere model for wave-based neural processing with wave-specific features.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], sphere_dims: List[int], 
                 output_dim: int, sequence_length: int, radii: List[float] = None, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the multi-sphere wave model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            sphere_dims: List of sphere dimensions
            output_dim: Output dimension
            sequence_length: Length of wave sequence
            radii: List of sphere radii
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sphere_dims = sphere_dims
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        if radii is None:
            radii = [1.0] * len(sphere_dims)
        
        self.radii = radii
        
        # Temporal convolution for wave processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1),
            activation,
            nn.Conv1d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            activation
        )
        
        # Multi-sphere encoder
        self.sphere_encoder = MultiSphereEncoder(hidden_dims[0], hidden_dims[1:], sphere_dims, radii, activation, dropout)
        
        # Wave-specific processing on spheres
        self.sphere_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sphere_dim, sphere_dim),
                activation,
                nn.Linear(sphere_dim, sphere_dim)
            )
            for sphere_dim in sphere_dims
        ])
        
        # Decoder
        self.decoder = MultiSphereDecoder(sphere_dims, hidden_dims[::-1], output_dim, activation, dropout)
        
        # Sequence reconstruction
        self.sequence_reconstructor = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], input_dim * sequence_length)
        )
        
    def forward(self, x: tensor.convert_to_tensor) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Dictionary containing:
            - 'output': Output tensor of shape (batch_size, output_dim)
            - 'spheres': List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            - 'reconstructed_sequence': Reconstructed sequence tensor of shape (batch_size, sequence_length, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape for temporal convolution
        x_reshaped = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        
        # Apply temporal convolution
        conv_out = self.temporal_conv(x_reshaped)
        
        # Global average pooling
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
        
        # Encode to multiple spheres
        spheres = self.sphere_encoder(pooled)
        
        # Apply wave-specific processing on spheres
        processed_spheres = [processor(sphere) for processor, sphere in zip(self.sphere_processors, spheres)]
        
        # Decode from multiple spheres
        output = self.decoder(processed_spheres)
        
        # Reconstruct sequence
        seq_flat = self.sequence_reconstructor(output)
        reconstructed_sequence = seq_flat.view(batch_size, seq_len, -1)
        
        return {
            'output': output,
            'spheres': spheres,
            'processed_spheres': processed_spheres,
            'reconstructed_sequence': reconstructed_sequence
        }
    
    def encode(self, x: tensor.convert_to_tensor) -> List[tensor.convert_to_tensor]:
        """
        Encode input to multiple spheres.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            List of sphere tensors, each of shape (batch_size, sphere_dim_i)
        """
        # Reshape for temporal convolution
        x_reshaped = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        
        # Apply temporal convolution
        conv_out = self.temporal_conv(x_reshaped)
        
        # Global average pooling
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
        
        # Encode to multiple spheres
        return self.sphere_encoder(pooled)
    
    def process_spheres(self, spheres: List[tensor.convert_to_tensor]) -> List[tensor.convert_to_tensor]:
        """
        Apply wave-specific processing on spheres.
        
        Args:
            spheres: List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            
        Returns:
            List of processed sphere tensors, each of shape (batch_size, sphere_dim_i)
        """
        return [processor(sphere) for processor, sphere in zip(self.sphere_processors, spheres)]
    
    def decode(self, spheres: List[tensor.convert_to_tensor]) -> tensor.convert_to_tensor:
        """
        Decode from multiple spheres.
        
        Args:
            spheres: List of sphere tensors, each of shape (batch_size, sphere_dim_i)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.decoder(spheres)
    
    def reconstruct_sequence(self, output: tensor.convert_to_tensor, batch_size: int) -> tensor.convert_to_tensor:
        """
        Reconstruct sequence from output.
        
        Args:
            output: Output tensor of shape (batch_size, output_dim)
            batch_size: Batch size
            
        Returns:
            Reconstructed sequence tensor of shape (batch_size, sequence_length, input_dim)
        """
        seq_flat = self.sequence_reconstructor(output)
        return seq_flat.view(batch_size, self.sequence_length, -1)

# Convenience function to create a multi-sphere model
def create_multi_sphere_model(input_dim: int, 
                             hidden_dims: List[int], 
                             sphere_dims: List[int], 
                             output_dim: int, 
                             radii: List[float] = None, 
                             activation: str = 'relu', 
                             dropout: float = 0.1) -> MultiSphereModel:
    """
    Create a multi-sphere model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        sphere_dims: List of sphere dimensions
        output_dim: Output dimension
        radii: List of sphere radii
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability
        
    Returns:
        Multi-sphere model
    """
    # Get activation function
    if activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    elif activation == 'sigmoid':
        act_fn = nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    return MultiSphereModel(input_dim, hidden_dims, sphere_dims, output_dim, radii, act_fn, dropout)

# Convenience function to create a multi-sphere harmonic model
def create_multi_sphere_harmonic_model(input_dim: int, 
                                      hidden_dims: List[int], 
                                      sphere_dims: List[int], 
                                      max_degrees: List[int],
                                      output_dim: int, 
                                      radii: List[float] = None, 
                                      activation: str = 'relu', 
                                      dropout: float = 0.1) -> MultiSphereHarmonicModel:
    """
    Create a multi-sphere harmonic model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        sphere_dims: List of sphere dimensions
        max_degrees: List of maximum degrees for spherical harmonics
        output_dim: Output dimension
        radii: List of sphere radii
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability
        
    Returns:
        Multi-sphere harmonic model
    """
    # Get activation function
    if activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    elif activation == 'sigmoid':
        act_fn = nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    return MultiSphereHarmonicModel(input_dim, hidden_dims, sphere_dims, max_degrees, output_dim, radii, act_fn, dropout)

# Convenience function to create a multi-sphere wave model
def create_multi_sphere_wave_model(input_dim: int, 
                                  hidden_dims: List[int], 
                                  sphere_dims: List[int], 
                                  output_dim: int, 
                                  sequence_length: int,
                                  radii: List[float] = None, 
                                  activation: str = 'relu', 
                                  dropout: float = 0.1) -> MultiSphereWaveModel:
    """
    Create a multi-sphere wave model.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        sphere_dims: List of sphere dimensions
        output_dim: Output dimension
        sequence_length: Length of wave sequence
        radii: List of sphere radii
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability
        
    Returns:
        Multi-sphere wave model
    """
    # Get activation function
    if activation == 'relu':
        act_fn = nn.ReLU()
    elif activation == 'tanh':
        act_fn = nn.Tanh()
    elif activation == 'sigmoid':
        act_fn = nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    return MultiSphereWaveModel(input_dim, hidden_dims, sphere_dims, output_dim, sequence_length, radii, act_fn, dropout)