"""
Wave Autoencoder model.

This module provides an autoencoder model for wave-based neural processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from ember_ml.nn import tensor # Added import
from ember_ml.nn.tensor.types import TensorLike # Added import

class WaveEncoder(nn.Module):
    """
    Encoder for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], latent_size: int, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the wave encoder.
        
        Args:
            input_size: Input dimension
            hidden_sizes: List of hidden layer dimensions
            latent_size: Latent dimension
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        
        # Build encoder layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, latent_size))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: TensorLike) -> TensorLike:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Latent representation tensor of shape (batch_size, latent_size)
        """
        return self.encoder(x)

class WaveDecoder(nn.Module):
    """
    Decoder for wave-based neural processing.
    """
    
    def __init__(self, latent_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the wave decoder.
        
        Args:
            latent_size: Latent dimension
            hidden_sizes: List of hidden layer dimensions
            output_size: Output dimension
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build decoder layers
        layers = []
        prev_size = latent_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: TensorLike) -> TensorLike:
        """
        Forward pass.
        
        Args:
            z: Latent representation tensor of shape (batch_size, latent_size)
            
        Returns:
            Reconstructed output tensor of shape (batch_size, output_size)
        """
        return self.decoder(z)

class WaveVariationalEncoder(nn.Module):
    """
    Variational encoder for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], latent_size: int, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1):
        """
        Initialize the wave variational encoder.
        
        Args:
            input_size: Input dimension
            hidden_sizes: List of hidden layer dimensions
            latent_size: Latent dimension
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        
        # Build encoder layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = hidden_size
        
        # Mean and log variance layers
        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_size, latent_size)
        self.log_var = nn.Linear(prev_size, latent_size)
        
    def forward(self, x: TensorLike) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (z, mean, log_var)
            - z: Sampled latent representation tensor of shape (batch_size, latent_size)
            - mean: Mean tensor of shape (batch_size, latent_size)
            - log_var: Log variance tensor of shape (batch_size, latent_size)
        """
        # Encode
        h = self.encoder(x)
        
        # Get mean and log variance
        mean = self.mean(h)
        log_var = self.log_var(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, log_var

class WaveAutoencoder(nn.Module):
    """
    Autoencoder for wave-based neural processing.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], latent_size: int, 
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.1, variational: bool = False):
        """
        Initialize the wave autoencoder.
        
        Args:
            input_size: Input dimension
            hidden_sizes: List of hidden layer dimensions
            latent_size: Latent dimension
            activation: Activation function
            dropout: Dropout probability
            variational: Whether to use variational autoencoder
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.variational = variational
        
        # Encoder
        if variational:
            self.encoder = WaveVariationalEncoder(input_size, hidden_sizes, latent_size, activation, dropout)
        else:
            self.encoder = WaveEncoder(input_size, hidden_sizes, latent_size, activation, dropout)
        
        # Decoder
        self.decoder = WaveDecoder(latent_size, hidden_sizes[::-1], input_size, activation, dropout)
        
    def forward(self, x: TensorLike) -> Dict[str, TensorLike]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Dictionary containing:
            - 'reconstruction': Reconstructed output tensor of shape (batch_size, input_size)
            - 'latent': Latent representation tensor of shape (batch_size, latent_size)
            - 'mean': Mean tensor of shape (batch_size, latent_size) (only for variational)
            - 'log_var': Log variance tensor of shape (batch_size, latent_size) (only for variational)
        """
        if self.variational:
            z, mean, log_var = self.encoder(x)
            reconstruction = self.decoder(z)
            
            return {
                'reconstruction': reconstruction,
                'latent': z,
                'mean': mean,
                'log_var': log_var
            }
        else:
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            
            return {
                'reconstruction': reconstruction,
                'latent': z
            }
    
    def encode(self, x: TensorLike) -> TensorLike:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Latent representation tensor of shape (batch_size, latent_size)
        """
        if self.variational:
            z, _, _ = self.encoder(x)
            return z
        else:
            return self.encoder(x)
    
    def decode(self, z: TensorLike) -> TensorLike:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation tensor of shape (batch_size, latent_size)
            
        Returns:
            Reconstructed output tensor of shape (batch_size, input_size)
        """
        return self.decoder(z)

class WaveConvolutionalAutoencoder(nn.Module):
    """
    Convolutional autoencoder for wave-based neural processing.
    """
    
    def __init__(self, input_channels: int, input_size: int, hidden_channels: List[int], 
                 kernel_sizes: List[int], latent_size: int, activation: nn.Module = nn.ReLU(), 
                 dropout: float = 0.1, variational: bool = False):
        """
        Initialize the wave convolutional autoencoder.
        
        Args:
            input_channels: Number of input channels
            input_size: Input size
            hidden_channels: List of hidden channel dimensions
            kernel_sizes: List of kernel sizes
            latent_size: Latent dimension
            activation: Activation function
            dropout: Dropout probability
            variational: Whether to use variational autoencoder
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.latent_size = latent_size
        self.variational = variational
        
        # Calculate output size after convolutions
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Encoder convolutional layers
        self.encoder_conv = self._build_encoder_conv()
        
        # Encoder fully connected layers
        if variational:
            self.encoder_mean = nn.Linear(self.conv_output_size, latent_size)
            self.encoder_log_var = nn.Linear(self.conv_output_size, latent_size)
        else:
            self.encoder_fc = nn.Linear(self.conv_output_size, latent_size)
        
        # Decoder fully connected layers
        self.decoder_fc = nn.Linear(latent_size, self.conv_output_size)
        
        # Decoder convolutional layers
        self.decoder_conv = self._build_decoder_conv()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def _calculate_conv_output_size(self) -> int:
        """
        Calculate the output size after convolutions.
        
        Returns:
            Output size
        """
        size = self.input_size
        channels = self.input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(self.hidden_channels, self.kernel_sizes)):
            # Convolution
            size = (size - kernel_size + 1) // 2  # With stride=2
            channels = out_channels
        
        return size * size * channels
    
    def _build_encoder_conv(self) -> nn.Sequential:
        """
        Build encoder convolutional layers.
        
        Returns:
            Sequential module of encoder convolutional layers
        """
        layers = []
        in_channels = self.input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(self.hidden_channels, self.kernel_sizes)):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=2))
            layers.append(self.activation)
            
            if self.dropout is not None:
                layers.append(self.dropout)
                
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_decoder_conv(self) -> nn.Sequential:
        """
        Build decoder convolutional layers.
        
        Returns:
            Sequential module of decoder convolutional layers
        """
        layers = []
        in_channels = self.hidden_channels[-1]
        
        for i in range(len(self.hidden_channels) - 1, 0, -1):
            out_channels = self.hidden_channels[i - 1]
            kernel_size = self.kernel_sizes[i]
            
            layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2))
            layers.append(self.activation)
            
            if self.dropout is not None:
                layers.append(self.dropout)
                
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.ConvTranspose1d(in_channels, self.input_channels, self.kernel_sizes[0], stride=2))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: TensorLike) -> TensorLike:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, input_size)
            
        Returns:
            Latent representation tensor of shape (batch_size, latent_size)
        """
        batch_size = x.size(0)
        
        # Convolutional encoding
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        
        # Fully connected encoding
        if self.variational:
            mean = self.encoder_mean(x)
            log_var = self.encoder_log_var(x)
            
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
            
            return z
        else:
            return self.encoder_fc(x)
    
    def decode(self, z: TensorLike) -> TensorLike:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation tensor of shape (batch_size, latent_size)
            
        Returns:
            Reconstructed output tensor of shape (batch_size, input_channels, input_size)
        """
        batch_size = z.size(0)
        
        # Fully connected decoding
        x = self.decoder_fc(z)
        
        # Reshape for convolutional decoding
        x = x.view(batch_size, self.hidden_channels[-1], -1)
        
        # Convolutional decoding
        x = self.decoder_conv(x)
        
        return x
    
    def forward(self, x: TensorLike) -> Dict[str, TensorLike]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, input_size)
            
        Returns:
            Dictionary containing:
            - 'reconstruction': Reconstructed output tensor of shape (batch_size, input_channels, input_size)
            - 'latent': Latent representation tensor of shape (batch_size, latent_size)
            - 'mean': Mean tensor of shape (batch_size, latent_size) (only for variational)
            - 'log_var': Log variance tensor of shape (batch_size, latent_size) (only for variational)
        """
        batch_size = x.size(0)
        
        # Convolutional encoding
        conv_encoded = self.encoder_conv(x)
        conv_encoded_flat = conv_encoded.view(batch_size, -1)
        
        # Fully connected encoding
        if self.variational:
            mean = self.encoder_mean(conv_encoded_flat)
            log_var = self.encoder_log_var(conv_encoded_flat)
            
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = self.encoder_fc(conv_encoded_flat)
            mean = None
            log_var = None
        
        # Fully connected decoding
        decoded_flat = self.decoder_fc(z)
        
        # Reshape for convolutional decoding
        decoded_conv = decoded_flat.view(batch_size, self.hidden_channels[-1], -1)
        
        # Convolutional decoding
        reconstruction = self.decoder_conv(decoded_conv)
        
        result = {
            'reconstruction': reconstruction,
            'latent': z
        }
        
        if self.variational:
            result['mean'] = mean
            result['log_var'] = log_var
        
        return result

# Convenience function to create a wave autoencoder
def create_wave_autoencoder(input_size: int, 
                           hidden_sizes: List[int], 
                           latent_size: int, 
                           activation: str = 'relu', 
                           dropout: float = 0.1, 
                           variational: bool = False) -> WaveAutoencoder:
    """
    Create a wave autoencoder.
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer dimensions
        latent_size: Latent dimension
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability
        variational: Whether to use variational autoencoder
        
    Returns:
        Wave autoencoder model
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
    
    return WaveAutoencoder(input_size, hidden_sizes, latent_size, act_fn, dropout, variational)

# Convenience function to create a wave convolutional autoencoder
def create_wave_conv_autoencoder(input_channels: int, 
                                input_size: int, 
                                hidden_channels: List[int], 
                                kernel_sizes: List[int], 
                                latent_size: int, 
                                activation: str = 'relu', 
                                dropout: float = 0.1, 
                                variational: bool = False) -> WaveConvolutionalAutoencoder:
    """
    Create a wave convolutional autoencoder.
    
    Args:
        input_channels: Number of input channels
        input_size: Input size
        hidden_channels: List of hidden channel dimensions
        kernel_sizes: List of kernel sizes
        latent_size: Latent dimension
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability
        variational: Whether to use variational autoencoder
        
    Returns:
        Wave convolutional autoencoder model
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
    
    return WaveConvolutionalAutoencoder(input_channels, input_size, hidden_channels, 
                                       kernel_sizes, latent_size, act_fn, dropout, variational)