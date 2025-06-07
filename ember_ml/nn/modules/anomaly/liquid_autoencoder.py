"""
Liquid Autoencoder for Anomaly Detection

This module provides an implementation of an autoencoder using liquid neural networks
(CfC and LTC) for time series anomaly detection.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn import CfC, LTC, ELTC, CTGRU, CTRNN

class LiquidAutoencoder(Module):
    """
    Autoencoder using liquid neurons for anomaly detection.
    
    This model uses liquid neural networks (CfC, LTC, ELTC, CTGRU, or CTRNN)
    to create an autoencoder architecture for time series anomaly detection.
    The model reconstructs the input sequence and uses the reconstruction error
    as an anomaly score.
    
    Features:
    - Time-aware processing with variable time steps
    - Multiple liquid neural network options
    - Reconstruction-based anomaly scoring
    - Support for multivariate time series
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_type: str = 'cfc',
        num_layers: int = 2,
        backbone_units: int = 64,
        backbone_layers: int = 2,
        **kwargs
    ):
        """
        Initialize the liquid autoencoder.
        
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden representation
            cell_type: Type of liquid neural network ('cfc', 'ltc', 'eltc', 'ctgru', 'ctrnn')
            num_layers: Number of layers in encoder and decoder
            backbone_units: Number of units in backbone networks
            backbone_layers: Number of layers in backbone networks
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        
        # Select cell type
        if cell_type == 'cfc':
            cell_class = CfC
        elif cell_type == 'ltc':
            cell_class = LTC
        elif cell_type == 'eltc':
            cell_class = ELTC
        elif cell_type == 'ctgru':
            cell_class = CTGRU
        elif cell_type == 'ctrnn':
            cell_class = CTRNN
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        
        # Create encoder
        self.encoder = cell_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            return_sequences=True
        )
        
        # Create decoder
        self.decoder = cell_class(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            return_sequences=True
        )
    
    def forward(self, x, time_delta=None):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            time_delta: Time steps between observations (optional)
            
        Returns:
            Reconstructed tensor of shape (batch_size, seq_length, input_size)
        """
        # Encode
        encoded = self.encoder(x, time_delta=time_delta)
        
        # Decode
        decoded = self.decoder(encoded, time_delta=time_delta)
        
        return decoded
    
    def compute_anomaly_score(self, x, time_delta=None):
        """
        Compute reconstruction error as anomaly score.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            time_delta: Time steps between observations (optional)
            
        Returns:
            Anomaly scores of shape (batch_size, seq_length)
        """
        # Get reconstruction
        reconstructed = self(x, time_delta=time_delta)
        
        # Compute squared error
        error = ops.square(ops.subtract(x, reconstructed))
        
        # Use time-weighted MSE if time_delta is provided
        if time_delta is not None:
            # Avoid division by zero
            weights = ops.divide(
                tensor.ones_like(time_delta),
                ops.add(time_delta, tensor.convert_to_tensor(1e-6))
            )
            # Weight errors by inverse time delta
            weighted_error = ops.multiply(error, weights)
            return ops.stats.mean(weighted_error, axis=-1)
        
        # Regular MSE
        return ops.stats.mean(error, axis=-1)
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the autoencoder."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "cell_type": self.cell_type,
            "num_layers": self.num_layers,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LiquidAutoencoder':
        """Creates an autoencoder from its configuration."""
        return cls(**config)