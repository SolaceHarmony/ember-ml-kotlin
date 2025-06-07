"""
Liquid Forecaster for Time Series Prediction

This module provides an implementation of a forecasting model using liquid neural networks
(CfC, LTC, ELTC, CTGRU, or CTRNN) for time series prediction with uncertainty estimation.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor
from ember_ml.nn.modules.rnn import CfC, LTC, ELTC, CTGRU, CTRNN
from ember_ml.nn.modules.activations import Softplus

class LiquidForecaster(Module):
    """
    Time series forecasting model using liquid neurons.
    
    This model uses liquid neural networks (CfC, LTC, ELTC, CTGRU, or CTRNN)
    to create a forecasting architecture for time series prediction with
    uncertainty estimation. The model can perform both single-step and
    multi-step forecasting.
    
    Features:
    - Time-aware processing with variable time steps
    - Multiple liquid neural network options
    - Multi-step forecasting
    - Uncertainty estimation
    - Support for multivariate time series
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_steps: int = 1,
        cell_type: str = 'cfc',
        num_layers: int = 2,
        backbone_units: int = 64,
        backbone_layers: int = 2,
        **kwargs
    ):
        """
        Initialize the liquid forecaster.
        
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden representation
            output_steps: Number of steps to forecast
            cell_type: Type of liquid neural network ('cfc', 'ltc', 'eltc', 'ctgru', 'ctrnn')
            num_layers: Number of layers in feature extractor
            backbone_units: Number of units in backbone networks
            backbone_layers: Number of layers in backbone networks
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_steps = output_steps
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
        
        # Create feature extractor
        self.feature_extractor = cell_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            return_sequences=True
        )
        
        # Create prediction heads
        self.forecast_head = Parameter(tensor.zeros((hidden_size, input_size * output_steps)))
        self.forecast_bias = Parameter(tensor.zeros((input_size * output_steps,)))
        
        self.uncertainty_head = Parameter(tensor.zeros((hidden_size, input_size * output_steps)))
        self.uncertainty_bias = Parameter(tensor.zeros((input_size * output_steps,)))
        
        # Activation for uncertainty
        self.softplus = Softplus()
    
    def forward(self, x, time_delta=None):
        """
        Forward pass through the forecaster.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            time_delta: Time steps between observations (optional)
            
        Returns:
            Tuple of (forecast, uncertainty) where:
            - forecast: Tensor of shape (batch_size, output_steps, input_size)
            - uncertainty: Tensor of shape (batch_size, output_steps, input_size)
        """
        # Extract features
        features = self.feature_extractor(x, time_delta=time_delta)
        
        # Use last state for prediction
        last_features = features[:, -1]
        
        # Generate predictions
        forecast = ops.add(
            ops.matmul(last_features, self.forecast_head),
            self.forecast_bias
        )
        
        # Generate uncertainty estimates
        uncertainty = ops.add(
            ops.matmul(last_features, self.uncertainty_head),
            self.uncertainty_bias
        )
        uncertainty = self.softplus(uncertainty)
        
        # Reshape for multi-step predictions
        batch_size = tensor.shape(x)[0]
        forecast = tensor.reshape(forecast, (batch_size, self.output_steps, self.input_size))
        uncertainty = tensor.reshape(uncertainty, (batch_size, self.output_steps, self.input_size))
        
        return forecast, uncertainty
    
    def compute_loss(self, x, y, time_delta=None):
        """
        Compute negative log likelihood loss with uncertainty.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            y: Target tensor of shape (batch_size, input_size)
            time_delta: Time steps between observations (optional)
            
        Returns:
            Loss value
        """
        # Get predictions and uncertainty estimates
        pred, uncertainty = self(x, time_delta=time_delta)
        
        # Use first prediction step
        first_step_pred = pred[:, 0]
        first_step_uncertainty = uncertainty[:, 0]
        
        # Compute squared error
        squared_error = ops.square(ops.subtract(first_step_pred, y))
        
        # Compute uncertainty term
        uncertainty_term = ops.log(ops.add(first_step_uncertainty, tensor.convert_to_tensor(1e-6)))
        
        # Compute negative log likelihood
        loss = ops.stats.mean(
            ops.add(
                ops.divide(
                    squared_error,
                    ops.multiply(2.0, ops.add(first_step_uncertainty, tensor.convert_to_tensor(1e-6)))
                ),
                uncertainty_term
            )
        )
        
        return loss
    
    def evaluate(self, x, y, time_delta=None):
        """
        Evaluate forecasting performance.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            y: Target tensor of shape (batch_size, input_size)
            time_delta: Time steps between observations (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions and uncertainty
        predictions, uncertainties = self(x, time_delta=time_delta)
        
        # Use first prediction step
        first_step_pred = predictions[:, 0]
        first_step_uncertainty = uncertainties[:, 0]
        
        # Compute metrics
        mse = ops.stats.mean(ops.square(ops.subtract(first_step_pred, y)))
        mae = ops.stats.mean(ops.abs(ops.subtract(first_step_pred, y)))
        
        # Compute calibration (percentage of true values within uncertainty bounds)
        std = ops.sqrt(first_step_uncertainty)
        lower_bound = ops.subtract(first_step_pred, ops.multiply(2.0, std))
        upper_bound = ops.add(first_step_pred, ops.multiply(2.0, std))
        
        in_bounds = ops.logical_and(
            ops.greater_equal(y, lower_bound),
            ops.less_equal(y, upper_bound)
        )
        calibration = ops.stats.mean(tensor.cast(in_bounds, tensor.float32))
        
        return {
            'mse': mse,
            'mae': mae,
            'calibration': calibration
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the forecaster."""
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_steps": self.output_steps,
            "cell_type": self.cell_type,
            "num_layers": self.num_layers,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LiquidForecaster':
        """Creates a forecaster from its configuration."""
        return cls(**config)