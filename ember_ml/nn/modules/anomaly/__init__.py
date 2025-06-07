"""
Anomaly detection modules for Ember ML.

This module provides various anomaly detection models and techniques,
including reconstruction-based approaches using liquid neural networks.
"""

from ember_ml.nn.modules.anomaly.liquid_autoencoder import LiquidAutoencoder

__all__ = [
    "LiquidAutoencoder",
]