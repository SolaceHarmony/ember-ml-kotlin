"""
Wave models module.

This module provides models for wave-based neural processing,
including wave transformers, wave RNNs, wave autoencoders, and multi-sphere models.
"""

from ember_ml.wave.models.wave_transformer import *
from ember_ml.wave.models.wave_rnn import *
from ember_ml.wave.models.wave_autoencoder import *
from ember_ml.wave.models.multi_sphere import *

__all__ = [
    'wave_transformer',
    'wave_rnn',
    'wave_autoencoder',
    'multi_sphere',
]