"""Wave utilities module for signal processing operations.

This module provides backend-agnostic implementations of wave processing utilities,
including conversion, analysis, and visualization functions.

Components:
    wave_conversion: Tools for converting between different wave representations
    wave_analysis: Signal analysis functions using ops abstraction
    wave_visualization: Visualization utilities for wave data

All functions maintain backend independence through the ops abstraction layer.
"""

from ember_ml.wave.utils.wave_conversion import *
from ember_ml.wave.utils.wave_analysis import *
from ember_ml.wave.utils.wave_visualization import *

__all__ = [
    'wave_conversion',
    'wave_analysis',
    'wave_visualization',
]