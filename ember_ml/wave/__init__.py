"""
Wave-based neural processing module.

This module provides implementations of wave-based neural processing,
including binary wave neurons, harmonic embeddings, and wave memory analysis.
"""

# Define the classes that will be imported by the harmonic module
class HarmonicProcessor:
    """Harmonic processor for wave-based neural processing."""
    pass

class FrequencyAnalyzer:
    """Frequency analyzer for wave-based neural processing."""
    pass

class WaveSynthesizer:
    """Wave synthesizer for wave-based neural processing."""
    pass

# Import the rest of the modules
from ember_ml.wave.binary import *
from ember_ml.wave.memory import *
from ember_ml.wave.audio import *
from ember_ml.wave.limb import *

# Import the harmonic module itself
import ember_ml.wave.harmonic

__all__ = [
    'binary',
    'harmonic',
    'memory',
    'audio',
    'limb',
    'HarmonicProcessor',
    'FrequencyAnalyzer',
    'WaveSynthesizer',
]
