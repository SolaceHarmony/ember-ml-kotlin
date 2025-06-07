"""
Neuron Maps for Ember ML.

This module provides neuron map implementations for defining
connectivity patterns in neural networks.
"""

# Import base classes
from ember_ml.nn.modules.wiring.neuron_map import NeuronMap
from ember_ml.nn.modules.wiring.enhanced_neuron_map import EnhancedNeuronMap

# Import specific implementations
from ember_ml.nn.modules.wiring.ncp_map import NCPMap
from ember_ml.nn.modules.wiring.enhanced_ncp_map import EnhancedNCPMap
from ember_ml.nn.modules.wiring.fully_connected_map import FullyConnectedMap
from ember_ml.nn.modules.wiring.random_map import RandomMap
from ember_ml.nn.modules.wiring.frequency_wiring import FrequencyWiring
from ember_ml.nn.modules.wiring.fully_connected_ncp_map import FullyConnectedNCPMap
from ember_ml.nn.modules.wiring.guce_ncp_map import GUCENCPMap
from ember_ml.nn.modules.wiring.guce_ncp import GUCENCP
from ember_ml.nn.modules.wiring.language_wiring import LanguageWiring
from ember_ml.nn.modules.wiring.robotics_wiring import RoboticsWiring
from ember_ml.nn.modules.wiring.signal_wiring import SignalWiring
from ember_ml.nn.modules.wiring.vision_wiring import VisionWiring


__all__ = [
    # Base classes
    "NeuronMap",
    "EnhancedNeuronMap",

    # Specific implementations
    "NCPMap",
    "EnhancedNCPMap",
    "FullyConnectedMap",
    "RandomMap",
    "FrequencyWiring",
    "FullyConnectedNCPMap",
    "GUCENCPMap",
    "GUCENCP",
    "LanguageWiring",
    "RoboticsWiring",
    "SignalWiring",
    "VisionWiring",
]