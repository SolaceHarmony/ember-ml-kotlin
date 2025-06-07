"""Neural modulation and neuromodulatory system implementations.

This module provides backend-agnostic implementations of neural modulation,
including dopamine modulation and other neuromodulatory effects.

Components:
    DopamineModulator: Implements dopamine-based modulation
        - Time-dependent modulation of neural activity
        - State-dependent learning rate adjustment
        - Reward signal processing
        
    DopamineState: Represents dopamine system state
        - Tracks dopamine levels
        - Maintains temporal dynamics
        - Handles state transitions

All implementations use the ops abstraction layer for computations.
"""

from ember_ml.nn.modulation.dopamine import DopamineModulator, DopamineState

__all__ = [
    "DopamineModulator",
    "DopamineState",
]
