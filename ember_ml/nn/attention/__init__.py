"""
Specialized neurons module.

This module provides implementations of specialized neurons,
including attention neurons and base neurons.
"""

# Import classes now located within this package
from ember_ml.nn.attention.base import BaseAttention, AttentionLayer, MultiHeadAttention, AttentionMask, AttentionScore
from ember_ml.nn.attention.causal import CausalAttention, PredictionAttention, AttentionState # CausalAttention moved here
from ember_ml.nn.attention.temporal import TemporalAttention, PositionalEncoding
from ember_ml.nn.attention.mechanisms import CausalAttention as MechanismCausalAttention # Import mechanism if different
from ember_ml.nn.attention.attention import LTCNeuronWithAttention # Keep this if it's still relevant

__all__ = [
    # Core Attention Classes
    "BaseAttention",
    "AttentionLayer",
    "MultiHeadAttention",
    # Specific Implementations
    "CausalAttention",
    "PredictionAttention",
    "TemporalAttention",
    # Utilities / Supporting classes
    "AttentionMask",
    "AttentionScore",
    "AttentionState",
    "PositionalEncoding",
    # Specialized Neuron (If still relevant)
    "LTCNeuronWithAttention",
    # Note: MechanismCausalAttention might be redundant if same as models.attention.causal.CausalAttention
]
