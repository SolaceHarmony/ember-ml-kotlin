"""
Attention mechanisms module.

This module provides implementations of attention mechanisms,
including temporal and causal attention.
"""

# Use relative import for subpackage
from .mechanisms import CausalAttention, AttentionState

__all__ = [
    "CausalAttention",
    "AttentionState"
    ]
