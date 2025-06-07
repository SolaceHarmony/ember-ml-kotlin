"""
Attention state management for neural network components.

This module provides the AttentionState class which encapsulates different
attention weights used in the network's attention mechanism.
"""

from dataclasses import dataclass


@dataclass
class AttentionState:
    """
    Represents the attention state of a neuron with different weight components.
    
    Attributes:
        temporal_weight (float): Weight based on recent history importance
        causal_weight (float): Weight based on prediction accuracy
        novelty_weight (float): Weight based on input novelty/curiosity
    """
    
    temporal_weight: float = 0.0
    causal_weight: float = 0.0
    novelty_weight: float = 0.0
    
    def compute_total(self) -> float:
        """
        Compute the total attention value by averaging all weights.
        
        Returns:
            float: The average of temporal, causal, and novelty weights
        """
        return (
            self.temporal_weight +
            self.causal_weight +
            self.novelty_weight
        ) / 3.0