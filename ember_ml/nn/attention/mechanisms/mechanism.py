"""
Causal attention mechanism implementation.

This module provides the CausalAttention class which implements a mechanism
for computing and tracking attention based on temporal, causal, and novelty factors.
"""

from typing import Dict, List, Tuple
from ember_ml import ops
from ember_ml.nn.attention.mechanisms.state import AttentionState


class CausalAttention:
    """
    Implements a causal attention mechanism that considers temporal relationships,
    prediction accuracy, and novelty in input patterns.
    
    The mechanism maintains a history of attention states and updates them based
    on various factors including temporal decay and prediction errors.
    
    Attributes:
        states (Dict[int, AttentionState]): Maps neuron IDs to their attention states
        history (List[Tuple[int, float]]): Records attention values over time
        decay_rate (float): Rate at which attention decays over time
        novelty_threshold (float): Threshold for considering input as novel
        memory_length (int): Maximum length of attention history to maintain
    """
    
    def __init__(self,
                 decay_rate: float = 0.1,
                 novelty_threshold: float = 0.3,
                 memory_length: int = 100):
        """
        Initialize the attention mechanism.
        
        Args:
            decay_rate: Rate at which attention decays over time
            novelty_threshold: Threshold for considering input as novel
            memory_length: Maximum number of historical states to maintain
        """
        self.states: Dict[int, AttentionState] = {}
        self.history: List[Tuple[int, float]] = []
        self.decay_rate = decay_rate
        self.novelty_threshold = novelty_threshold
        self.memory_length = memory_length
    
    def update(self,
              neuron_id: int,
              prediction_error: float,
              current_state: float,
              target_state: float) -> float:
        """
        Update attention state for a neuron based on its current context.
        
        Args:
            neuron_id: Identifier for the neuron
            prediction_error: Difference between predicted and actual values
            current_state: Current neuron state
            target_state: Target/desired neuron state
            
        Returns:
            float: Total attention value after update
        """
        # Get or create attention state for this neuron
        state = self.states.get(neuron_id, AttentionState())
        
        # Update temporal weight based on history length
        temporal_decay = ops.exp(-self.decay_rate * len(self.history))
        state.temporal_weight = current_state * temporal_decay
        
        # Update causal weight based on prediction accuracy
        prediction_accuracy = 1.0 - min(abs(prediction_error), 1.0)
        state.causal_weight = prediction_accuracy
        
        # Update novelty weight based on state change
        novelty = abs(target_state - current_state)
        if novelty > self.novelty_threshold:
            state.novelty_weight = novelty
        else:
            state.novelty_weight *= (1 - self.decay_rate)
        
        # Store updated state
        self.states[neuron_id] = state
        
        # Update history and maintain length limit
        total_attention = state.compute_total()
        self.history.append((neuron_id, total_attention))
        if len(self.history) > self.memory_length:
            self.history.pop(0)
        
        return total_attention