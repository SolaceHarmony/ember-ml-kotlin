"""
Dopamine modulation system implementation.

This module provides the DopamineModulator class that simulates basic
dopamine dynamics for modulating neural behavior based on input strength.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ember_ml.ops import stats

@dataclass
class DopamineState:
    """
    Represents the current state of the dopamine system.
    
    Attributes:
        level (float): Current dopamine level
        baseline (float): Baseline dopamine level
        recent_changes (List[float]): History of recent level changes
    """
    level: float = 0.0
    baseline: float = 0.0
    recent_changes: List[float] = None
    
    def __post_init__(self):
        """Initialize empty history if none provided."""
        if self.recent_changes is None:
            self.recent_changes = []


class DopamineModulator:
    """
    Simulates basic dopamine dynamics for neural modulation.
    
    This implementation models a simple dopamine system that:
    - Increases dopamine with strong input
    - Maintains a natural decay over time
    - Tracks history of dopamine changes
    - Provides modulation factors for neural dynamics
    
    Attributes:
        dopamine_level (float): Current dopamine concentration
        increase_rate (float): Rate of dopamine increase with strong input
        decay_rate (float): Natural decay rate of dopamine
        history (List[Tuple[float, float]]): History of (time, level) pairs
    """
    
    def __init__(self,
                 increase_rate: float = 0.05,
                 decay_rate: float = 0.01,
                 history_length: int = 1000):
        """
        Initialize the dopamine modulation system.
        
        Args:
            increase_rate: Rate of dopamine increase with strong input
            decay_rate: Rate of natural dopamine decay
            history_length: Maximum number of historical states to maintain
        """
        self.state = DopamineState()
        self.increase_rate = increase_rate
        self.decay_rate = decay_rate
        self.history_length = history_length
        self.history: List[Tuple[float, float]] = []  # (time, level)
        self._time = 0.0
    
    def update(self, input_strength: float, dt: float = 0.1) -> float:
        """
        Update dopamine levels based on input strength.
        
        Args:
            input_strength: Measure of input signal strength
            dt: Time step size
            
        Returns:
            float: New dopamine level
            
        Note:
            Dopamine increases when input_strength > 0.5 (threshold)
            and naturally decays otherwise.
        """
        # Record previous level for history
        prev_level = self.state.level
        
        # Increase dopamine if input is strong (above 0.5 threshold)
        if input_strength > 0.5:
            increase = self.increase_rate * (input_strength - 0.5)
            self.state.level += increase
        
        # Apply natural decay
        self.state.level = max(0.0, self.state.level - self.decay_rate * dt)
        
        # Update history
        self._time += dt
        self.history.append((self._time, self.state.level))
        
        # Maintain history length
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        # Track recent changes
        change = self.state.level - prev_level
        self.state.recent_changes.append(change)
        if len(self.state.recent_changes) > 10:  # Keep last 10 changes
            self.state.recent_changes.pop(0)
        
        return self.state.level
    
    def get_dopamine_modulation(self) -> float:
        """
        Get the current dopamine modulation factor.
        
        Returns:
            float: Modulation factor (1.0 + dopamine_level)
            
        Note:
            Higher dopamine results in larger modulation factor,
            which typically leads to slower decay in neural dynamics.
        """
        return 1.0 + self.state.level
    
    def get_recent_trend(self) -> float:
        """
        Calculate recent trend in dopamine changes.
        
        Returns:
            float: Average recent change in dopamine level
        """
        if not self.state.recent_changes:
            return 0.0
        return stats.mean(self.state.recent_changes)
    
    def reset(self) -> None:
        """Reset the dopamine system to initial state."""
        self.state = DopamineState()
        self.history.clear()
        self._time = 0.0