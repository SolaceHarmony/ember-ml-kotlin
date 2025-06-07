"""
Analysis metrics and measurement utilities for wave memory systems.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import time

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike
from ..utils.math_helpers import (
    compute_energy_stability,
    compute_interference_strength,
    compute_phase_coherence
)

@dataclass
class AnalysisMetrics:
    """
    Stores analysis metrics for wave memory system evaluation.
    
    Attributes:
        computation_time: Time spent on core computations (seconds)
        interference_strength: Average interference between waves (0-1)
        energy_stability: Stability of system energy over time (0-1)
        phase_coherence: Coherence of wave phases (0-1)
        total_time: Total analysis time including visualization (seconds)
    """
    computation_time: float
    interference_strength: float
    energy_stability: float
    phase_coherence: float
    total_time: float
    
    def __post_init__(self):
        """Validate metric values."""
        if self.computation_time < 0:
            raise ValueError("Computation time must be non-negative")
            
        if not 0 <= self.interference_strength <= 1:
            raise ValueError("Interference strength must be between 0 and 1")
            
        if not 0 <= self.energy_stability <= 1:
            raise ValueError("Energy stability must be between 0 and 1")
            
        if not 0 <= self.phase_coherence <= 1:
            raise ValueError("Phase coherence must be between 0 and 1")
            
        if self.total_time < self.computation_time:
            raise ValueError("Total time must be greater than computation time")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            'computation_time': self.computation_time,
            'interference_strength': self.interference_strength,
            'energy_stability': self.energy_stability,
            'phase_coherence': self.phase_coherence,
            'total_time': self.total_time
        }
    
    def __str__(self) -> str:
        """Format metrics for display."""
        return (
            f"Analysis Metrics:\n"
            f"----------------\n"
            f"Computation Time: {self.computation_time:.4f}s\n"
            f"Total Time: {self.total_time:.4f}s\n"
            f"Interference Strength: {self.interference_strength:.4f}\n"
            f"Energy Stability: {self.energy_stability:.4f}\n"
            f"Phase Coherence: {self.phase_coherence:.4f}"
        )

class MetricsCollector:
    """
    Collects and computes analysis metrics during simulation.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.perf_counter()
        self.computation_start = None
        self.computation_time = 0.0
        self.wave_history = []
        
    def start_computation(self):
        """Mark start of computation phase."""
        self.computation_start = time.perf_counter()
        
    def end_computation(self):
        """Mark end of computation phase."""
        if self.computation_start is not None:
            self.computation_time += time.perf_counter() - self.computation_start
            self.computation_start = None
            
    def record_wave_states(self, states: List[TensorLike]):
        """
        Record wave states for analysis.
        
        Args:
            states: List of wave state vectors
        """
        self.wave_history.append([state.copy() for state in states])
        
    def compute_metrics(self) -> AnalysisMetrics:
        """
        Compute final analysis metrics.
        
        Returns:
            AnalysisMetrics object with computed values
        """
        if not self.wave_history:
            raise ValueError("No wave states recorded")
            
        # Convert history to numpy array for efficient computation
        history = tensor.convert_to_tensor(self.wave_history)
        
        # Compute energy history
        energies = [
            [stats.sum(state**2) for state in states]
            for states in self.wave_history
        ]
        total_energy = [sum(e) for e in energies]
        
        # Compute metrics
        interference = compute_interference_strength(self.wave_history[-1])
        stability = compute_energy_stability(total_energy)
        coherence = compute_phase_coherence(self.wave_history[-1])
        
        return AnalysisMetrics(
            computation_time=self.computation_time,
            interference_strength=interference,
            energy_stability=stability,
            phase_coherence=coherence,
            total_time=time.perf_counter() - self.start_time
        )
    
    def get_wave_history(self) -> TensorLike:
        """
        Get recorded wave history.
        
        Returns:
            Array of shape (time_steps, num_spheres, state_dim)
        """
        return tensor.convert_to_tensor(self.wave_history)