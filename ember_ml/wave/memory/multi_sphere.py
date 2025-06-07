"""
Multi-sphere wave model implementation for wave memory systems.
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ember_ml.nn import tensor # Added import
from ember_ml.nn.tensor.types import TensorLike # Added import
from .sphere_overlap import SphereState, SphereOverlap, OverlapNetwork
from ..utils.math_helpers import normalize_vector, partial_interference

class MultiSphereWaveModel:
    """
    Models wave interactions across multiple spherical domains.
    Handles wave propagation, interference, and state evolution.
    """
    
    def __init__(self, 
                 M: int = 3, 
                 reflection: float = 0.2, 
                 transmission: float = 0.7,
                 noise_std: float = 0.0):
        """
        Initialize multi-sphere wave model.
        
        Args:
            M: Number of spheres
            reflection: Default reflection coefficient
            transmission: Default transmission coefficient
            noise_std: Standard deviation of noise
        """
        if M < 1:
            raise ValueError("Must have at least one sphere")
            
        self.M = M
        
        # Create sphere states
        self.spheres = [
            SphereState(
                fast_state=tensor.convert_to_tensor([1.0, 0.0, 0.0, 0.0]),
                slow_state=tensor.convert_to_tensor([1.0, 0.0, 0.0, 0.0]),
                noise_std=noise_std
            )
            for _ in range(M)
        ]
        
        # Create overlap network
        self.overlaps = OverlapNetwork(
            overlaps=[
                SphereOverlap(i, i+1, reflection, transmission)
                for i in range(M-1)
            ],
            num_spheres=M
        )
        
    def set_initial_state(self, 
                         idx: int, 
                         fast_vec: List[float], 
                         slow_vec: Optional[List[float]] = None):
        """
        Set initial state for a specific sphere.
        
        Args:
            idx: Sphere index
            fast_vec: Fast state vector
            slow_vec: Optional slow state vector
        """
        if not 0 <= idx < self.M:
            raise ValueError(f"Invalid sphere index: {idx}")
            
        self.spheres[idx].fast_state = normalize_vector(tensor.convert_to_tensor(fast_vec))
        
        if slow_vec is not None:
            self.spheres[idx].slow_state = normalize_vector(tensor.convert_to_tensor(slow_vec))
            
    def run(self, 
            steps: int,
            input_waves_seq: List[List[Optional[TensorLike]]],
            gating_seq: List[List[bool]]) -> TensorLike:
        """
        Run simulation for specified number of steps.
        
        Args:
            steps: Number of time steps
            input_waves_seq: Sequence of input waves for each sphere
            gating_seq: Sequence of gating signals for each sphere
            
        Returns:
            Array of shape (steps, num_spheres, 4) containing wave history
        """
        if len(input_waves_seq) != steps:
            raise ValueError("Input sequence length must match steps")
            
        if len(gating_seq) != steps:
            raise ValueError("Gating sequence length must match steps")
            
        history = []
        
        for t in range(steps):
            # Get inputs for current time step
            input_waves = input_waves_seq[t]
            gating = gating_seq[t]
            
            if len(input_waves) != self.M or len(gating) != self.M:
                raise ValueError(
                    f"Input/gating length at step {t} does not match number of spheres"
                )
            
            # Update each sphere
            for i, sphere in enumerate(self.spheres):
                if input_waves[i] is not None and gating[i]:
                    self._update_sphere_state(i, input_waves[i])
                    
            # Record current state
            history.append([
                sphere.fast_state.copy() for sphere in self.spheres
            ])
            
            # Process overlaps
            self._process_overlaps()
            
        return tensor.convert_to_tensor(history)
    
    def _update_sphere_state(self, idx: int, input_wave: TensorLike):
        """
        Update state of a single sphere based on input.
        
        Args:
            idx: Sphere index
            input_wave: Input wave vector
        """
        sphere = self.spheres[idx]
        
        # Update fast state with stronger interference
        sphere.fast_state = partial_interference(
            sphere.fast_state, input_wave, alpha=0.8
        )
        
        # Update slow state with weaker interference
        sphere.slow_state = partial_interference(
            sphere.slow_state, input_wave, alpha=0.2
        )
        
        # Add noise if specified
        if sphere.noise_std > 0:
            noise = tensor.random_normal(0, sphere.noise_std, 4)
            sphere.fast_state = normalize_vector(sphere.fast_state + noise)
            sphere.slow_state = normalize_vector(sphere.slow_state + noise)
    
    def _process_overlaps(self):
        """Process wave interactions at sphere overlaps."""
        # Store original states to prevent interference with updates
        original_states = [
            (sphere.fast_state.copy(), sphere.slow_state.copy())
            for sphere in self.spheres
        ]
        
        # Process each overlap
        for overlap in self.overlaps.overlaps:
            sphere_A = self.spheres[overlap.idx_A]
            sphere_B = self.spheres[overlap.idx_B]
            
            # Get original states
            fast_A, slow_A = original_states[overlap.idx_A]
            fast_B, slow_B = original_states[overlap.idx_B]
            
            # Compute reflections and transmissions
            reflect_A = partial_interference(fast_A, fast_B, overlap.reflection_coeff)
            reflect_B = partial_interference(fast_B, fast_A, overlap.reflection_coeff)
            trans_A = partial_interference(fast_A, fast_B, overlap.transmission_coeff)
            trans_B = partial_interference(fast_B, fast_A, overlap.transmission_coeff)
            
            # Update states with interference results
            sphere_A.fast_state = reflect_A
            sphere_B.fast_state = reflect_B
            sphere_A.slow_state = trans_B
            sphere_B.slow_state = trans_A
            
    def get_sphere_states(self) -> List[Tuple[TensorLike, TensorLike]]:
        """
        Get current states of all spheres.
        
        Returns:
            List of (fast_state, slow_state) tuples
        """
        return [
            (sphere.fast_state.copy(), sphere.slow_state.copy())
            for sphere in self.spheres
        ]