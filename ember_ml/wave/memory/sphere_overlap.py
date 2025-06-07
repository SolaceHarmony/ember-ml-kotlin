"""
Core data structures for defining sphere interactions in wave memory systems.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ember_ml.nn import tensor # Added import
from ember_ml.nn.tensor.types import TensorLike # Added import

@dataclass
class SphereOverlap:
    """
    Defines the interaction parameters between two adjacent spheres.
    
    Attributes:
        idx_A: Index of first sphere
        idx_B: Index of second sphere
        reflection_coeff: Coefficient for wave reflection at boundary
        transmission_coeff: Coefficient for wave transmission across boundary
    """
    idx_A: int
    idx_B: int
    reflection_coeff: float
    transmission_coeff: float
    
    def __post_init__(self):
        """Validate overlap parameters after initialization."""
        if self.idx_A == self.idx_B:
            raise ValueError("Sphere indices must be different")
            
        if not 0 <= self.reflection_coeff <= 1:
            raise ValueError("Reflection coefficient must be between 0 and 1")
            
        if not 0 <= self.transmission_coeff <= 1:
            raise ValueError("Transmission coefficient must be between 0 and 1")
            
        if abs(self.reflection_coeff + self.transmission_coeff - 1.0) > 1e-6:
            raise ValueError("Reflection and transmission coefficients must sum to 1")

@dataclass
class SphereState:
    """
    Represents the state of a single sphere in the wave memory system.
    
    Attributes:
        fast_state: Fast-changing wave state vector
        slow_state: Slow-changing wave state vector
        noise_std: Standard deviation of noise to add during updates
    """
    fast_state: TensorLike
    slow_state: TensorLike
    noise_std: float = 0.0
    
    def __post_init__(self):
        """Convert states to numpy arrays and validate."""
        if not isinstance(self.fast_state, TensorLike):
            self.fast_state = tensor.convert_to_tensor(self.fast_state, dtype=tensor.float32)
            
        if not isinstance(self.slow_state, TensorLike):
            self.slow_state = tensor.convert_to_tensor(self.slow_state, dtype=tensor.float32)
            
        if self.fast_state.shape != (4,):
            raise ValueError("Fast state must be a 4D vector")
            
        if self.slow_state.shape != (4,):
            raise ValueError("Slow state must be a 4D vector")
            
        if self.noise_std < 0:
            raise ValueError("Noise standard deviation must be non-negative")

@dataclass
class OverlapNetwork:
    """
    Represents the network of overlaps between spheres in the system.
    
    Attributes:
        overlaps: List of sphere overlap definitions
        num_spheres: Total number of spheres in the system
    """
    overlaps: List[SphereOverlap]
    num_spheres: int
    
    def __post_init__(self):
        """Validate overlap network configuration."""
        if not self.overlaps:
            return
            
        # Check sphere indices are valid
        max_idx = max(
            max(o.idx_A for o in self.overlaps),
            max(o.idx_B for o in self.overlaps)
        )
        if max_idx >= self.num_spheres:
            raise ValueError("Overlap references invalid sphere index")
            
        # Check for duplicate overlaps
        overlap_pairs = set()
        for overlap in self.overlaps:
            pair = tuple(sorted([overlap.idx_A, overlap.idx_B]))
            if pair in overlap_pairs:
                raise ValueError(f"Duplicate overlap between spheres {pair}")
            overlap_pairs.add(pair)
    
    def get_neighbors(self, sphere_idx: int) -> List[int]:
        """
        Get indices of spheres that overlap with given sphere.
        
        Args:
            sphere_idx: Index of sphere to find neighbors for
            
        Returns:
            List of neighboring sphere indices
        """
        neighbors = []
        for overlap in self.overlaps:
            if overlap.idx_A == sphere_idx:
                neighbors.append(overlap.idx_B)
            elif overlap.idx_B == sphere_idx:
                neighbors.append(overlap.idx_A)
        return sorted(neighbors)
    
    def get_overlap(self, idx_A: int, idx_B: int) -> Optional[SphereOverlap]:
        """
        Get overlap between two spheres if it exists.
        
        Args:
            idx_A: Index of first sphere
            idx_B: Index of second sphere
            
        Returns:
            SphereOverlap if exists, None otherwise
        """
        for overlap in self.overlaps:
            if (overlap.idx_A == idx_A and overlap.idx_B == idx_B) or \
               (overlap.idx_A == idx_B and overlap.idx_B == idx_A):
                return overlap
        return None