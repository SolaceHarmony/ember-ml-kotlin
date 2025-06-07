"""
Wave segment implementation using HPC limb arithmetic for precise wave computations.
"""

from typing import List, Optional
from ember_ml.nn.tensor.types import TensorLike # Added import
from ember_ml.nn import tensor # Moved import to top level
import numpy as np
from .hpc_limb import HPCLimb, hpc_add, hpc_sub, hpc_shr

class WaveSegment:
    """
    Represents a segment of a wave using HPC limb arithmetic for precise computations.
    Each segment maintains its state and can interact with neighboring segments.
    """
    
    def __init__(self, 
                 initial_state: float = 0.01,
                 wave_max: float = 1.0,
                 ca_threshold: float = 0.02,
                 k_threshold: float = 0.03):
        """
        Initialize wave segment with given parameters.
        
        Args:
            initial_state: Initial wave amplitude (0.0-1.0)
            wave_max: Maximum wave amplitude
            ca_threshold: Calcium channel activation threshold
            k_threshold: Potassium channel activation threshold
        """
        # Convert float parameters to HPC limb integers by scaling
        scale = 1_000_000  # Use 1M for 6 decimal places precision
        
        self.wave_max = HPCLimb(int(wave_max * scale))
        self.wave_state = HPCLimb(int(initial_state * scale))
        self.ca_threshold = HPCLimb(int(ca_threshold * scale))
        self.k_threshold = HPCLimb(int(k_threshold * scale))
        
        # History tracking
        self.toggle_history: List[HPCLimb] = []
        self.conduction_history: List[HPCLimb] = []
        
    def propagate(self, conduction_val: HPCLimb) -> None:
        """
        Update segment state based on conduction value.
        
        Args:
            conduction_val: Amount of conduction from neighboring segments
        """
        # Store conduction value in history
        self.conduction_history.append(conduction_val)
        
        # Apply conduction after delay if history exists
        if self.toggle_history:
            add_val = self.toggle_history.pop(0)
            self.wave_state = hpc_add(self.wave_state, add_val)
            
        # Store new toggle value
        self.toggle_history.append(conduction_val)
        
    def update_ion_channels(self) -> None:
        """Update ion channel effects on wave state."""
        wave_val = self.wave_state.to_int()
        
        # Calcium channel effect
        if wave_val > self.ca_threshold.to_int():
            # Increase by wave_state/16
            increment = hpc_shr(self.wave_state, 4)
            self.wave_state = hpc_add(self.wave_state, increment)
            
        # Potassium channel effect
        if wave_val > self.k_threshold.to_int():
            # Decrease by wave_state/8
            decrement = hpc_shr(self.wave_state, 3)
            self.wave_state = hpc_sub(self.wave_state, decrement)
            
    def get_conduction_value(self) -> HPCLimb:
        """Calculate conduction value to propagate to neighbors."""
        return hpc_shr(self.wave_state, 3)  # wave_state/8
        
    def get_normalized_state(self) -> float:
        """Get current wave state normalized to [0,1] range."""
        return self.wave_state.to_int() / self.wave_max.to_int()

class WaveSegmentArray:
    """
    Array of wave segments that can interact with each other.
    Handles propagation and boundary conditions.
    """
    
    def __init__(self, num_segments: int = 8):
        """Initialize array of wave segments."""
        self.segments = [WaveSegment() for _ in range(num_segments)]
        
    def update(self) -> None:
        """Update all segments for one time step."""
        # Calculate conduction values
        conduction_vals = []
        for seg in self.segments:
            conduction_vals.append(seg.get_conduction_value())
            
        # Propagate to neighbors
        for i, seg in enumerate(self.segments):
            # Get conduction from neighbors
            left_val = conduction_vals[i-1] if i > 0 else HPCLimb(0)
            right_val = conduction_vals[i+1] if i < len(self.segments)-1 else HPCLimb(0)
            
            # Combine neighbor conduction
            total_conduction = hpc_add(left_val, right_val)
            
            # Update segment
            seg.propagate(total_conduction)
            seg.update_ion_channels()
            
        # Apply boundary reflections
        if self.segments:
            self._apply_boundary_reflection(self.segments[0])  # Left boundary
            self._apply_boundary_reflection(self.segments[-1]) # Right boundary
            
    def _apply_boundary_reflection(self, segment: WaveSegment) -> None:
        """Apply partial reflection at boundary."""
        reflection = hpc_shr(segment.wave_state, 4)  # wave_state/16
        segment.wave_state = hpc_sub(segment.wave_state, reflection)
        
    def get_wave_state(self) -> TensorLike:
        """Get array of normalized wave states."""
        # Removed import from here: from ember_ml.nn import tensor
        return tensor.convert_to_tensor([seg.get_normalized_state() for seg in self.segments])