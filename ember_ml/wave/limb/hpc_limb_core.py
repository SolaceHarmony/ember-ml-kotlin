"""
HPC (High-Precision Computing) Limb core functionality for wave processing.
Implements exact 64-bit chunked arithmetic for wave computations.
"""

from array import array
from typing import List, Union, Optional
import numpy as np

# Constants for limb arithmetic
CHUNK_BITS = 64
CHUNK_BASE = 1 << CHUNK_BITS
CHUNK_MASK = CHUNK_BASE - 1

def int_to_limbs(value: int) -> array:
    """Convert a nonnegative Python int to array of 64-bit limbs."""
    if value < 0:
        raise ValueError("Negative ints not supported")
        
    limbs = array('Q')  # 'Q' = unsigned long long
    
    while value > 0:
        limbs.append(value & CHUNK_MASK)
        value >>= CHUNK_BITS
        
    if not limbs:
        limbs.append(0)
        
    return limbs

def limbs_to_int(limbs: array) -> int:
    """Combine array of 64-bit limbs to a single Python int."""
    val = 0
    shift = 0
    
    for limb in limbs:
        val += (limb << shift)
        shift += CHUNK_BITS
        
    return val

def hpc_add(A: array, B: array) -> array:
    """Add two HPC-limb arrays."""
    out_len = max(len(A), len(B))
    out = array('Q', [0] * (out_len + 1))
    carry = 0
    
    for i in range(out_len):
        av = A[i] if i < len(A) else 0
        bv = B[i] if i < len(B) else 0
        s_val = av + bv + carry
        out[i] = s_val & CHUNK_MASK
        carry = s_val >> CHUNK_BITS
        
    if carry:
        out[out_len] = carry
    else:
        out.pop()  # remove unused last limb
        
    return out

def hpc_sub(A: array, B: array) -> array:
    """Subtract B from A, assuming A >= B."""
    out_len = max(len(A), len(B))
    out = array('Q', [0] * out_len)
    carry = 0
    
    for i in range(out_len):
        av = A[i] if i < len(A) else 0
        bv = B[i] if i < len(B) else 0
        diff = av - bv - carry
        
        if diff < 0:
            diff += CHUNK_BASE
            carry = 1
        else:
            carry = 0
            
        out[i] = diff & CHUNK_MASK
        
    while len(out) > 1 and out[-1] == 0:
        out.pop()
        
    return out

def hpc_shr(A: array, shift_bits: int) -> array:
    """Right shift HPC-limb array by shift_bits."""
    if shift_bits <= 0:
        return array('Q', A)
        
    out = array('Q', A)
    limb_shifts = shift_bits // CHUNK_BITS
    bit_shifts = shift_bits % CHUNK_BITS
    
    if limb_shifts >= len(out):
        return array('Q', [0])
        
    out = out[limb_shifts:]
    
    if bit_shifts == 0:
        if not out:
            out.append(0)
        return out
        
    carry = 0
    for i in reversed(range(len(out))):
        cur = out[i] | (carry << CHUNK_BITS)
        out[i] = (cur >> bit_shifts) & CHUNK_MASK
        carry = cur & ((1 << bit_shifts) - 1)
        
    while len(out) > 1 and out[-1] == 0:
        out.pop()
        
    if not out:
        out.append(0)
        
    return out

def hpc_compare(A: array, B: array) -> int:
    """Compare two HPC-limb arrays. Returns -1 if A<B, 0 if A=B, 1 if A>B."""
    if len(A) < len(B):
        return -1
    if len(A) > len(B):
        return 1
        
    for i in reversed(range(len(A))):
        if A[i] < B[i]:
            return -1
        if A[i] > B[i]:
            return 1
            
    return 0

class HPCWaveSegment:
    """
    Represents a wave segment using HPC limb arithmetic for precise computations.
    """
    
    def __init__(self, wave_max_val: int = 1_000_000, start_val: int = 0):
        """
        Initialize wave segment with given parameters.
        
        Args:
            wave_max_val: Maximum wave amplitude
            start_val: Initial wave value
        """
        self.wave_max = int_to_limbs(wave_max_val)
        self.wave_state = int_to_limbs(start_val)
        
    def update(self, input_val: array) -> None:
        """
        Update wave segment state with input value.
        
        Args:
            input_val: Input value as HPC-limb array
        """
        self.wave_state = hpc_add(self.wave_state, input_val)
        
        # Apply bounds checking
        if hpc_compare(self.wave_state, self.wave_max) > 0:
            self.wave_state = array('Q', self.wave_max)
            
    def get_state(self) -> array:
        """Get current wave state as HPC-limb array."""
        return array('Q', self.wave_state)
        
    def get_normalized_state(self) -> float:
        """Get current wave state normalized to [0,1] range."""
        return limbs_to_int(self.wave_state) / limbs_to_int(self.wave_max)