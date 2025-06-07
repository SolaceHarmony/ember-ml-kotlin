"""
HPC (High-Precision Computing) Limb arithmetic operations for wave processing.
Implements exact 64-bit chunked arithmetic for wave computations.
"""

from array import array
from typing import List, Union

# Constants for limb arithmetic
CHUNK_BITS = 64
CHUNK_BASE = 1 << CHUNK_BITS
CHUNK_MASK = CHUNK_BASE - 1

class HPCLimb:
    """
    Represents a number using 64-bit limbs for high-precision arithmetic.
    Supports basic arithmetic operations needed for wave processing.
    """
    
    def __init__(self, value: int = 0):
        """Initialize HPCLimb from integer value."""
        self.limbs = int_to_limbs(value)
    
    def copy(self) -> 'HPCLimb':
        """Create a deep copy."""
        new_limb = HPCLimb(0)
        new_limb.limbs = array('Q', self.limbs)
        return new_limb
    
    def to_int(self) -> int:
        """Convert back to Python integer."""
        return limbs_to_int(self.limbs)
    
    def __repr__(self) -> str:
        return f"HPCLimb({self.to_int()})"

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

def hpc_add(A: HPCLimb, B: HPCLimb) -> HPCLimb:
    """Add two HPCLimb numbers."""
    out_len = max(len(A.limbs), len(B.limbs))
    out = array('Q', [0] * (out_len + 1))
    carry = 0
    
    for i in range(out_len):
        av = A.limbs[i] if i < len(A.limbs) else 0
        bv = B.limbs[i] if i < len(B.limbs) else 0
        s_val = av + bv + carry
        out[i] = s_val & CHUNK_MASK
        carry = s_val >> CHUNK_BITS
        
    if carry:
        out[out_len] = carry
    else:
        out.pop()  # remove unused last limb
        
    result = HPCLimb(0)
    result.limbs = out
    return result

def hpc_sub(A: HPCLimb, B: HPCLimb) -> HPCLimb:
    """Subtract B from A, assuming A >= B."""
    out_len = max(len(A.limbs), len(B.limbs))
    out = array('Q', [0] * out_len)
    carry = 0
    
    for i in range(out_len):
        av = A.limbs[i] if i < len(A.limbs) else 0
        bv = B.limbs[i] if i < len(B.limbs) else 0
        diff = av - bv - carry
        
        if diff < 0:
            diff += CHUNK_BASE
            carry = 1
        else:
            carry = 0
            
        out[i] = diff & CHUNK_MASK
        
    while len(out) > 1 and out[-1] == 0:
        out.pop()
        
    result = HPCLimb(0)
    result.limbs = out
    return result

def hpc_shr(A: HPCLimb, shift_bits: int) -> HPCLimb:
    """Right shift HPCLimb by shift_bits."""
    if shift_bits <= 0:
        return A.copy()
        
    out = array('Q', A.limbs)
    limb_shifts = shift_bits // CHUNK_BITS
    bit_shifts = shift_bits % CHUNK_BITS
    
    if limb_shifts >= len(out):
        return HPCLimb(0)
        
    out = out[limb_shifts:]
    
    if bit_shifts == 0:
        if not out:
            out.append(0)
        result = HPCLimb(0)
        result.limbs = out
        return result
        
    carry = 0
    for i in reversed(range(len(out))):
        cur = out[i] | (carry << CHUNK_BITS)
        out[i] = (cur >> bit_shifts) & CHUNK_MASK
        carry = cur & ((1 << bit_shifts) - 1)
        
    while len(out) > 1 and out[-1] == 0:
        out.pop()
        
    if not out:
        out.append(0)
        
    result = HPCLimb(0)
    result.limbs = out
    return result