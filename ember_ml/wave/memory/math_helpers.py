"""
Mathematical utility functions for wave memory analysis.
Provides core operations for vector manipulation and wave calculations.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.tensor.types import TensorLike

def normalize_vector(vec: TensorLike, epsilon: float = 1e-12) -> TensorLike:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized vector
    """
    norm = ops.linearalg.norm(vec)
    if norm > epsilon:
        return vec / norm
    return tensor.convert_to_tensor([1.0, 0.0, 0.0, 0.0])  # Default 4D unit vector

def compute_phase_angle(vec: TensorLike) -> float:
    """
    Compute phase angle between first component and remaining components.
    
    Args:
        vec: Input vector (assumes 4D vector)
        
    Returns:
        Phase angle in radians
    """
    return np.arctan2(ops.linearalg.norm(vec[1:]), vec[0])

def compute_energy(vec: TensorLike) -> float:
    """
    Compute energy (squared L2 norm) of a vector.
    
    Args:
        vec: Input vector
        
    Returns:
        Energy value
    """
    return stats.sum(vec**2)

def partial_interference(base: TensorLike, 
                       new: TensorLike, 
                       alpha: float,
                       epsilon: float = 1e-12) -> TensorLike:
    """
    Compute partial interference between two vectors.
    
    Args:
        base: Base vector
        new: New vector to interfere with
        alpha: Interference strength (0 to 1)
        epsilon: Small value for numerical stability
        
    Returns:
        Resulting vector after interference
    """
    base = normalize_vector(base)
    new = normalize_vector(new)
    
    # Compute dot product and angle
    dot_prod = ops.clip(ops.dot(base, new), -1.0, 1.0)
    angle = np.arccos(dot_prod)
    
    if angle < epsilon:
        return base
        
    # Compute perpendicular component
    perp = new - dot_prod * base
    perp_norm = ops.linearalg.norm(perp)
    
    if perp_norm < epsilon:
        return base
        
    # Compute interference direction
    direction = perp / perp_norm
    v_scaled = alpha * angle * direction
    
    # Return interpolated vector
    return normalize_vector(
        base * ops.cos(angle) + direction * ops.sin(angle)
    )

def compute_phase_coherence(vectors: List[TensorLike]) -> float:
    """
    Compute phase coherence between multiple vectors.
    
    Args:
        vectors: List of vectors
        
    Returns:
        Phase coherence value between 0 and 1
    """
    if len(vectors) < 2:
        return 1.0
        
    phases = [compute_phase_angle(vec) for vec in vectors]
    diffs = [abs(p1 - p2) for i, p1 in enumerate(phases) 
             for p2 in phases[i+1:]]
             
    return float(stats.mean(ops.cos(diffs)))

def compute_interference_strength(vectors: List[TensorLike]) -> float:
    """
    Compute average interference strength between vectors.
    
    Args:
        vectors: List of vectors
        
    Returns:
        Average interference strength
    """
    if len(vectors) < 2:
        return 0.0
        
    # Compute all pairwise dot products
    dot_products = tensor.convert_to_tensor([
        [ops.abs(ops.dot(v1, v2)) for v2 in vectors]
        for v1 in vectors
    ])
    
    # Zero out diagonal (self-interference)
    np.fill_diagonal(dot_products, 0)
    
    # Return average non-zero interference
    return float(stats.mean(dot_products[dot_products != 0]))

def compute_energy_stability(energy_history: List[float]) -> float:
    """
    Compute energy stability metric from energy history.
    
    Args:
        energy_history: List of energy values over time
        
    Returns:
        Stability metric between 0 and 1
    """
    if not energy_history:
        return 1.0
        
    # Use inverse of energy standard deviation as stability metric
    return 1.0 / (1.0 + stats.std(energy_history))

def create_rotation_matrix(angle: float, axis: int = 0) -> TensorLike:
    """
    Create 4D rotation matrix for given angle and axis.
    
    Args:
        angle: Rotation angle in radians
        axis: Rotation axis (0=xy, 1=xz, 2=xw, 3=yz, 4=yw, 5=zw)
        
    Returns:
        4x4 rotation matrix
    """
    c, s = ops.cos(angle), ops.sin(angle)
    matrix = ops.eye(4)
    
    if axis == 0:  # xy rotation
        matrix[0:2, 0:2] = [[c, -s], [s, c]]
    elif axis == 1:  # xz rotation
        matrix[[0,2]][:,[0,2]] = [[c, -s], [s, c]]
    elif axis == 2:  # xw rotation
        matrix[[0,3]][:,[0,3]] = [[c, -s], [s, c]]
    elif axis == 3:  # yz rotation
        matrix[[1,2]][:,[1,2]] = [[c, -s], [s, c]]
    elif axis == 4:  # yw rotation
        matrix[[1,3]][:,[1,3]] = [[c, -s], [s, c]]
    elif axis == 5:  # zw rotation
        matrix[[2,3]][:,[2,3]] = [[c, -s], [s, c]]
        
    return matrix