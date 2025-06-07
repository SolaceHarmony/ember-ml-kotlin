"""Geometric operations for non-Euclidean neural computations."""

from regex import T
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module, Parameter
from ember_ml.ops import stats
from ember_ml.ops import linearalg
from ember_ml import ops
from typing import Tuple
from ember_ml.nn.tensor.types import TensorLike
from typing import Any


def normalize_sphere(vec: tensor.convert_to_tensor, eps: float = 1e-12) -> tensor.convert_to_tensor:
    """Normalize vectors to the unit sphere.
    
    Args:
        vec: Input vectors (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized vectors on unit sphere
    """
    norm = linearalg.norm(vec, dim=-1, keepdim=True)
    mask = norm > eps
    return ops.where(mask, vec / norm, vec)

def log_map_sphere(
    p: TensorLike,
    q: TensorLike,
    eps: float = 1e-12
) -> Any:
    """Logarithmic map on unit sphere (Log_p(q)).
    
    Maps point q to the tangent space at p.
    
    Args:
        p: Base point(s) (..., 3)
        q: Target point(s) (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Tangent vector(s) at p
    """
    # Normalize inputs to unit sphere
    p_n = normalize_sphere(p, eps)
    q_n = normalize_sphere(q, eps)
    
    # Compute angle between p and q
    dot_prod = stats.sum(p_n * q_n, dim=-1, keepdim=True)
    dot_prod = ops.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
    theta = ops.arccos(dot_prod)
    
    # Handle small angles
    small_angle = theta < eps
    if small_angle.any():
        return tensor.zeros_like(p)
    
    # Compute direction in tangent space
    perp = q_n - dot_prod * p_n
    perp_norm = linearalg.norm(perp, dim=-1, keepdim=True)
    perp_mask = perp_norm > eps
    
    # Combine results
    dir_vec = ops.where(perp_mask, perp / perp_norm, tensor.zeros_like(perp))
    return dir_vec * theta

def exp_map_sphere(
    p: tensor.convert_to_tensor,
    v: tensor.convert_to_tensor,
    eps: float = 1e-12
) -> tensor.convert_to_tensor:
    """Exponential map on unit sphere (Exp_p(v)).
    
    Maps tangent vector v at p to the sphere.
    
    Args:
        p: Base point(s) (..., 3)
        v: Tangent vector(s) at p (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Point(s) on sphere
    """
    # Get vector norm
    v_norm = linearalg.norm(v, dim=-1, keepdim=True)
    small_norm = v_norm < eps
    
    if small_norm.any():
        return p
    
    # Normalize base point and direction
    p_n = normalize_sphere(p, eps)
    dir_vec = v / v_norm
    
    # Remove component along p
    proj_p = stats.sum(dir_vec * p_n, dim=-1, keepdim=True) * p_n
    dir_vec = dir_vec - proj_p
    dir_vec = normalize_sphere(dir_vec, eps)
    
    # Compute new point
    new_point = ops.cos(v_norm) * p_n + ops.sin(v_norm) * dir_vec
    return normalize_sphere(new_point, eps)

def parallel_transport_sphere(
    p: tensor.convert_to_tensor,
    q: tensor.convert_to_tensor,
    v: tensor.convert_to_tensor,
    eps: float = 1e-12
) -> tensor.convert_to_tensor:
    """Parallel transport tangent vector v from p to q on sphere.
    
    Args:
        p: Start point(s) (..., 3)
        q: End point(s) (..., 3)
        v: Tangent vector(s) at p (..., 3)
        eps: Small constant for numerical stability
        
    Returns:
        Transported vector(s) at q
    """
    # Normalize points
    p_n = normalize_sphere(p, eps)
    q_n = normalize_sphere(q, eps)
    
    # Get geodesic
    dot_prod = stats.sum(p_n * q_n, dim=-1, keepdim=True)
    dot_prod = ops.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
    theta = ops.arccos(dot_prod)
    
    # Handle small angles
    small_angle = theta < eps
    if small_angle.any():
        return v
        
    # Get transport direction
    transport_dir = q_n - dot_prod * p_n
    transport_dir = normalize_sphere(transport_dir, eps)
    
    # Transport v
    v_proj_p = stats.sum(v * p_n, dim=-1, keepdim=True) * p_n
    v_perp = v - v_proj_p
    
    transported = (
        ops.cos(theta) * v_perp +
        ops.sin(theta) * ops.cross(transport_dir, v_perp)
    )
    
    return transported

class SphericalLinear(Module):
    """Linear transformation in spherical geometry.
    
    Maps tangent vectors between spherical tangent spaces.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(tensor.random_normal(out_features, in_features))
        self.bias = Parameter(tensor.random_normal(out_features))
        
    def forward(
        self,
        x: tensor.convert_to_tensor,
        base_point: tensor.convert_to_tensor
    ) -> Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]:
        """Apply spherical linear transformation.
        
        Args:
            x: Input tangent vectors at base_point
            base_point: Point on sphere where input vectors live
            
        Returns:
            (output tangent vectors, new base point)
        """
        # Linear transform in tangent space
        output = ops.linear(x, self.weight, self.bias)
        
        # Map result to sphere
        new_point = exp_map_sphere(base_point, output)
        
        # Get output in tangent space at new point
        output_tangent = log_map_sphere(new_point, output)
        
        return output_tangent, new_point