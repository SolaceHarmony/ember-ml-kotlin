"""Wave memory analysis and processing module.

This module provides implementations for analyzing and processing wave-based
memory systems, including metrics, visualizations, and mathematical helpers.

Components:
    metrics: Quantitative analysis of wave memory systems
        - Memory capacity measurements
        - Interference detection
        - Temporal stability analysis
        
    visualizer: Tools for visualizing wave memory dynamics
        - State space visualization
        - Temporal evolution plots
        - Interference pattern analysis
        
    math_helpers: Mathematical utilities for wave processing
        - Wave equation solvers
        - Boundary condition handlers
        - Phase space analysis
        
    multi_sphere: Multi-sphere wave interaction models
        - Sphere-to-sphere coupling
        - Wave propagation between spheres
        - Boundary interaction handling
        
    sphere_overlap: Sphere intersection and interaction handling
        - Overlap region definition
        - Coupling strength calculation
        - Wave transmission modeling

All implementations maintain backend independence through ops abstraction.
"""

from ember_ml.wave.memory.metrics import *
from ember_ml.wave.memory.visualizer import *
from ember_ml.wave.memory.math_helpers import *
from ember_ml.wave.memory.multi_sphere import *
from ember_ml.wave.memory.sphere_overlap import *

__all__ = [
    'metrics',
    'visualizer',
    'math_helpers',
    'multi_sphere',
    'sphere_overlap',
]
