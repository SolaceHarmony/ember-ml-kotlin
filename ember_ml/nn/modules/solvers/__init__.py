"""
Solver modules for Ember ML.

This module provides various solvers for optimization problems,
including both gradient-based and non-gradient approaches.
"""

from ember_ml.nn.modules.solvers.expectation_solver import ExpectationSolver

__all__ = [
    "ExpectationSolver",
]