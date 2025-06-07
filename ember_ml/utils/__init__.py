"""Utility modules for backend-agnostic operations.

This package contains utility modules that support the ember_ml framework
while maintaining strict backend independence.

Components:
    backend_utils: Backend-agnostic utility functions
        - Backend selection and management
        - Tensor conversion utilities
        - Device management helpers
        - Memory optimization tools
"""

from ember_ml.utils import backend_utils

__all__ = ['backend_utils']