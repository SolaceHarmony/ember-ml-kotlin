"""
Module class re-export.

This module re-exports the BaseModule class as Module.
"""

from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter

__all__ = ['Module', 'Parameter']