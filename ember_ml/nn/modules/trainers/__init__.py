"""
Trainer modules for Ember ML.

This module provides various trainers for model training,
including memory-optimized trainers for specific hardware.
"""

from ember_ml.nn.modules.trainers.memory_optimized_trainer import MemoryOptimizedTrainer

__all__ = [
    "MemoryOptimizedTrainer",
]