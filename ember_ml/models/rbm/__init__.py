"""
Restricted Boltzmann Machine (RBM) Module

This package provides an implementation of Restricted Boltzmann Machines
using the ember_ml Module system.
"""

from ember_ml.models.rbm.training import (
    contrastive_divergence_step,
    train_rbm,
    transform_in_chunks,
    save_rbm,
    load_rbm
)
from ember_ml.models.rbm.rbm import RestrictedBoltzmannMachine


__all__ = [
    'RestrictedBoltzmannMachine',
    'contrastive_divergence_step',
    'train_rbm',
    'transform_in_chunks',
    'save_rbm',
    'load_rbm'
]
