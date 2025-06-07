"""
Base optimizer class for ember_ml.

This module provides a backend-agnostic implementation of the base optimizer class
that works with any backend (NumPy, PyTorch, MLX).
"""

from typing import Dict, List, Optional, Union, Any, Callable

from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor
class Optimizer:
    """
    Base class for all optimizers.
    
    This class provides the basic interface for optimizers.
    All optimizers should inherit from this class.
    """
    
    def __init__(self, params=None, lr=0.01):
        """
        Initialize the optimizer.
        
        Args:
            params: Parameters to optimize (Module or list of parameters)
            lr: Learning rate
        """
        self.defaults = {'lr': lr}
        self.state = {}  # Optimizer state
        self.param_groups = []
        
        if params is not None:
            self.add_param_group(params)
    
    def add_param_group(self, params):
        """
        Add a parameter group to the optimizer.
        
        Args:
            params: Module or list of parameters
        """
        if isinstance(params, Module):
            # If params is a Module, extract its parameters
            params = list(params.parameters())
        
        # Create parameter group
        param_group = {
            'params': params,
            **self.defaults
        }
        
        # Initialize state for each parameter
        for param in params:
            if param not in self.state:
                self.state[param] = {}
        
        self.param_groups.append(param_group)
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = tensor.zeros_like(param.grad)
    
    def step(self):
        """
        Perform a single optimization step.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step method")