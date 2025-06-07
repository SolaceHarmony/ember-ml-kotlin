"""
Stochastic Gradient Descent (SGD) optimizer for ember_ml.

This module provides a backend-agnostic implementation of the SGD optimizer
that works with any backend (NumPy, PyTorch, MLX).
"""

# No specific typing imports needed currently

from ember_ml import ops
from ember_ml.training.optimizer.base import Optimizer
from ember_ml.nn import tensor
class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    This optimizer implements the standard SGD algorithm with momentum and weight decay.
    """
    
    def __init__(self, params=None, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        """
        Initialize the SGD optimizer.
        
        Args:
            params: Parameters to optimize (Module or list of parameters)
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            nesterov: Whether to use Nesterov momentum
        """
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'nesterov': nesterov
        }
        # Call super without args, as defaults are set here and add_param_group is called below
        super().__init__()
        # Set defaults *after* super().__init__ initializes state/param_groups but *before* adding params
        self.defaults = defaults
        
        # Manually add parameter group using the correct defaults
        if params is not None:
            self.add_param_group(params)
    
    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            lr = group['lr']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = ops.add(grad, ops.multiply(weight_decay, param.data))
                
                # Get state for this parameter
                state = self.state.get(param, {})
                
                # Initialize momentum buffer if needed
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = tensor.zeros_like(param.data)
                
                # Update momentum buffer
                buf = state['momentum_buffer']
                buf = ops.add(ops.multiply(momentum, buf), grad)
                state['momentum_buffer'] = buf
                
                # Apply Nesterov momentum if enabled
                if nesterov:
                    grad = ops.add(grad, ops.multiply(momentum, buf))
                else:
                    grad = buf
                
                # Update parameter
                param.data = ops.subtract(param.data, ops.multiply(lr, grad))
                
                # Update state
                self.state[param] = state