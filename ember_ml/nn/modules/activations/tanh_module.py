# ember_ml/nn/modules/activations/tanh.py
"""
Hyperbolic Tangent (Tanh) activation module.
"""
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor

class Tanh(Module):
    """
    Applies the Hyperbolic Tangent function element-wise.
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def __init__(self):
        super().__init__()
    def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
        """Forward pass of Tanh."""
        # Import lazily and call the backend-agnostic activation op
        from ember_ml.nn.modules.activations import tanh # Import from parent __init__
        return tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"