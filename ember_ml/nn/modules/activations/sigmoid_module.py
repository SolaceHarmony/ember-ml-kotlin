# ember_ml/nn/modules/activations/sigmoid.py
"""
Sigmoid activation module.
"""
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor

class Sigmoid(Module):
    """
    Applies the Sigmoid function element-wise.
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()
    def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
        """Forward pass of Sigmoid."""
        # Import lazily and call the backend-agnostic activation op
        from ember_ml.nn.modules.activations import sigmoid # Import from parent __init__
        return sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"