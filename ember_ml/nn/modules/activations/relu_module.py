# ember_ml/nn/modules/activations/relu.py
"""
Rectified Linear Unit (ReLU) activation module.
"""
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor # Assuming EmberTensor type hints

class ReLU(Module):
    """
    Applies the Rectified Linear Unit function element-wise.
    ReLU(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
        """Forward pass of ReLU."""
        # Call the backend-agnostic activation op
        from ember_ml.nn.modules.activations import relu # Import from parent __init__
        return relu(x)

    def __repr__(self) -> str:
        return "ReLU()"