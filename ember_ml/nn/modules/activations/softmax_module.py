# ember_ml/nn/modules/activations/softmax.py
"""
Softmax activation module.
"""
from typing import Optional, Dict, Any # Added Dict, Any
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor

class Softmax(Module):
    """
    Applies the Softmax function along a specified axis.
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        axis (Optional[int]): The axis along which the Softmax should be computed.
                               Defaults to -1 (the last axis).
    """
    def __init__(self, axis: Optional[int] = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: tensor.EmberTensor) -> tensor.EmberTensor:
        """Forward pass of Softmax."""
        # Import lazily and call the backend-agnostic activation op
        from ember_ml.nn.modules.activations import softmax # Import from parent __init__
        return softmax(x, axis=self.axis)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the Softmax module."""
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    # from_config can rely on BaseModule implementation