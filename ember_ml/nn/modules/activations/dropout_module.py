# ember_ml/nn/modules/activations/dropout.py
"""
Dropout regularization module.
"""
from typing import Dict, Any # Added imports
from ember_ml import ops
from ember_ml.nn.modules import Module
from ember_ml.nn import tensor

class Dropout(Module):
    """
    Applies Dropout to the input.

    Dropout consists in randomly setting a fraction `rate` of input units
    to 0 at each update during training time, which helps prevent overfitting.
    Units not set to 0 are scaled up by 1/(1 - rate) such that the sum over
    all inputs is unchanged.

    Note: The Dropout layer only applies when `training` is set to True in the forward pass.

    Args:
        rate (float): Fraction of the input units to drop. Should be between 0 and 1.
        seed (Optional[int]): Seed for the random number generator.
    """
    def __init__(self, rate: float, seed: int | None = None):
        super().__init__()
        if not 0 <= rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got: {rate}")
        self.rate = rate
        self.seed = seed
        # Training/evaluation mode is controlled by the 'training' flag in forward()

    def forward(self, x: tensor.EmberTensor, training: bool = True) -> tensor.EmberTensor:
        """
        Forward pass for Dropout.

        Args:
            x: Input tensor.
            training: Whether the layer should behave in training mode (apply dropout)
                      or inference mode (do nothing). Defaults to True.

        Returns:
            Output tensor after applying dropout.
        """
        if training and self.rate > 0:
            # In training mode with a non-zero rate, apply dropout
            # 1. Create a random bernoulli mask (1s where elements are kept)
            keep_prob = 1.0 - self.rate
            # Get the device from the input tensor x
            # Use getattr to safely access _device, default to None if not present
            device = getattr(x, '_device', None)

            # Generate the mask on the same device as the input tensor x
            mask = tensor.random_bernoulli(tensor.shape(x), p=keep_prob,
                                          dtype=x.dtype, device=device, seed=self.seed)
            # 2. Scale the result by 1 / (1 - rate) to preserve expectation
            # Avoid division by zero if rate is close to 1 (though rate < 1 is enforced in init)
            scale = 1.0 / keep_prob if keep_prob > 0 else 0.0
            # 3. Apply mask and scale
            return ops.multiply(ops.multiply(x, mask), scale)
        else:
            # In inference mode or if rate is 0, return the input unchanged
            return x

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the Dropout module."""
        config = super().get_config()
        config.update({
            "rate": self.rate,
            "seed": self.seed,
        })
        return config

    # from_config can rely on BaseModule implementation