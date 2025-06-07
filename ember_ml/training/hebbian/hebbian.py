"""
Hebbian learning implementation.

This module provides the HebbianLayer class that implements Hebbian learning
for adapting connection weights based on correlated activity between
input and output units.
"""

from typing import Tuple

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops import stats


class HebbianLayer:
    """
    Implements a layer with Hebbian learning capabilities.
    
    This layer adapts its weights based on correlated activity between
    input and output units following Hebb's rule: "Neurons that fire together,
    wire together."
    
    Attributes:
        weights (NDArray[tensor.float32]): Connection weight matrix
        eta (float): Learning rate for weight updates
        input_size (int): Number of input units
        output_size (int): Number of output units
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 eta: float = 0.01,
                 weight_scale: float = 0.01):
        """
        Initialize a Hebbian learning layer.
        
        Args:
            input_size: Number of input units
            output_size: Number of output units
            eta: Learning rate for weight updates
            weight_scale: Scale factor for initial weight values
        """
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta

        # Initialize small random weights
        self.weights = tensor.random_normal(
            (output_size, input_size),
            mean=0.0,
            stddev=weight_scale,
        )
        
        # Keep track of weight statistics
        self._weight_history: list[Tuple[float, float]] = []  # (mean, std)
    
    def forward(self, inputs: tensor.EmberTensor) -> tensor.EmberTensor:
        """
        Compute forward pass through the layer.
        
        Args:
            inputs: Input activity vector (input_size,)
            
        Returns:
            NDArray[tensor.float32]: Output activity vector (output_size,)
            
        Raises:
            ValueError: If input shape doesn't match input_size
        """
        inputs_t = tensor.convert_to_tensor(inputs)
        if tensor.shape(inputs_t) != (self.input_size,):
            raise ValueError(
                f"Expected input shape ({self.input_size},), "
                f"got {tensor.shape(inputs_t)}"
            )

        return ops.matmul(self.weights, inputs_t)
    
    def hebbian_update(
        self,
        inputs: tensor.EmberTensor,
        outputs: tensor.EmberTensor,
    ) -> None:
        """
        Update weights using Hebbian learning rule.
        
        The update follows the rule:
        Δw = η * (post-synaptic activity ⊗ pre-synaptic activity)
        where ⊗ is the outer product.
        
        Args:
            inputs: Pre-synaptic activity vector (input_size,)
            outputs: Post-synaptic activity vector (output_size,)
            
        Raises:
            ValueError: If input/output shapes don't match layer dimensions
        """
        inputs_t = tensor.convert_to_tensor(inputs)
        outputs_t = tensor.convert_to_tensor(outputs)
        if tensor.shape(inputs_t) != (self.input_size,):
            raise ValueError(
                f"Expected input shape ({self.input_size},), "
                f"got {tensor.shape(inputs_t)}"
            )
        if tensor.shape(outputs_t) != (self.output_size,):
            raise ValueError(
                f"Expected output shape ({self.output_size},), "
                f"got {tensor.shape(outputs_t)}"
            )

        # Compute weight updates using outer product
        inputs_row = tensor.reshape(inputs_t, (1, self.input_size))
        outputs_col = tensor.reshape(outputs_t, (self.output_size, 1))
        delta_w = ops.multiply(
            self.eta,
            ops.matmul(outputs_col, inputs_row),
        )

        # Apply updates
        self.weights = ops.add(self.weights, delta_w)
        
        # Record weight statistics
        self._weight_history.append(
            (float(stats.mean(self.weights)), float(stats.std(self.weights)))
        )
    
    def get_weight_stats(self) -> list[Tuple[float, float]]:
        """
        Get history of weight statistics.
        
        Returns:
            list[Tuple[float, float]]: List of (mean, std) pairs for weights
        """
        return self._weight_history.copy()
    
    def reset_weights(self, weight_scale: float = 0.01) -> None:
        """
        Reset weights to random values.
        
        Args:
            weight_scale: Scale factor for new random weights
        """
        self.weights = tensor.random_normal(
            (self.output_size, self.input_size),
            mean=0.0,
            stddev=weight_scale,
        )
        self._weight_history.clear()
    
    def get_weights(self) -> tensor.EmberTensor:
        """
        Get the current weight matrix.
        
        Returns:
            NDArray[tensor.float32]: Copy of weight matrix
        """
        return tensor.copy(self.weights)
