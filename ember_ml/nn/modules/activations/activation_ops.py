# ember_ml/nn/modules/activations/ops/activation_ops.py
import abc
from typing import Any, Optional

# Placeholder for TensorLike and DType - ideally import from a shared types module
TensorLike = Any
DType = Any

class ActivationOps(abc.ABC):
    """Abstract base class for backend-specific activation function implementations."""

    @abc.abstractmethod
    def relu(self, x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> TensorLike:
        """
        Apply Rectified Linear Unit activation.
        
        Args:
            x: Input tensor
            dtype: Optional output data type
            device: Optional device specification
            
        Returns:
            Output tensor with ReLU activation applied
        """
        pass

    @abc.abstractmethod
    def sigmoid(self, x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> TensorLike:
        """
        Apply Sigmoid activation.
        
        Args:
            x: Input tensor
            dtype: Optional output data type
            device: Optional device specification
            
        Returns:
            Output tensor with Sigmoid activation applied
        """
        pass

    @abc.abstractmethod
    def tanh(self, x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> TensorLike:
        """
        Apply Hyperbolic Tangent activation.
        
        Args:
            x: Input tensor
            dtype: Optional output data type
            device: Optional device specification
            
        Returns:
            Output tensor with Tanh activation applied
        """
        pass

    @abc.abstractmethod
    def softmax(self, x: TensorLike, axis: int = -1, dtype: Optional[DType] = None, device: Optional[str] = None) -> TensorLike:
        """
        Apply Softmax activation.
        
        Args:
            x: Input tensor
            axis: Axis along which to compute softmax, default is -1 (last dimension)
            dtype: Optional output data type
            device: Optional device specification
            
        Returns:
            Output tensor with Softmax activation applied
        """
        pass

    @abc.abstractmethod
    def softplus(self, x: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> TensorLike:
        """
        Apply Softplus activation.
        
        Args:
            x: Input tensor
            dtype: Optional output data type
            device: Optional device specification
            
        Returns:
            Output tensor with Softplus activation applied
        """
        pass

    # Potentially add others like leaky_relu, elu, etc. if needed