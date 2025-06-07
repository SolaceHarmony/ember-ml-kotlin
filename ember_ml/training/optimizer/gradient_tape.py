"""
Gradient Tape implementation for automatic differentiation.

This module provides a backend-agnostic Gradient Tape for Ember ML that works
with any backend using the frontend abstraction layer.
"""

from typing import Any, List, Optional, Union, Tuple, Dict, Set
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Parameter

class GradientTape:
    """
    Gradient Tape for automatic differentiation.

    This context manager records operations for automatic differentiation.
    Operations are recorded if they are executed within this context manager and
    at least one of their inputs is being "watched".

    Trainable parameters (created by Parameter) are automatically watched.
    Tensors can be manually watched by invoking the `watch` method on this context
    manager.

    Example:
    ```python
    x = tensor.convert_to_tensor([1.0, 2.0, 3.0])
    with GradientTape() as tape:
        tape.watch(x)
        y = ops.multiply(x, x)
    grad = tape.gradient(y, x)  # [2.0, 4.0, 6.0]
    ```
    """
    def __init__(self, persistent: bool = False, watch_accessed_variables: bool = True):
        """
        Initialize the GradientTape.

        Args:
            persistent: Boolean controlling whether a persistent gradient tape
                is created. False by default, which means at most one call can
                be made to the gradient() method on this object.
            watch_accessed_variables: Boolean controlling whether the tape will
                automatically watch any trainable parameters accessed while the tape
                is active. Defaults to True.
        """
        self._recording = False
        self._persistent = persistent
        self._watch_accessed_variables = watch_accessed_variables
        self._watched_tensors: Set[Any] = set()
        self._used = False

    def __enter__(self):
        """Enter the context manager and start recording."""
        self._recording = True
        # The actual recording mechanism is handled by the backend
        # through the ops abstraction layer
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and stop recording."""
        self._recording = False

    def watch(self, tensor_or_variable: Any):
        """
        Mark a tensor to be watched for gradient computation.

        Args:
            tensor_or_variable: The tensor or variable to watch.
        """
        # Add the tensor to our set of watched tensors
        if isinstance(tensor_or_variable, (list, tuple)):
            for t in tensor_or_variable:
                self._watched_tensors.add(t)
        else:
            self._watched_tensors.add(tensor_or_variable)

    def watched_variables(self) -> List[Any]:
        """
        Returns variables watched by this tape in order of construction.

        Returns:
            A list of variables being watched by this tape.
        """
        # Filter for Parameter objects
        return [t for t in self._watched_tensors if isinstance(t, Parameter)]

    def gradient(self, target: Any, sources: Union[Any, List[Any]], 
                 unconnected_gradients: str = "none") -> Union[Any, List[Any]]:
        """
        Compute the gradients of the target with respect to the sources.

        Args:
            target: The tensor to differentiate (output of the computation).
            sources: A tensor or list of tensors to differentiate with respect to (inputs to the computation).
            unconnected_gradients: Specifies behavior when target is not connected to sources.
                Valid values are "none" (default) or "zero".

        Returns:
            A tensor or list of tensors representing the gradients of target with respect to each source.

        Raises:
            RuntimeError: If called on a used, non-persistent tape.
            ValueError: If unconnected_gradients is not "none" or "zero".
        """
        if not self._persistent and self._used:
            raise RuntimeError("A non-persistent GradientTape can only be used to "
                              "compute one set of gradients")
        
        # Mark the tape as used
        self._used = True
        
        # Validate unconnected_gradients
        if unconnected_gradients not in ["none", "zero"]:
            raise ValueError(f"Unconnected gradients must be 'none' or 'zero', got {unconnected_gradients}")
        
        # Convert target to tensor if needed
        if not isinstance(target, tensor.EmberTensor):
            target = tensor.convert_to_tensor(target)
        
        # Handle single source case
        single_source = not isinstance(sources, (list, tuple))
        if single_source:
            sources = [sources]
        
        # Convert sources to tensors if needed
        sources = [s if isinstance(s, tensor.EmberTensor) else tensor.convert_to_tensor(s) for s in sources]
        
        # Compute gradients using the ops abstraction layer
        # This delegates to the backend-specific implementation
        try:
            # Use ops.gradient to compute gradients
            # This function should be implemented in the backend to handle automatic differentiation
            gradients = ops.gradient(target, sources)
            
            # Handle unconnected gradients
            if unconnected_gradients == "zero":
                gradients = [g if g is not None else tensor.zeros_like(s) for g, s in zip(gradients, sources)]
        except Exception as e:
            raise RuntimeError(f"Error computing gradients: {e}")
        
        # Return single gradient if single source
        if single_source:
            return gradients[0]
        
        return gradients

    def jacobian(self, target: Any, source: Any, 
                 unconnected_gradients: str = "none") -> Any:
        """
        Computes the jacobian of target with respect to source.

        The jacobian is the matrix of all first-order partial derivatives of a vector-valued
        function with respect to its inputs.

        Args:
            target: A tensor to be differentiated.
            source: A tensor with respect to which the differentiation will take place.
            unconnected_gradients: Specifies behavior when target is not connected to source.
                Valid values are "none" (default) or "zero".

        Returns:
            A tensor containing the jacobian of the target with respect to the source.

        Raises:
            RuntimeError: If called on a used, non-persistent tape.
            ValueError: If unconnected_gradients is not "none" or "zero".
        """
        if not self._persistent and self._used:
            raise RuntimeError("A non-persistent GradientTape can only be used to "
                              "compute one set of gradients or jacobians")
        
        # Mark the tape as used
        self._used = True
        
        # Validate unconnected_gradients
        if unconnected_gradients not in ["none", "zero"]:
            raise ValueError(f"Unconnected gradients must be 'none' or 'zero', got {unconnected_gradients}")
        
        # Convert tensors if needed
        if not isinstance(target, tensor.EmberTensor):
            target = tensor.convert_to_tensor(target)
        
        if not isinstance(source, tensor.EmberTensor):
            source = tensor.convert_to_tensor(source)
        
        # Get shapes
        target_shape = tensor.shape(target)
        source_shape = tensor.shape(source)
        
        # Flatten target if it's not already a vector
        flat_target = tensor.reshape(target, [-1])
        flat_target_size = tensor.shape(flat_target)[0]
        
        # Initialize jacobian matrix
        jacobian_shape = target_shape + source_shape
        
        # Compute jacobian by taking gradients of each output element with respect to all inputs
        # This is a simple implementation that computes one gradient per output element
        jacobian_rows = []
        
        for i in range(flat_target_size):
            # Extract the i-th element of the flattened target
            # We need to use ops functions for slicing and indexing
            target_element = ops.gather(flat_target, tensor.convert_to_tensor([i]))
            
            # Compute gradient of this element with respect to source
            grad = ops.gradient(target_element, [source])[0]
            
            # Handle unconnected gradients
            if grad is None:
                if unconnected_gradients == "zero":
                    grad = tensor.zeros_like(source)
                else:
                    # For "none", we still need a placeholder in the jacobian
                    # Use zeros but this could be handled differently
                    grad = tensor.zeros_like(source)
            
            # Reshape gradient to match source shape
            grad_flat = tensor.reshape(grad, [-1])
            jacobian_rows.append(grad_flat)
        
        # Stack the gradients to form the jacobian
        jacobian = tensor.stack(jacobian_rows)
        
        # Reshape to the expected output shape
        jacobian = tensor.reshape(jacobian, jacobian_shape)
        
        return jacobian

    def batch_jacobian(self, target: Any, source: Any,
                      unconnected_gradients: str = "none") -> Any:
        """
        Computes and stacks per-example jacobians.

        This function is useful when target[i,...] is independent of source[j,...] for j != i.
        This assumption allows more efficient computation compared to jacobian().

        Args:
            target: A tensor with shape [batch_size, ...] to be differentiated.
            source: A tensor with shape [batch_size, ...] with respect to which the 
                   differentiation will take place.
            unconnected_gradients: Specifies behavior when target is not connected to source.
                Valid values are "none" (default) or "zero".

        Returns:
            A tensor containing the batch jacobian of the target with respect to the source.

        Raises:
            RuntimeError: If called on a used, non-persistent tape.
            ValueError: If unconnected_gradients is not "none" or "zero".
            ValueError: If batch dimensions don't match.
        """
        if not self._persistent and self._used:
            raise RuntimeError("A non-persistent GradientTape can only be used to "
                              "compute one set of gradients or jacobians")
        
        # Mark the tape as used
        self._used = True
        
        # Validate unconnected_gradients
        if unconnected_gradients not in ["none", "zero"]:
            raise ValueError(f"Unconnected gradients must be 'none' or 'zero', got {unconnected_gradients}")
        
        # Convert tensors if needed
        if not isinstance(target, tensor.EmberTensor):
            target = tensor.convert_to_tensor(target)
        
        if not isinstance(source, tensor.EmberTensor):
            source = tensor.convert_to_tensor(source)
        
        # Get shapes
        target_shape = tensor.shape(target)
        source_shape = tensor.shape(source)
        
        # Ensure batch dimensions match
        if target_shape[0] != source_shape[0]:
            raise ValueError(f"Batch dimensions must match. Got {target_shape[0]} and {source_shape[0]}")
        
        batch_size = target_shape[0]
        
        # Flatten target and source along non-batch dimensions
        target_size = 1
        for dim in target_shape[1:]:
            target_size *= dim
        
        source_size = 1
        for dim in source_shape[1:]:
            source_size *= dim
        
        # Reshape target and source
        flat_target = tensor.reshape(target, [batch_size, target_size])
        flat_source = tensor.reshape(source, [batch_size, source_size])
        
        # Initialize batch jacobian
        batch_jacobian_shape = [batch_size, target_size, source_size]
        batch_jacobian = []
        
        # Compute jacobian for each example in the batch
        for i in range(batch_size):
            # Extract the i-th example
            target_i = ops.gather(flat_target, tensor.convert_to_tensor([i]))
            source_i = ops.gather(flat_source, tensor.convert_to_tensor([i]))
            
            # Compute jacobian for this example
            with GradientTape(persistent=True) as tape_i:
                tape_i.watch(source_i)
                
                # Compute gradients for each output element
                jacobian_rows = []
                for j in range(target_size):
                    # Extract the j-th element of the target
                    target_ij = ops.gather(target_i, tensor.convert_to_tensor([j]))
                    
                    # Compute gradient
                    grad = tape_i.gradient(target_ij, source_i, unconnected_gradients=unconnected_gradients)
                    
                    # Reshape gradient
                    grad_flat = tensor.reshape(grad, [-1])
                    jacobian_rows.append(grad_flat)
                
                # Stack the gradients to form the jacobian for this example
                jacobian_i = tensor.stack(jacobian_rows)
            
            batch_jacobian.append(jacobian_i)
        
        # Stack the jacobians for all examples
        result = tensor.stack(batch_jacobian)
        
        # Reshape to the expected output shape
        result_shape = [batch_size] + list(target_shape[1:]) + list(source_shape[1:])
        result = tensor.reshape(result, result_shape)
        
        return result