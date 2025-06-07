"""
Test script for MLX tensor cast operation.

This script tests both the standalone cast() function and the MLXTensor.cast() method.
"""

import mlx.core as mx
from ember_ml.backend.mlx.tensor import MLXTensor, cast

# Create a test tensor
tensor = mx.array([1, 2, 3], dtype=mx.float32)

# Test the standalone cast() function
result1 = cast(tensor, 'float32')
print("Standalone function result dtype:", result1.dtype)

# Test the MLXTensor.cast() method
tensor_obj = MLXTensor()
result2 = tensor_obj.cast(tensor, 'float32')
print("Method result dtype:", result2.dtype)

# Verify that both approaches give the same result
print("Results are the same:", result1.dtype == result2.dtype)