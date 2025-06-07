"""
Simple test script for MLX tensor cast operation.

This script tests the standalone cast() function directly.
"""

import mlx.core as mx
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Import the MLXTensor class
from ember_ml.backend.mlx.tensor.tensor import MLXTensor
from ember_ml.backend.mlx.tensor.dtype import MLXDType

# Import the cast function and _validate_dtype function directly
from ember_ml.backend.mlx.tensor.ops.casting import cast, _validate_dtype

# Create a test tensor
tensor = mx.array([1, 2, 3], dtype=mx.float32)
print("Original tensor dtype:", tensor.dtype)

# Test the standalone cast() function
result = cast(tensor, 'float32')
print("Cast result dtype:", result.dtype)

# Verify that the dtype changed
print("Dtype changed:", tensor.dtype != result.dtype)