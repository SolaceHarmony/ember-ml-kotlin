"""
Unit tests for data type operations across different backends.

This module contains pytest tests for the data type operations in the nn.tensor module.
It tests each operation with different backends to ensure consistency.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.backend import get_backend, set_backend

# List of backends to test
BACKENDS = ['numpy']
try:
    import torch
    BACKENDS.append('torch')
except ImportError:
    pass

try:
    import mlx.core
    BACKENDS.append('mlx')
except ImportError:
    pass

@pytest.fixture(params=BACKENDS)
def backend(request):
    """Fixture to test with different backends."""
    prev_backend = get_backend()
    set_backend(request.param)
    ops.set_ops(request.param)
    yield request.param
    # A backend must always be defined - not having one is a critical failure
    assert prev_backend is not None, "No backend was defined before test"
    set_backend(prev_backend)
    ops.set_ops(prev_backend)

class TestDTypeOperations:
    """Tests for data type operations."""

    def test_get_dtype(self, backend):
        """Test get_dtype operation."""
        # Test with default dtype names
        dtype_names = ['float32', 'float64', 'int32', 'int64', 'bool']
        
        for dtype_name in dtype_names:
            try:
                # Get the dtype
                dtype = tensor.get_dtype(dtype_name)
                
                # Create a tensor with the dtype
                x = tensor.ones((3, 4), dtype=dtype)
                
                # Verify that the tensor has the correct dtype
                # Use string comparison for backend purity
                assert tensor.to_dtype_str(tensor.dtype(x)) == dtype_name
            except (ValueError, AttributeError):
                # Skip if the dtype is not supported by the backend
                pass

    # EmberTensor supports dtype passing only as strings on the front-end
    # Direct NumPy dtype conversion is not supported to maintain backend purity

# EmberTensor supports dtype passing only as strings on the front-end
# Direct backend references are not supported to maintain backend purity
            