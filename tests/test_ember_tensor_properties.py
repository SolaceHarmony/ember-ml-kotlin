"""
Test EmberTensor properties to ensure proper storage and access of dtype and backend.
"""

import pytest
from ember_ml.nn import tensor
from ember_ml.backend import set_backend, get_backend


@pytest.mark.parametrize("backend", ["numpy", "torch", "mlx"])
def test_ember_tensor_backend_property(backend):
    """Test that EmberTensor correctly stores and returns backend information."""
    # Set the backend
    original_backend = get_backend()
    set_backend(backend)
    
    try:
        # Create a tensor
        t = tensor.EmberTensor([1, 2, 3], dtype=tensor.float32)
        
        # Check that _tensor is created with the correct backend implementation
        assert t._tensor is not None, "Internal tensor representation should not be None"
        
        # Check that _dtype is stored correctly
        assert hasattr(t, '_dtype'), "EmberTensor should have _dtype attribute"
        assert t._dtype is not None, "_dtype should not be None"
        assert str(t._dtype) == "float32", f"Expected dtype float32, got {t._dtype}"
        
        # Check that backend property returns the correct value
        assert t.backend == backend, f"backend property should return {backend}, got {t.backend}"
        
        # Check that dtype property returns the correct value
        assert str(t.dtype) == "float32", f"dtype property should return float32, got {t.dtype}"
    
    finally:
        # Restore the original backend
        if original_backend is not None:
            set_backend(original_backend)
        else:
            # Default to numpy if original_backend was None
            set_backend("numpy")


if __name__ == "__main__":
    # Run the test with the numpy backend
    test_ember_tensor_backend_property("numpy")
    print("All tests passed!")