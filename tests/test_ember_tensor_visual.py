"""
Visual confirmation test for EmberTensor properties.
"""

from ember_ml.nn import tensor
from ember_ml import ops


def test_ember_tensor_properties():
    """Print EmberTensor properties for visual confirmation."""
    # Test with each backend
    for backend_name in ["numpy", "torch", "mlx"]:
        print(f"\n=== Testing with {backend_name} backend ===")
        
        # Set the backend
        ops.set_backend(backend_name)
        
        # Create a tensor
        t = tensor.EmberTensor([1, 2, 3], dtype=tensor.float32)
        
        # Print tensor information
        print(f"Tensor: {t}")
        print(f"Internal tensor representation type: {type(t._tensor)}")
        print(f"Internal dtype (_dtype): {t._dtype}")
        print(f"Backend: {t.backend}")
        print(f"dtype property: {t.dtype}")
        
        # Verify that dtype property uses _dtype
        if str(t.dtype) == str(t._dtype):
            print("✅ dtype property correctly uses _dtype value")
        else:
            print(f"❌ dtype property ({t.dtype}) doesn't match _dtype ({t._dtype})")


if __name__ == "__main__":
    test_ember_tensor_properties()
    
    # Reset to numpy backend at the end
    ops.set_backend("numpy")