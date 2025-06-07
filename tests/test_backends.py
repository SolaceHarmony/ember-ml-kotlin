"""
Test script to verify EmberTensor works with different backends.
"""

from ember_ml.nn.tensor import EmberTensor, int32, float32
from ember_ml.backend import get_backend, set_backend

def print_tensor_info(tensor, backend_name):
    """Print information about a tensor with the current backend."""
    print(f"\n=== {backend_name} Backend ===")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor data: {tensor.numpy()}")
    print(f"Backend: {get_backend()}")

def test_backend(backend_name):
    """Test EmberTensor with a specific backend."""
    try:
        # Set the backend
        set_backend(backend_name)
        print(f"\nSwitched to {backend_name} backend")
        
        # Create tensors with different dtypes
        tensor_int = EmberTensor([1, 2, 3], dtype=int32)
        tensor_float = EmberTensor([1.0, 2.0, 3.0], dtype=float32)
        
        # Print tensor information
        print_tensor_info(tensor_int, backend_name)
        print_tensor_info(tensor_float, backend_name)
        
        # Test detach
        tensor_with_grad = EmberTensor([1, 2, 3], requires_grad=True)
        detached = tensor_with_grad.detach()
        print(f"requires_grad before detach: {tensor_with_grad.requires_grad}")
        print(f"requires_grad after detach: {detached.requires_grad}")
        
        # Test reshape
        reshaped = EmberTensor([[1, 2], [3, 4]])
        print(f"Original shape: {reshaped.shape}")
        tensor_obj = EmberTensor([0])
        reshaped = tensor_obj.reshape(reshaped.to_backend_tensor(), (4,))
        print(f"Reshaped shape: {reshaped.shape}")
        
        return True
    except Exception as e:
        print(f"Error with {backend_name} backend: {str(e)}")
        return False

# Test with all available backends
backends = ['numpy', 'torch', 'mlx']
results = {}

for backend in backends:
    print(f"\n{'='*50}")
    print(f"Testing with {backend} backend")
    print(f"{'='*50}")
    results[backend] = test_backend(backend)

# Print summary
print("\n\n=== Summary ===")
for backend, success in results.items():
    print(f"{backend}: {'Success' if success else 'Failed'}")