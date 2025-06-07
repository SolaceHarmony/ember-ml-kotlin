"""
Debug script to print EmberTensor properties for all backends.
"""

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.backend import set_backend


def print_tensor_properties():
    """Print detailed EmberTensor properties for all backends."""
    backends = ["numpy", "torch", "mlx"]
    
    for backend_name in backends:
        print(f"\n{'='*50}")
        print(f"TESTING WITH {backend_name.upper()} BACKEND")
        print(f"{'='*50}")
        
        # Set the backend
        set_backend(backend_name)
        
        # Create a tensor
        t = tensor.EmberTensor([1, 2, 3], dtype=tensor.float32)
        
        # Print basic tensor info
        print(f"Tensor: {t}")
        print(f"Type: {type(t)}")
        print(f"Shape: {t.shape}")
        
        # Print internal representation
        print(f"\nInternal tensor representation:")
        print(f"  Type: {type(t._tensor)}")
        print(f"  Value: {t._tensor}")
        
        # Print dtype information
        print(f"\nDtype information:")
        print(f"  Internal _dtype: {t._dtype}")
        print(f"  dtype property: {t.dtype}")
        print(f"  Type of dtype: {type(t.dtype)}")
        
        # Print backend information
        print(f"\nBackend information:")
        print(f"  backend property: {t.backend}")
        
        # Print device information
        print(f"\nDevice information:")
        print(f"  device property: {t.device}")
        
        # Test iteration
        print(f"\nIteration test:")
        print(f"  List from iteration: {list(t)}")
        
        # Test serialization
        print(f"\nSerialization test:")
        state = t.__getstate__()
        print(f"  __getstate__ returns: {state}")
        print(f"  Keys in state: {state.keys()}")
        
        # Test conversion to numpy
        print(f"\nNumPy conversion:")
        numpy_array = t.to_numpy()
        print(f"  numpy() returns: {numpy_array}")
        print(f"  Type: {type(numpy_array)}")
        
        # Test convert_to_tensor
        print(f"\nconvert_to_tensor test:")
        converted = tensor.convert_to_tensor(t)
        print(f"  Is same object: {converted is t}")
        print(f"  Type: {type(converted)}")
        
        # Test _convert_to_backend_tensor
        print(f"\n_convert_to_backend_tensor test:")
        from ember_ml.nn.tensor.common import _convert_to_backend_tensor
        backend_tensor = _convert_to_backend_tensor([4, 5, 6])
        print(f"  Type: {type(backend_tensor)}")
        print(f"  Value: {backend_tensor}")
        
        print(f"\n{'-'*50}")


if __name__ == "__main__":
    print_tensor_properties()
    
    # Reset to numpy backend at the end
    set_backend("numpy")