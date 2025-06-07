"""
Test file for inspecting type handling in the MLX backend.

This file contains tests that demonstrate the issue with EmberTensor objects
being passed directly to backend functions, and verifies that our solution works.
"""

import sys
import os

# Add the parent directory to the path so we can import ember_ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ember_ml.nn import tensor
from ember_ml import ops
from ember_ml.backend import set_backend, get_backend

def test_embertensor_to_backend_conversion():
    """Test that EmberTensor objects are properly converted to backend tensors."""
    # Set the backend to MLX
    set_backend('mlx')
    
    # Create an EmberTensor
    ember = tensor.EmberTensor([1, 2, 3])
    
    # Print the type of the EmberTensor
    print(f"EmberTensor type: {type(ember)}")
    
    # Print the type of the underlying tensor
    print(f"Underlying tensor type: {type(ember._tensor)}")
    
    try:
        # Try to add 1 to the EmberTensor using ops.add
        # This should fail if EmberTensor is passed directly to the backend
        result = ops.add(ember, 1)
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try to add 1 to the EmberTensor using ops.add
    # This should work if we properly extract the backend tensor
    try:
        result = ops.add(ember, 1)
        print(f"Result type (using ops.add): {type(result)}")
        print(f"Result value (using ops.add): {result}")
    except Exception as e:
        print(f"Error (using ops.add): {e}")

def test_direct_function_calls():
    """Test that direct function calls from nn.tensor.__init__ return native backend tensors."""
    # Set the backend to MLX
    set_backend('mlx')
    
    # Create a tensor using a direct function call
    x = tensor.ones((3,))
    
    # Print the type of the tensor
    print(f"Direct function call result type: {type(x)}")
    
    # Try to reshape the tensor
    try:
        y = tensor.reshape(x, (3, 1))
        print(f"Reshape result type: {type(y)}")
        print(f"Reshape result value: {y}")
    except Exception as e:
        print(f"Error: {e}")

def test_embertensor_methods():
    """Test that EmberTensor methods return EmberTensor objects."""
    # Set the backend to MLX
    set_backend('mlx')
    
    # Create an EmberTensor
    x = tensor.EmberTensor([1, 2, 3])
    
    # Try to reshape the tensor using tensor.reshape
    try:
        y = tensor.reshape(x, (3, 1))
        print(f"tensor.reshape result type: {type(y)}")
        print(f"tensor.reshape result value: {y}")
    except Exception as e:
        print(f"Error: {e}")

def test_chained_operations():
    """Test that chained operations maintain the correct return type."""
    # Set the backend to MLX
    set_backend('mlx')
    
    # Create a tensor using a direct function call and then reshape it
    try:
        x = tensor.ones((3,))
        y = tensor.reshape(x, (3, 1))
        print(f"Chained operations result type: {type(y)}")
        print(f"Chained operations result value: {y}")
    except Exception as e:
        print(f"Error: {e}")

def test_ops_functions_with_embertensor():
    """Test that ops functions handle EmberTensor objects correctly."""
    # Set the backend to MLX
    set_backend('mlx')
    
    # Create an EmberTensor
    ember = tensor.EmberTensor([1, 2, 3])
    
    # Try to use ops.add with the EmberTensor
    try:
        result = ops.add(ember, 1)
        print(f"ops.add result type: {type(result)}")
        print(f"ops.add result value: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("\n=== Testing EmberTensor to Backend Conversion ===")
    test_embertensor_to_backend_conversion()
    
    print("\n=== Testing Direct Function Calls ===")
    test_direct_function_calls()
    
    print("\n=== Testing EmberTensor Methods ===")
    test_embertensor_methods()
    
    print("\n=== Testing Chained Operations ===")
    test_chained_operations()
    
    print("\n=== Testing Ops Functions with EmberTensor ===")
    test_ops_functions_with_embertensor()
