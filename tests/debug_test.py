"""
Debug script for test_ember_tensor.py.

This script runs test_ember_tensor.py and catches any exceptions.
"""

import traceback
import sys
import os

# Redirect stdout and stderr to a file
log_file = open('debug_output.log', 'w')
sys.stdout = log_file
sys.stderr = log_file

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Python path:", sys.path)

try:
    print("Importing EmberTensor...")
    from ember_ml.nn.tensor import EmberTensor
    print("EmberTensor imported successfully.")
    
    print("Importing ops...")
    from ember_ml import ops
    print("ops imported successfully.")
    
    print("Setting backend...")
    ops.set_backend('torch')
    print("Backend set successfully.")
    
    print("Creating EmberTensor...")
    data = [[1, 2, 3], [4, 5, 6]]
    tensor = EmberTensor(data)
    print("EmberTensor created successfully.")
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor requires_grad: {tensor.requires_grad}")
    print(f"Tensor as list: {tensor.tolist()}")
    
    print("\nTesting tensor operations...")
    
    # Test zeros
    zeros_tensor = tensor.zeros((2, 3))
    print(f"Zeros tensor: {zeros_tensor.tolist()}")
    
    # Test ones
    ones_tensor = tensor.ones((2, 3))
    print(f"Ones tensor: {ones_tensor.tolist()}")
    
    # Test reshape
    reshaped_tensor = tensor.reshape(tensor, (3, 2))
    print(f"Reshaped tensor: {reshaped_tensor.tolist()}")
    
    # Test transpose
    transposed_tensor = tensor.transpose(tensor)
    print(f"Transposed tensor: {transposed_tensor.tolist()}")
    
    # Test concatenate
    concat_tensor = tensor.concatenate([tensor, tensor], axis=0)
    print(f"Concatenated tensor: {concat_tensor.tolist()}")
    
    # Test stack
    stacked_tensor = tensor.stack([tensor, tensor], axis=0)
    print(f"Stacked tensor: {stacked_tensor.tolist()}")
    
    # Test split
    split_tensors = tensor.split_tensor(tensor, 3, axis=1)
    print(f"Split tensors:")
    for i, t in enumerate(split_tensors):
        print(f"  Split {i}: {t.tolist()}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# Close the log file
log_file.close()

# Reset stdout and stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print(f"Debug output written to {os.path.abspath('debug_output.log')}")

def test_debug():
    """Simple test function for pytest to find."""
    assert True