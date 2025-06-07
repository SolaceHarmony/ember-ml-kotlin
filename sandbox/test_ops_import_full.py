"""
Test script to verify the import pattern works correctly.
"""

from ember_ml import ops
from ember_ml.nn import tensor

# Set the backend (optional, as it should auto-select)
ops.set_backend('numpy')  # or 'torch' or 'mlx'

# Test the stats.mean function with the requested pattern
try:
    # Create a 2D array to demonstrate axis parameter
    import numpy as np
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Input data:\n{data}")
    
    # Calculate mean along axis 0 (columns)
    x = ops.stats.mean(data, axis=0)
    print(f"Mean along axis 0: {x}")
    
    # Calculate mean along axis 1 (rows)
    y = ops.stats.mean(data, axis=1)
    print(f"Mean along axis 1: {y}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Print the type of the result to verify it's using the correct backend
print(f"Type of result: {type(x)}")