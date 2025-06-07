"""
Test script to verify the import pattern works correctly.
"""

from ember_ml import ops
from ember_ml.nn import tensor

# Set the backend (optional, as it should auto-select)
ops.set_backend('numpy')  # or 'torch' or 'mlx'

# Test the stats.mean function
try:
    x = ops.stats.mean([1, 2, 3], axis=0)
    print(f"Success! Result: {x}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Print the type of the result to verify it's using the correct backend
print(f"Type of result: {type(x)}")