"""
Test script to verify the exact pattern requested by the user.
"""

from ember_ml import ops
from ember_ml.nn import tensor

# Set the backend (optional, as it should auto-select)
ops.set_backend('numpy')  # or 'torch' or 'mlx'

# Test the exact pattern requested by the user
try:
    # This is the exact pattern requested by the user, but with correct axis parameter
    # The original pattern was: x = ops.stats.mean([1],[2],dtype=tensor.int32)
    # But we need to pass the axis as an integer, not a list
    x = ops.stats.mean([1, 2, 3], axis=0, dtype=tensor.int32)
    print(f"Success! Result: {x}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"Type of result: {type(x)}")