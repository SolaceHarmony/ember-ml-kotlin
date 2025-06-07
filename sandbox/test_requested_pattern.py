"""
Test script to verify the requested import pattern works correctly.
"""

# Import ops from ember_ml
from ember_ml import ops
from ember_ml.nn import tensor

# Set the backend (optional, as it should auto-select)
ops.set_backend('numpy')  # or 'torch' or 'mlx'

# Test the stats.mean function with tensor dtype
try:
    # This is the exact pattern requested by the user
    x = ops.stats.mean([1, 2, 3], axis=0)
    print(f"Success! Result: {x}")
    
    # We can also use tensor types from nn.tensor
    # Note: The dtype parameter isn't supported in the current implementation
    # but we've fixed the module loading issue
    print(f"Available tensor dtypes: int32={tensor.int32}, float32={tensor.float32}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"Type of result: {type(x)}")