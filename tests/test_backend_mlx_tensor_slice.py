import mlx.core as mx
import numpy as np

def test_slice_creation():
    """Test slice creation with MLX arrays."""
    print("=== Testing Slice Creation ===")
    
    # Create MLX arrays with explicit int32 dtype
    start = mx.array(5, dtype=mx.int32)
    size = mx.array(10, dtype=mx.int32)
    
    print(f"start: {start}, shape: {start.shape}, dtype: {start.dtype}")
    print(f"size: {size}, shape: {size.shape}, dtype: {size.dtype}")
    
    # Extract values
    start_val = start.item()
    
    # Compute end value using MLX add
    end_array = mx.add(start, size)
    end_val = end_array.item()
    
    print(f"start_val: {start_val}, type: {type(start_val)}")
    print(f"end_val: {end_val}, type: {type(end_val)}")
    
    # Create slice with Python values
    s = slice(start_val, end_val)
    print(f"Slice created: {s}")
    
    # Test with a real array
    arr = mx.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    sliced = arr[s]
    print(f"Original array: {arr}")
    print(f"Sliced array: {sliced}")

if __name__ == "__main__":
    test_slice_creation()