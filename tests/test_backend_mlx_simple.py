import mlx.core as mx

# Create a simple array
arr = mx.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a slice
start = 2
size = 5
end = start + size

# Print the slice
print(f"Original array: {arr}")
print(f"Slice (start={start}, end={end}): {arr[start:end]}")

# Try with MLX arrays
start_arr = mx.array(2)
size_arr = mx.array(5)
end_arr = mx.add(start_arr, size_arr)

# Extract as Python values
start_val = start_arr.item()
end_val = end_arr.item()

# Create slice with Python values
print(f"start_val: {start_val}, type: {type(start_val)}")
print(f"end_val: {end_val}, type: {type(end_val)}")
print(f"Slice with Python values: {arr[start_val:end_val]}")