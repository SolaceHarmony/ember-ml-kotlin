import mlx.core as mx
from ember_ml.backend.mlx.tensor.tensor import MLXTensor

def test_mlx_tensor_slice():
    """Test the slice method in MLXTensor class."""
    print("=== Testing MLXTensor.slice ===")
    
    # Create an MLXTensor instance
    mlx_tensor = MLXTensor()
    
    # Create a test array
    arr = mx.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    print(f"Original array: {arr}")
    
    # Test slice with integer starts and sizes
    starts = [5]
    sizes = [10]
    sliced = mlx_tensor.slice(arr, starts, sizes)
    print(f"Sliced array (start=5, size=10): {sliced}")
    
    # Test slice with -1 size (to end)
    starts = [5]
    sizes = [-1]
    sliced_to_end = mlx_tensor.slice(arr, starts, sizes)
    print(f"Sliced array (start=5, size=-1): {sliced_to_end}")
    
    # Test multi-dimensional slice
    arr_2d = mx.reshape(arr, (4, 4))
    print(f"2D array:\n{arr_2d}")
    
    starts = [1, 1]
    sizes = [2, 2]
    sliced_2d = mlx_tensor.slice(arr_2d, starts, sizes)
    print(f"Sliced 2D array (starts=[1,1], sizes=[2,2]):\n{sliced_2d}")

if __name__ == "__main__":
    test_mlx_tensor_slice()