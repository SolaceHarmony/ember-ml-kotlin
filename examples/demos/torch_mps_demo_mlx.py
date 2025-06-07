"""
Demonstration of MLX backend on Apple Silicon.

This script shows how to use the Ember ML framework with the MLX backend,
which is optimized for Apple Silicon devices.
This version only tests MLX to avoid backend switching issues.
"""

import time
import platform
from typing import List

from ember_ml import ops
from ember_ml.ops import set_backend
from ember_ml.nn import tensor
set_backend('mlx')

def benchmark_matrix_multiply(
    sizes: List[int] = [1000, 2000, 4000]
) -> None:
    """Benchmark matrix multiplication with MLX.
    
    This function benchmarks matrix multiplication operations using the MLX
    backend. It creates random matrices of different sizes and measures the
    time taken to perform matrix multiplication.
    
    Args:
        sizes: List of matrix sizes to benchmark
    """
    print("\n--- Benchmarking MLX backend ---")
    
    for size in sizes:
        # Create random matrices
        a = tensor.random_normal(shape=(size, size))
        b = tensor.random_normal(shape=(size, size))
        
        # Print a sample of the matrices to prove they contain data
        if size <= 1000:  # Only for smaller matrices to avoid flooding the output
            print(f"  Matrix A sample: {tensor.item(a[0, 0]):.4f}, {tensor.item(a[0, 1]):.4f}, ...")
            print(f"  Matrix B sample: {tensor.item(b[0, 0]):.4f}, {tensor.item(b[0, 1]):.4f}, ...")
        
        # Warm-up
        result = ops.matmul(a, b)
        
        # Print a sample of the result to prove the computation happened
        if size <= 1000:  # Only for smaller matrices to avoid flooding the output
            print(f"  Result sample: {tensor.item(result[0, 0]):.4f}, {tensor.item(result[0, 1]):.4f}, ...")
        
        # Benchmark with more iterations for more accurate timing
        start_time = time.time()
        result = None
        for i in range(10):  # 10 iterations
            result = ops.matmul(a, b)
            # Print progress to show it's working
            # Only print for first and last iteration to avoid flooding
            # Note: Using Python operators for loop indices is acceptable as they're not tensor operations
            if i == 0 or i == 9:
                print(f"    Iteration {i+1}: Matrix multiplication completed")
        end_time = time.time()
        
        # Print a sample of the final result to prove the computation happened
        if size <= 1000 and result is not None:  # Only for smaller matrices to avoid flooding the output
            print(f"  Final result sample: {tensor.item(result[0, 0]):.4f}, {tensor.item(result[0, 1]):.4f}, ...")
        
        # Calculate average time using ops functions
        time_diff = ops.subtract(tensor.convert_to_tensor(end_time),
                                tensor.convert_to_tensor(start_time))
        avg_time = ops.divide(time_diff, tensor.convert_to_tensor(10.0))  # 10 iterations
        
        print(f"  Matrix size {size}x{size}: {avg_time.item():.8f} seconds")


def check_mlx_availability() -> bool:
    """Check if MLX is available.
    
    Returns:
        True if MLX is available, False otherwise
    """
    # MLX is typically available on Apple Silicon
    is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    
    # Try to set the backend to MLX
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


def main() -> None:
    """Main function to demonstrate MLX backend performance.
    
    This function benchmarks matrix multiplication operations using the MLX
    backend, which is optimized for Apple Silicon devices.
    """
    print("MLX Backend Performance Demonstration")
    print("=====================================")
    
    # Check if MLX is available
    mlx_available = check_mlx_availability()
    
    print(f"MLX available: {mlx_available}")
    
    if not mlx_available:
        print("\nMLX is not available on this system. It requires Apple Silicon (M1/M2/M3) hardware.")
        return

    # Use smaller matrix sizes for faster demonstration
    sizes = [500, 1000, 2000]
    
    # Benchmark MLX backend
    benchmark_matrix_multiply(sizes=sizes)
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()