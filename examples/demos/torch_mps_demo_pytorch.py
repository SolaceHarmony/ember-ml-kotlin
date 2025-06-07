"""
Demonstration of PyTorch's MPS (Metal Performance Shaders) backend on Apple Silicon.

This script shows how to use the Ember ML framework with the PyTorch backend,
including MPS acceleration for matrix operations on Apple Silicon devices.
This version only tests PyTorch to avoid backend switching issues.
"""

import time
import platform
from typing import List, Optional, Dict

# Import backend module and set the backend to PyTorch
from ember_ml.ops import set_backend, get_backend
from ember_ml.nn import tensor
# Set the backend to PyTorch and save it to the .ember/backend file
# This ensures the backend persists across runs
set_backend('torch')

# Now import ops
from ember_ml import ops


def benchmark_matrix_multiply(
    device: Optional[str] = None, 
    sizes: List[int] = [1000, 2000, 4000]
) -> None:
    """Benchmark matrix multiplication with PyTorch on different devices.
    
    This function benchmarks matrix multiplication operations using the PyTorch
    backend on the specified device. It creates random matrices of different sizes
    and measures the time taken to perform matrix multiplication.
    
    Args:
        device: The device to use (e.g., 'cpu', 'cuda', 'mps', or None)
        sizes: List of matrix sizes to benchmark
    """
    # Create the message using f-strings
    device_info = f" on {device}" if device else ""
    message = f"--- Benchmarking PyTorch backend{device_info} ---"
    print(f"\n{message}")
    
    for size in sizes:
        # Create random matrices
        a = tensor.random_normal(shape=(size, size), device=device)
        b = tensor.random_normal(shape=(size, size), device=device)
        
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


def check_pytorch_devices() -> Dict[str, bool]:
    """Check which PyTorch devices are available.
    
    Returns:
        Dictionary with availability information for different PyTorch devices
    """
    availability = {
        'cpu': True,
        'cuda': False,
        'mps': False
    }
    
    # Check for CUDA and MPS using ops.get_available_devices
    devices = ops.get_available_devices()
    availability['cuda'] = 'cuda' in devices or any(d.startswith('cuda:') for d in devices)
    availability['mps'] = 'mps' in devices
    
    # For Apple Silicon, assume MPS might be available if we're on macOS with Apple Silicon
    if not availability['mps'] and platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # We'll set this to True and let the benchmark function handle any errors
        availability['mps'] = True
    
    return availability


def main() -> None:
    """Main function to demonstrate PyTorch backend performance.
    
    This function benchmarks matrix multiplication operations using the PyTorch
    backend on different devices (CPU, MPS, CUDA) if available.
    """
    print("PyTorch Backend Performance Comparison")
    print("======================================")
    
    # Verify the backend is set correctly
    print(f"Current backend: {get_backend()}")
    
    # Check available devices
    devices = check_pytorch_devices()
    
    print(f"PyTorch CPU available: {devices['cpu']}")
    print(f"PyTorch CUDA available: {devices['cuda']}")
    print(f"PyTorch MPS available: {devices['mps']}")
    
    # Use smaller matrix sizes for faster demonstration
    sizes = [500, 1000, 2000]
    
    # Benchmark PyTorch backend on CPU
    benchmark_matrix_multiply(device='cpu', sizes=sizes)
    
    # Benchmark PyTorch backend on MPS if available
    if devices['mps']:
        try:
            benchmark_matrix_multiply(device='mps', sizes=sizes)
        except Exception as e:
            print(f"\nError running PyTorch on MPS: {e}")
    
    # Benchmark PyTorch backend on CUDA if available
    if devices['cuda']:
        try:
            benchmark_matrix_multiply(device='cuda', sizes=sizes)
        except Exception as e:
            print(f"\nError running PyTorch on CUDA: {e}")
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()