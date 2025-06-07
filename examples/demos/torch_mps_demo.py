"""
Demonstration of PyTorch's MPS (Metal Performance Shaders) backend on Apple Silicon.

This script shows how to use the Ember ML framework with different backends,
including PyTorch with MPS acceleration for matrix operations on Apple Silicon devices.
"""

import time
import platform
from typing import List, Optional, Dict

from ember_ml import ops
from ember_ml.ops import set_backend, auto_select_backend
from ember_ml.nn import tensor

def benchmark_matrix_multiply(
    backend: str, 
    device: Optional[str] = None, 
    sizes: List[int] = [1000, 2000, 4000]
) -> None:
    """Benchmark matrix multiplication with different backends and devices.
    
    This function benchmarks matrix multiplication operations using the specified
    backend and device. It creates random matrices of different sizes and measures
    the time taken to perform matrix multiplication.
    
    Args:
        backend: The backend to use ('numpy', 'torch', or 'mlx')
        device: The device to use (e.g., 'cpu', 'cuda', 'mps', or None)
        sizes: List of matrix sizes to benchmark
    """
    # Create the message using f-strings instead of + operator
    device_info = f" on {device}" if device else ""
    message = f"--- Benchmarking {backend} backend{device_info} ---"
    print(f"\n{message}")
    
    # Set the backend
    set_backend(backend)
    
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
        for i in range(10):  # Increased from 3 to 10 iterations
            result = ops.matmul(a, b)
            # Print progress to show it's working
            if i == 0 or i == 9:  # Only print for first and last iteration to avoid flooding
                print(f"    Iteration {i+1}: Matrix multiplication completed")
        end_time = time.time()
        
        # Print a sample of the final result to prove the computation happened
        if size <= 1000 and result is not None:  # Only for smaller matrices to avoid flooding the output
            print(f"  Final result sample: {tensor.item(result[0, 0]):.4f}, {tensor.item(result[0, 1]):.4f}, ...")
        
        # Calculate average time using ops.divide
        time_diff = ops.subtract(tensor.convert_to_tensor(end_time),
                                tensor.convert_to_tensor(start_time))
        avg_time = ops.divide(time_diff, tensor.convert_to_tensor(10.0))  # Adjusted divisor
        
        print(f"  Matrix size {size}x{size}: {avg_time.item():.4f} seconds")


def check_available_backends() -> Dict[str, bool]:
    """Check which backends and devices are available.
    
    This function uses the auto_select_backend function to check which backends
    and devices are available without directly importing backend implementations.
    
    Returns:
        Dictionary with availability information for different backends and devices
    """
    availability = {
        'numpy': False,
        'torch_cpu': False,
        'torch_cuda': False,
        'torch_mps': False,
        'mlx': False
    }
    
    # First check PyTorch to ensure we detect MPS before MLX takes over
    try:
        # Try PyTorch on CPU
        set_backend('torch')
        availability['torch_cpu'] = True
        
        # Check for CUDA and MPS using ops.get_available_devices
        devices = ops.get_available_devices()
        availability['torch_cuda'] = 'cuda' in devices or any(d.startswith('cuda:') for d in devices)
        availability['torch_mps'] = 'mps' in devices
        
        # For Apple Silicon, assume MPS might be available if we're on macOS with Apple Silicon
        if not availability['torch_mps'] and platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # We'll set this to True and let the benchmark function handle any errors
            # This avoids direct backend imports in the frontend
            availability['torch_mps'] = True
    except (ImportError, ValueError):
        pass
    
    # Try NumPy
    try:
        set_backend('numpy')
        availability['numpy'] = True
    except (ImportError, ValueError):
        pass
    
    # Try MLX last
    try:
        set_backend('mlx')
        availability['mlx'] = True
    except (ImportError, ValueError):
        pass
    
    return availability


def main() -> None:
    """Main function to demonstrate backend performance comparison.
    
    This function benchmarks matrix multiplication operations using different
    backends (NumPy, PyTorch, MLX) and devices (CPU, MPS, CUDA) if available.
    """
    print("Ember ML Backend Performance Comparison")
    print("=======================================")
    
    # Check backend availability
    availability = check_available_backends()
    
    # Auto-select the best backend
    best_backend, best_device = auto_select_backend()
    
    # Create message using f-strings instead of + operator
    device_info = f" on {best_device}" if best_device else ""
    best_backend_msg = f"Best available backend: {best_backend}{device_info}"
    print(best_backend_msg)
    
    print(f"NumPy available: {availability['numpy']}")
    print(f"PyTorch CPU available: {availability['torch_cpu']}")
    print(f"PyTorch CUDA available: {availability['torch_cuda']}")
    print(f"PyTorch MPS available: {availability['torch_mps']}")
    print(f"MLX available: {availability['mlx']}")
    
    # Use smaller matrix sizes for faster demonstration
    sizes = [500, 1000, 2000]
    
    # Benchmark NumPy backend (CPU only)
    if availability['numpy']:
        benchmark_matrix_multiply('numpy', sizes=sizes)
    
    # Benchmark PyTorch backend on CPU
    if availability['torch_cpu']:
        benchmark_matrix_multiply('torch', device='cpu', sizes=sizes)
    
    # Benchmark PyTorch backend on MPS if available
    if availability['torch_mps']:
        benchmark_matrix_multiply('torch', device='mps', sizes=sizes)
    
    # Benchmark PyTorch backend on CUDA if available
    if availability['torch_cuda']:
        benchmark_matrix_multiply('torch', device='cuda', sizes=sizes)
    
    # Benchmark MLX backend if available (optimized for Apple Silicon)
    if availability['mlx']:
        benchmark_matrix_multiply('mlx', sizes=sizes)
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()