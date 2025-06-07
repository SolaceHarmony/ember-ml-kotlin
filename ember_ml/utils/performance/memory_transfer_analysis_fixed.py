"""
Memory Transfer Analysis for PyTorch Backends

This script analyzes whether tensors stay in GPU/MPS memory between operations
or if there's unnecessary data transfer happening.
"""

import time
import torch

def print_tensor_info(tensor, name):
    """Print detailed information about a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    
    # Try to get memory info if possible
    if tensor.device.type == 'cuda':
        print(f"  Memory allocated: {torch.cuda.memory_allocated(tensor.device) / 1024**2:.2f} MB")
    elif tensor.device.type == 'mps':
        # MPS doesn't have a direct memory query API like CUDA
        print("  Memory allocated: [Not available for MPS]")

def time_operation(operation_name, operation_fn, *args, **kwargs):
    """Time an operation and return the result and time taken."""
    start_time = time.time()
    result = operation_fn(*args, **kwargs)
    end_time = time.time()
    print(f"{operation_name}: {(end_time - start_time) * 1000:.2f} ms")
    return result

def analyze_memory_transfers(device='mps', size=5000, dtype=torch.float32):
    """
    Analyze memory transfers between operations.
    
    Args:
        device: Device to use ('cpu', 'cuda', or 'mps')
        size: Size of matrices to use
        dtype: Data type to use
    """
    print(f"\n=== Memory Transfer Analysis (Device: {device}, Size: {size}, Dtype: {dtype}) ===")
    
    # Create tensors on the specified device
    print("\nCreating tensors...")
    a = time_operation("Create tensor A", 
                      lambda: torch.randn(size, size, dtype=dtype, device=device))
    b = time_operation("Create tensor B", 
                      lambda: torch.randn(size, size, dtype=dtype, device=device))
    
    print_tensor_info(a, "Tensor A")
    print_tensor_info(b, "Tensor B")
    
    # Perform operations and measure time
    print("\nPerforming operations...")
    
    # First operation: matrix multiplication
    c = time_operation("A @ B (first time)", 
                      lambda: ops.matmul(a, b))
    print_tensor_info(c, "Result C")
    
    # Second operation: same matrix multiplication
    # If tensors stay in GPU memory, this should be faster
    d = time_operation("A @ B (second time)", 
                      lambda: ops.matmul(a, b))
    print_tensor_info(d, "Result D")
    
    # Third operation: matrix multiplication after moving tensors to CPU and back
    print("\nMoving tensors to CPU and back...")
    a_cpu = time_operation("A to CPU", lambda: a.cpu())
    b_cpu = time_operation("B to CPU", lambda: b.cpu())
    
    a_back = time_operation("A back to device", lambda: a_cpu.to(device))
    b_back = time_operation("B back to device", lambda: b_cpu.to(device))
    
    # Matrix multiplication after moving
    e = time_operation("A_back @ B_back", 
                      lambda: ops.matmul(a_back, b_back))
    print_tensor_info(e, "Result E")
    
    # Chain of operations
    print("\nChain of operations...")
    
    # Single chain without intermediate variables
    f = time_operation("A @ B + (A @ B).T", 
                      lambda: ops.matmul(a, b) + ops.matmul(a, b).T)
    print_tensor_info(f, "Result F")
    
    # Chain with intermediate variables
    temp = time_operation("temp = A @ B", 
                         lambda: ops.matmul(a, b))
    g = time_operation("temp + temp.T", 
                      lambda: temp + temp.T)
    print_tensor_info(g, "Result G")
    
    # Check if results are the same
    print("\nVerifying results...")
    print(f"C and D are equal: {torch.allclose(c, d)}")
    print(f"C and E are equal: {torch.allclose(c, e)}")
    print(f"F and G are equal: {torch.allclose(f, g)}")

def main():
    """Main function to run the analysis."""
    print("PyTorch Memory Transfer Analysis")
    print("===============================")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    # Determine available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    # Test with different data types
    dtypes = [torch.float32, torch.float16]
    
    # Run analysis for each device and dtype
    for device in devices:
        for dtype in dtypes:
            try:
                analyze_memory_transfers(device, size=2000, dtype=dtype)
            except Exception as e:
                print(f"Error with device={device}, dtype={dtype}: {e}")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()