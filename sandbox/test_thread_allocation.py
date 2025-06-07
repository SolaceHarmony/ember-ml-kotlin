#!/usr/bin/env python3
"""
Test script for QR decomposition thread allocation calculations.
This script tests the thread allocation algorithm without running any kernels.
"""

# Device-specific information for Apple M3 Ultra
DEVICE_INFO = {
    'resource_limit': 499000,
    'max_buffer_length': 167503724544,
    'architecture': 'applegpu_g15d',
    'memory_size': 274877906944,
    'max_recommended_working_set_size': 223338299392,
    'device_name': 'Apple M3 Ultra'
}

def calculate_optimal_thread_allocation(m: int, n: int, debug: bool = True) -> dict:
    """
    Calculate optimal thread allocation for QR decomposition based on matrix dimensions
    and device-specific information.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing grid and threadgroup configurations
    """
    # Calculate total elements and work
    q_elements = m * m
    r_elements = m * n
    total_elements = q_elements + r_elements
    total_work = total_elements + m * n  # Include reflection operations
    
    # Get Metal device capabilities from device info
    thread_execution_width = 32  # Typical for Apple GPUs
    max_threads_per_threadgroup = 1024  # Maximum threads per threadgroup
    
    # For Apple M3 Ultra, we can use device-specific optimizations
    if DEVICE_INFO['device_name'] == 'Apple M3 Ultra':
        # For very small matrices (up to 8x8), use minimal threads
        if max(m, n) <= 8:
            threadgroup_size = thread_execution_width
            total_threads = threadgroup_size
        # For small matrices (up to 32x32), use one thread per element
        elif max(m, n) <= 32:
            threadgroup_size = thread_execution_width
            total_threads = min(total_elements, 256)
        # For medium matrices, scale more conservatively
        else:
            threadgroup_size = thread_execution_width
            # Use at most 1024 threads for any matrix
            total_threads = min(1024, total_elements)
    else:
        # Generic approach for unknown devices
        # Calculate optimal threadgroup size (multiple of thread_execution_width)
        threadgroup_size = thread_execution_width
        while threadgroup_size * 2 <= max_threads_per_threadgroup and threadgroup_size * 2 <= total_elements:
            threadgroup_size *= 2
        
        # Calculate optimal number of threads (one per element, but capped)
        MAX_TOTAL_THREADS = 1024  # Hard cap on total threads - much more conservative
        total_threads = min(total_elements, MAX_TOTAL_THREADS)
    
    # Round up to multiple of threadgroup_size
    total_threads = ((total_threads + threadgroup_size - 1) // threadgroup_size) * threadgroup_size
    
    # Calculate grid and threadgroup
    grid = (total_threads, 1, 1)
    threadgroup = (threadgroup_size, 1, 1)
    
    # Calculate work per thread
    work_per_thread = total_work / total_threads if total_threads > 0 else float('inf')
    
    # Check if work per thread is reasonable - use a more conservative limit for small matrices
    MAX_WORK_PER_THREAD = 1000 if max(m, n) <= 32 else 10000
    is_safe = work_per_thread <= MAX_WORK_PER_THREAD
    
    result = {
        'grid': grid,
        'threadgroup': threadgroup,
        'total_threads': total_threads,
        'threadgroup_size': threadgroup_size,
        'work_per_thread': work_per_thread,
        'is_safe': is_safe,
        'total_elements': total_elements,
        'total_work': total_work,
        'device': DEVICE_INFO['device_name']
    }
    
    if debug:
        print(f"\nMatrix dimensions: {m}x{n}")
        print(f"  Device: {DEVICE_INFO['device_name']}")
        print(f"  Total elements: {total_elements}")
        print(f"  Total work: {total_work}")
        print(f"  Threadgroup size: {threadgroup_size}")
        print(f"  Total threads: {total_threads}")
        print(f"  Work per thread: {work_per_thread:.2f}")
        print(f"  Max work per thread: {MAX_WORK_PER_THREAD}")
        print(f"  Is safe: {is_safe}")
    
    return result

def test_various_matrix_sizes():
    """Test thread allocation for various matrix sizes."""
    print("Testing thread allocation for various matrix sizes:")
    print("=" * 80)
    
    # Test tiny matrices
    for size in [(2, 2), (4, 4), (8, 8)]:
        calculate_optimal_thread_allocation(size[0], size[1])
    
    # Test small matrices
    for size in [(16, 16), (32, 32)]:
        calculate_optimal_thread_allocation(size[0], size[1])
    
    # Test medium matrices
    for size in [(64, 64), (128, 128)]:
        calculate_optimal_thread_allocation(size[0], size[1])
    
    # Test large matrices
    for size in [(256, 256), (512, 512), (1000, 1000)]:
        calculate_optimal_thread_allocation(size[0], size[1])
    
    # Test rectangular matrices
    for size in [(10, 5), (100, 50), (500, 100)]:
        calculate_optimal_thread_allocation(size[0], size[1])
    
    print("=" * 80)
    print("Thread allocation testing complete.")

if __name__ == "__main__":
    test_various_matrix_sizes()