#!/usr/bin/env python3
"""
Test script for QR decomposition thread allocation optimized for M3 Ultra GPU.
This script incorporates detailed knowledge of the M3 Ultra GPU architecture.
"""

# Device-specific information for Apple M3 Ultra
DEVICE_INFO = {
    'resource_limit': 499000,
    'max_buffer_length': 167503724544,
    'architecture': 'applegpu_g15d',
    'memory_size': 274877906944,
    'max_recommended_working_set_size': 223338299392,
    'device_name': 'Apple M3 Ultra',
    # M3 Ultra specific GPU information
    'gpu_cores': 80,  # 80-core variant
    'execution_units_per_core': 16,
    'alus_per_core': 128,
    'total_execution_units': 80 * 16,  # 1,280
    'total_alus': 80 * 128  # 10,240
}

def calculate_m3_ultra_thread_allocation(m: int, n: int, debug: bool = True) -> dict:
    """
    Calculate optimal thread allocation for QR decomposition based on M3 Ultra architecture.
    
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
    
    # Get M3 Ultra GPU capabilities
    thread_execution_width = 32  # Typical for Apple GPUs
    max_threads_per_threadgroup = 1024  # Maximum threads per threadgroup
    
    # M3 Ultra specific values
    total_execution_units = DEVICE_INFO['total_execution_units']  # 1,280
    total_alus = DEVICE_INFO['total_alus']  # 10,240
    
    # Calculate optimal thread allocation based on matrix size
    # For very small matrices (up to 8x8), use minimal threads
    if max(m, n) <= 8:
        threadgroup_size = thread_execution_width
        total_threads = threadgroup_size
    # For small matrices (up to 32x32), use one thread per element
    elif max(m, n) <= 32:
        threadgroup_size = thread_execution_width
        total_threads = min(total_elements, 256)
    # For medium matrices (up to 128x128), scale more aggressively
    elif max(m, n) <= 128:
        threadgroup_size = thread_execution_width
        # Use up to 2048 threads (still conservative compared to 10,240 ALUs)
        total_threads = min(2048, total_elements)
    # For large matrices, use even more threads
    else:
        threadgroup_size = thread_execution_width
        # Use up to 4096 threads (still conservative compared to 10,240 ALUs)
        total_threads = min(4096, total_elements)
    
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
        'device': DEVICE_INFO['device_name'],
        'gpu_cores': DEVICE_INFO['gpu_cores'],
        'total_execution_units': total_execution_units,
        'total_alus': total_alus
    }
    
    if debug:
        print(f"\nMatrix dimensions: {m}x{n}")
        print(f"  Device: {DEVICE_INFO['device_name']} ({DEVICE_INFO['gpu_cores']}-core GPU)")
        print(f"  Total elements: {total_elements}")
        print(f"  Total work: {total_work}")
        print(f"  Threadgroup size: {threadgroup_size}")
        print(f"  Total threads: {total_threads}")
        print(f"  Work per thread: {work_per_thread:.2f}")
        print(f"  Max work per thread: {MAX_WORK_PER_THREAD}")
        print(f"  Is safe: {is_safe}")
        print(f"  Thread utilization: {total_threads / total_alus * 100:.2f}% of ALUs")
    
    return result

def simulate_kernel_workload_check(m, n, grid_sz):
    """
    Simulate the exact workload check from the Metal kernel.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        grid_sz: Total number of threads in the grid
        
    Returns:
        Dictionary with the results of the workload check
    """
    print(f"\nSimulating kernel workload check for {m}x{n} matrix with {grid_sz} threads:")
    
    # STRICT WORKLOAD CHECK: Exactly as in the kernel
    if m <= 1000 and n <= 1000:  # Prevent overflow
        total_elements = m * m    # Q matrix elements
        total_elements += m * n   # R matrix elements
        total_elements += m * n   # Additional work for reflections
        
        print(f"  Total elements: {total_elements}")
        
        # Check if we have enough threads
        if grid_sz == 0:
            print("  ERROR: Division by zero (grid_sz is 0)")
            return {
                'success': False,
                'error_code': 2.0,
                'error_value': 0.0
            }
        
        # Calculate work per thread
        work_per_thread = total_elements / grid_sz
        
        print(f"  Work per thread: {work_per_thread:.2f}")
        
        # Check if work per thread is reasonable
        MAX_WORK_PER_THREAD = 10000
        if work_per_thread > MAX_WORK_PER_THREAD:
            print(f"  ERROR: Work per thread ({work_per_thread:.2f}) exceeds maximum ({MAX_WORK_PER_THREAD})")
            return {
                'success': False,
                'error_code': 2.0,
                'error_value': work_per_thread
            }
        
        # All checks passed
        print("  SUCCESS: Workload check passed")
        return {
            'success': True,
            'total_elements': total_elements,
            'work_per_thread': work_per_thread
        }
    else:
        # Matrix dimensions too large
        print(f"  ERROR: Matrix dimensions ({m}x{n}) exceed maximum (1000x1000)")
        return {
            'success': False,
            'error_code': 2.0,
            'error_value': m * n
        }

def test_m3_ultra_allocation():
    """Test the M3 Ultra optimized thread allocation."""
    print("Testing M3 Ultra optimized thread allocation:")
    print("=" * 80)
    
    # Test various matrix sizes
    test_sizes = [
        (2, 2),      # Tiny
        (8, 8),      # Tiny
        (32, 32),    # Small
        (64, 64),    # Medium
        (128, 128),  # Medium
        (256, 256),  # Large
        (512, 512),  # Large
        (1000, 1000),# Maximum allowed
        (10, 5),     # Rectangular
        (100, 50),   # Rectangular
        (500, 100)   # Rectangular
    ]
    
    for m, n in test_sizes:
        # Get M3 Ultra optimized thread allocation
        allocation = calculate_m3_ultra_thread_allocation(m, n)
        grid_sz = allocation['total_threads']
        
        # Simulate kernel workload check
        result = simulate_kernel_workload_check(m, n, grid_sz)
        
        # Verify that our Python calculation matches the kernel calculation
        if result['success']:
            python_work = allocation['work_per_thread']
            kernel_work = result['work_per_thread']
            match = abs(python_work - kernel_work) < 0.01
            print(f"  Python vs Kernel work calculation: {'MATCH' if match else 'MISMATCH'}")
            if not match:
                print(f"    Python: {python_work:.2f}")
                print(f"    Kernel: {kernel_work:.2f}")
        
        print("-" * 80)
    
    print("M3 Ultra thread allocation testing complete.")

if __name__ == "__main__":
    test_m3_ultra_allocation()