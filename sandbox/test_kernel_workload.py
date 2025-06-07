#!/usr/bin/env python3
"""
Test script to simulate the exact workload calculation from the Metal kernel.
This verifies that our Python calculations match what the kernel will compute.
"""

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

def test_with_optimal_allocation():
    """Test the kernel workload check with our optimal thread allocation."""
    from test_thread_allocation import calculate_optimal_thread_allocation
    
    print("Testing kernel workload check with optimal thread allocation:")
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
        (1001, 1000),# Exceeds maximum
        (10, 5),     # Rectangular
        (100, 50),   # Rectangular
        (500, 100)   # Rectangular
    ]
    
    for m, n in test_sizes:
        # Get optimal thread allocation
        allocation = calculate_optimal_thread_allocation(m, n, debug=False)
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
    
    print("Kernel workload check testing complete.")

if __name__ == "__main__":
    test_with_optimal_allocation()