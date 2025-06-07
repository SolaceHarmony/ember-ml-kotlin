#!/usr/bin/env python3
"""
Test script to identify potential shared memory issues in the QR kernel.
"""

def analyze_shared_memory_usage(m, n, grid_sz, threadgroup_sz):
    """
    Analyze shared memory usage in the QR kernel.
    
    Args:
        m: Number of rows in the matrix
        n: Number of columns in the matrix
        grid_sz: Total number of threads in the grid
        threadgroup_sz: Number of threads per threadgroup
        
    Returns:
        Dictionary with analysis results
    """
    # Constants from the kernel
    WARP_SIZE = 32
    NUM_LIMBS = 8
    
    # Calculate shared memory usage
    thread_limbs_size = WARP_SIZE * NUM_LIMBS * 4  # 4 bytes per uint
    simd_max_size = 8 * 4  # 8 floats, 4 bytes per float
    simd_sigma_size = 8 * 4  # 8 floats, 4 bytes per float
    
    total_shared_memory = thread_limbs_size + simd_max_size + simd_sigma_size
    
    # Calculate number of SIMD groups
    num_simd_groups = (grid_sz + WARP_SIZE - 1) // WARP_SIZE
    
    # Check for potential issues
    issues = []
    
    # Issue 1: thread_limbs array size issue
    if threadgroup_sz > WARP_SIZE:
        issues.append(f"threadgroup_size ({threadgroup_sz}) > WARP_SIZE ({WARP_SIZE})")
    
    # Issue 2: Thread ID indexing issue
    max_thread_id = grid_sz - 1
    if max_thread_id * NUM_LIMBS >= WARP_SIZE * NUM_LIMBS:
        issues.append(f"max_thread_id * NUM_LIMBS ({max_thread_id * NUM_LIMBS}) >= WARP_SIZE * NUM_LIMBS ({WARP_SIZE * NUM_LIMBS})")
    
    # Issue 3: Combining limbs loop issue
    if grid_sz > WARP_SIZE:
        issues.append(f"grid_sz ({grid_sz}) > WARP_SIZE ({WARP_SIZE})")
    
    # Issue 4: SIMD groups issue
    if num_simd_groups > 8:
        issues.append(f"num_simd_groups ({num_simd_groups}) > 8")
    
    return {
        'matrix_size': f"{m}x{n}",
        'grid_size': grid_sz,
        'threadgroup_size': threadgroup_sz,
        'thread_limbs_size': thread_limbs_size,
        'simd_max_size': simd_max_size,
        'simd_sigma_size': simd_sigma_size,
        'total_shared_memory': total_shared_memory,
        'num_simd_groups': num_simd_groups,
        'issues': issues
    }

def test_shared_memory_issues():
    """Test for shared memory issues with various matrix sizes."""
    from test_m3_ultra_allocation import calculate_m3_ultra_thread_allocation
    
    print("Testing for shared memory issues:")
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
        allocation = calculate_m3_ultra_thread_allocation(m, n, debug=False)
        grid_sz = allocation['total_threads']
        threadgroup_sz = allocation['threadgroup_size']
        
        # Analyze shared memory usage
        analysis = analyze_shared_memory_usage(m, n, grid_sz, threadgroup_sz)
        
        print(f"\nMatrix dimensions: {m}x{n}")
        print(f"  Grid size: {grid_sz}")
        print(f"  Threadgroup size: {threadgroup_sz}")
        print(f"  Shared memory usage: {analysis['total_shared_memory']} bytes")
        print(f"  Number of SIMD groups: {analysis['num_simd_groups']}")
        
        if analysis['issues']:
            print("  ISSUES DETECTED:")
            for issue in analysis['issues']:
                print(f"    - {issue}")
        else:
            print("  No issues detected")
        
        print("-" * 80)
    
    print("Shared memory analysis complete.")

if __name__ == "__main__":
    test_shared_memory_issues()