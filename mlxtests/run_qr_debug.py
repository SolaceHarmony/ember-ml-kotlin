#!/usr/bin/env python3
"""
Run QR decomposition debug tests to identify and fix the zero matrices issue.
"""
import os
import sys
import time
import mlx.core as mx

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from mlxtests.qr_simulation import simulate_metal_qr, compare_implementations
from mlxtests.hpc_16x8_method.debug_qr_decomp import debug_qr
from mlxtests.hpc_16x8_method.enhanced_qr_decomp import enhanced_tiled_qr

def run_comprehensive_tests():
    """Run comprehensive tests to identify the issue with zero matrices."""
    print("=" * 80)
    print("COMPREHENSIVE QR DECOMPOSITION DEBUG TESTS")
    print("=" * 80)
    
    # Set environment variables
    os.environ["MLX_USE_METAL"] = "1"
    
    # Test matrices
    test_matrices = [
        # Small well-conditioned matrix
        {
            "name": "Small well-conditioned matrix (4x4)",
            "matrix": mx.array([
                [4.0, 1.0, -2.0, 2.0],
                [1.0, 2.0, 0.0, 1.0],
                [-2.0, 0.0, 3.0, -2.0],
                [2.0, 1.0, -2.0, -1.0]
            ], dtype=mx.float32)
        },
        # Example matrix that failed
        {
            "name": "Example matrix that failed (5x5)",
            "matrix": mx.array([
                [-0.21517, 0.357598, -1.42385, -0.337991, 1.10607],
                [1.39705, -0.0175396, 0.347177, 1.87311, 0.797497],
                [-0.661596, -1.16188, -0.33521, 0.0483204, -0.01543],
                [0.639394, 1.11222, 0.415146, 0.142572, 1.26951],
                [1.17061, 0.106101, 0.514818, 2.10361, -0.635574]
            ], dtype=mx.float32)
        },
        # Random matrix
        {
            "name": "Random matrix (10x10)",
            "matrix": mx.random.normal((10, 10))
        },
        # Tall matrix
        {
            "name": "Tall matrix (10x5)",
            "matrix": mx.random.normal((10, 5))
        },
        # Wide matrix
        {
            "name": "Wide matrix (5x10)",
            "matrix": mx.random.normal((5, 10))
        }
    ]
    
    # Thread/grid configurations to test
    configurations = [
        ((32, 1, 1), (32, 1, 1), "Tiny (32x32)"),
        ((64, 1, 1), (64, 1, 1), "Very Small (64x64)"),
        ((128, 1, 1), (128, 1, 1), "Small (128x128)"),
        ((256, 1, 1), (256, 1, 1), "Medium (256x256)"),
        ((512, 1, 1), (512, 1, 1), "Large (512x512)")
    ]
    
    # Run tests for each matrix
    for test_case in test_matrices:
        print("\n" + "=" * 80)
        print(f"TESTING: {test_case['name']}")
        print("=" * 80)
        
        A = test_case["matrix"]
        
        # 1. Run Python simulation
        print("\n1. Running Python simulation...")
        t0 = time.time()
        Q_py, R_py, dbg_py = simulate_metal_qr(A, debug=True)
        dt_py = time.time() - t0
        print(f"Python simulation time: {dt_py:.6f} seconds")
        
        # Check if matrices contain all zeros
        q_py_zeros = mx.all(Q_py == 0.0).item()
        r_py_zeros = mx.all(R_py == 0.0).item()
        print(f"Python Q contains all zeros: {q_py_zeros}")
        print(f"Python R contains all zeros: {r_py_zeros}")
        
        # 2. Run original implementation
        print("\n2. Running original implementation...")
        t0 = time.time()
        Q_orig, R_orig, dbg_orig = enhanced_tiled_qr(A, debug=True)
        dt_orig = time.time() - t0
        print(f"Original implementation time: {dt_orig:.6f} seconds")
        
        # Check if matrices contain all zeros
        q_orig_zeros = mx.all(Q_orig == 0.0).item()
        r_orig_zeros = mx.all(R_orig == 0.0).item()
        print(f"Original Q contains all zeros: {q_orig_zeros}")
        print(f"Original R contains all zeros: {r_orig_zeros}")
        
        # 3. Run debug implementation with different configurations
        print("\n3. Running debug implementation with different configurations...")
        best_config = None
        best_error = float('inf')
        
        for grid_size, thread_size, config_name in configurations:
            print(f"\nTesting {config_name} configuration...")
            
            t0 = time.time()
            Q_debug, R_debug, dbg_debug = debug_qr(
                A, debug=True, grid_size=grid_size, thread_size=thread_size
            )
            dt_debug = time.time() - t0
            print(f"Debug implementation time: {dt_debug:.6f} seconds")
            
            # Check if matrices contain all zeros
            q_debug_zeros = mx.all(Q_debug == 0.0).item()
            r_debug_zeros = mx.all(R_debug == 0.0).item()
            print(f"Debug Q contains all zeros: {q_debug_zeros}")
            print(f"Debug R contains all zeros: {r_debug_zeros}")
            
            # Only analyze non-zero results
            if not q_debug_zeros:
                # Check orthogonality and reconstruction
                QTQ = mx.matmul(Q_debug.T, Q_debug)
                ortho_error = mx.mean(mx.abs(QTQ - mx.eye(Q_debug.shape[0]))).item()
                QR = mx.matmul(Q_debug, R_debug)
                recon_error = mx.mean(mx.abs(QR - A)).item()
                print(f"Orthogonality error: {ortho_error:.8f}")
                print(f"Reconstruction error: {recon_error:.8f}")
                
                # Track best configuration
                if recon_error < best_error:
                    best_error = recon_error
                    best_config = (grid_size, thread_size, config_name)
        
        if best_config:
            print(f"\nBest configuration for {test_case['name']}: {best_config[2]} with error {best_error:.8f}")
        else:
            print(f"\nNo successful configuration found for {test_case['name']}")
        
        # 4. Compare debug values between implementations
        print("\n4. Comparing debug values between implementations...")
        print("Debug values comparison:")
        print(f"{'Index':<5} {'Python':<15} {'Original':<15} {'Debug (Best)':<15} {'Py-Orig Diff':<15} {'Py-Debug Diff':<15}")
        
        for i in range(16):
            py_val = dbg_py[i].item()
            orig_val = dbg_orig[i].item()
            debug_val = dbg_debug[i].item() if 'dbg_debug' in locals() else float('nan')
            
            py_orig_diff = abs(py_val - orig_val)
            py_debug_diff = abs(py_val - debug_val) if 'dbg_debug' in locals() else float('nan')
            
            print(f"{i:<5} {py_val:<15.6e} {orig_val:<15.6e} {debug_val:<15.6e} {py_orig_diff:<15.6e} {py_debug_diff:<15.6e}")
    
    # Final analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("=" * 80)
    
    # Compare the implementations and identify patterns
    print("\nBased on the tests, here are the key findings:")
    
    # Check if Python simulation consistently works
    py_success = all(not mx.all(simulate_metal_qr(test_case["matrix"])[0] == 0.0).item() for test_case in test_matrices)
    print(f"1. Python simulation consistently works: {py_success}")
    
    # Check if original implementation consistently fails
    orig_failure = all(mx.all(enhanced_tiled_qr(test_case["matrix"])[0] == 0.0).item() for test_case in test_matrices)
    print(f"2. Original implementation consistently fails: {orig_failure}")
    
    # Check if debug implementation with specific configuration works
    debug_success = False
    best_overall_config = None
    for grid_size, thread_size, config_name in configurations:
        success_count = 0
        for test_case in test_matrices:
            Q, _, _ = debug_qr(test_case["matrix"], debug=True, grid_size=grid_size, thread_size=thread_size)
            if not mx.all(Q == 0.0).item():
                success_count += 1
        
        if success_count == len(test_matrices):
            debug_success = True
            best_overall_config = config_name
            break
    
    print(f"3. Debug implementation with specific configuration works: {debug_success}")
    if debug_success:
        print(f"   Best overall configuration: {best_overall_config}")
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    if py_success and orig_failure:
        print("1. The issue appears to be in the Metal kernel implementation, not the algorithm itself.")
        print("2. The most likely causes are:")
        print("   a. Thread synchronization issues")
        print("   b. Memory access patterns")
        print("   c. Numerical precision differences between Python and Metal")
    
    if debug_success:
        print(f"3. Use the {best_overall_config} configuration for the Metal kernel.")
    
    print("4. Consider the following fixes:")
    print("   a. Ensure proper thread synchronization with threadgroup_barrier")
    print("   b. Check for race conditions in the Metal kernel")
    print("   c. Verify memory access patterns, especially for Q and R matrices")
    print("   d. Adjust the EPSILON value for numerical stability")
    print("   e. Implement the algorithm in a more thread-safe manner")

if __name__ == "__main__":
    run_comprehensive_tests()