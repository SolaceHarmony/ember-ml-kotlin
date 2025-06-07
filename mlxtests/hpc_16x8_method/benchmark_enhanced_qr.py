"""
Benchmark for Enhanced Tiled HPC QR Decomposition.

This script compares the enhanced QR decomposition against the native MLX QR
implementation across various matrix sizes and conditions, measuring both
performance and numerical stability metrics.
"""
import time
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from typing import List, Tuple, Dict
from enhanced_qr_decomp import enhanced_tiled_qr

def generate_test_matrix(m: int, n: int, condition: float = 1.0) -> mx.array:
    """
    Generate a test matrix with specified condition number.
    
    Args:
        m: Number of rows
        n: Number of columns
        condition: Approximate condition number
        
    Returns:
        A matrix with the specified dimensions and condition number
    """
    # Create random orthogonal matrices
    u, _ = mx.linalg.qr(mx.random.normal((m, m)))
    v, _ = mx.linalg.qr(mx.random.normal((n, n)))
    
    # Create singular values with specified condition number
    min_dim = min(m, n)
    singular_vals = mx.array([condition ** (-(i/(min_dim-1))) for i in range(min_dim)])
    s = mx.zeros((m, n))
    for i in range(min_dim):
        s = s.at[i, i].set(singular_vals[i])
    
    # Combine to create matrix with specified condition number
    return mx.matmul(mx.matmul(u, s), v)

def run_benchmark(sizes: List[int], 
                 conditions: List[float] = [1.0], 
                 num_runs: int = 3) -> Dict:
    """
    Run benchmark comparing enhanced QR and native MLX QR.
    
    Args:
        sizes: List of matrix sizes to test (square matrices)
        conditions: List of condition numbers to test
        num_runs: Number of runs for each test
        
    Returns:
        Dictionary of benchmark results
    """
    results = {
        'sizes': sizes,
        'conditions': conditions,
        'enhanced_time': {},
        'native_time': {},
        'ortho_error_enhanced': {},
        'recon_error_enhanced': {},
        'ortho_error_native': {},
        'recon_error_native': {},
        'q_diff': {},
        'r_diff': {}
    }
    
    for condition in conditions:
        results['enhanced_time'][condition] = []
        results['native_time'][condition] = []
        results['ortho_error_enhanced'][condition] = []
        results['recon_error_enhanced'][condition] = []
        results['ortho_error_native'][condition] = []
        results['recon_error_native'][condition] = []
        results['q_diff'][condition] = []
        results['r_diff'][condition] = []
        
        for size in sizes:
            print(f"\nTesting matrix size {size}x{size} with condition number {condition}")
            
            # Generate test matrix
            A = generate_test_matrix(size, size, condition)
            
            # Run enhanced QR
            enhanced_times = []
            q_list, r_list = [], []
            
            for _ in range(num_runs):
                start_time = time.time()
                Q, R, _ = enhanced_tiled_qr(A, debug=False)
                end_time = time.time()
                
                enhanced_times.append(end_time - start_time)
                q_list.append(Q)
                r_list.append(R)
            
            # Use the last run for error calculations
            Q, R = q_list[-1], r_list[-1]
            
            # Check orthogonality of Q
            Q_T_Q = mx.matmul(Q.T, Q)
            I = mx.eye(size)
            ortho_error = float(mx.abs(Q_T_Q - I).max())
            
            # Check reconstruction error
            QR = mx.matmul(Q, R)
            recon_error = float(mx.abs(QR - A).max())
            
            # Run native QR
            native_times = []
            native_q_list, native_r_list = [], []
            
            try:
                for _ in range(num_runs):
                    start_time = time.time()
                    Q_native, R_native = mx.linalg.qr(A)
                    end_time = time.time()
                    
                    native_times.append(end_time - start_time)
                    native_q_list.append(Q_native)
                    native_r_list.append(R_native)
                
                # Use the last run for error calculations
                Q_native, R_native = native_q_list[-1], native_r_list[-1]
                
                # Check orthogonality of Q_native
                Q_native_T_Q_native = mx.matmul(Q_native.T, Q_native)
                ortho_error_native = float(mx.abs(Q_native_T_Q_native - I).max())
                
                # Check reconstruction error
                QR_native = mx.matmul(Q_native, R_native)
                recon_error_native = float(mx.abs(QR_native - A).max())
                
                # Compare solutions
                q_diff = float(mx.abs(Q - Q_native).max())
                r_diff = float(mx.abs(R - R_native).max())
                
            except Exception as e:
                print(f"Native QR failed: {str(e)}")
                ortho_error_native = np.nan
                recon_error_native = np.nan
                q_diff = np.nan
                r_diff = np.nan
                native_times = [np.nan]
            
            # Store results
            results['enhanced_time'][condition].append(np.mean(enhanced_times))
            results['native_time'][condition].append(np.mean(native_times))
            results['ortho_error_enhanced'][condition].append(ortho_error)
            results['recon_error_enhanced'][condition].append(recon_error)
            results['ortho_error_native'][condition].append(ortho_error_native)
            results['recon_error_native'][condition].append(recon_error_native)
            results['q_diff'][condition].append(q_diff)
            results['r_diff'][condition].append(r_diff)
            
            # Print summary
            print(f"Enhanced QR: {np.mean(enhanced_times):.4f}s, ortho error: {ortho_error:.2e}, recon error: {recon_error:.2e}")
            print(f"Native QR: {np.mean(native_times):.4f}s, ortho error: {ortho_error_native:.2e}, recon error: {recon_error_native:.2e}")
            print(f"Q diff: {q_diff:.2e}, R diff: {r_diff:.2e}")
    
    return results

def plot_results(results: Dict, save_path: str = None):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        save_path: Path to save the plot (if None, show plot)
    """
    sizes = results['sizes']
    conditions = results['conditions']
    
    fig, axes = plt.subplots(len(conditions), 4, figsize=(20, 5 * len(conditions)))
    
    # For a single condition, reshape axes
    if len(conditions) == 1:
        axes = axes.reshape(1, -1)
    
    for i, condition in enumerate(conditions):
        # Performance comparison
        axes[i, 0].plot(sizes, results['enhanced_time'][condition], '-o', label='Enhanced QR')
        axes[i, 0].plot(sizes, results['native_time'][condition], '-o', label='Native QR')
        axes[i, 0].set_title(f'Performance (condition={condition})')
        axes[i, 0].set_xlabel('Matrix Size')
        axes[i, 0].set_ylabel('Time (seconds)')
        axes[i, 0].set_xscale('log')
        axes[i, 0].set_yscale('log')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Orthogonality error
        axes[i, 1].plot(sizes, results['ortho_error_enhanced'][condition], '-o', label='Enhanced QR')
        axes[i, 1].plot(sizes, results['ortho_error_native'][condition], '-o', label='Native QR')
        axes[i, 1].set_title(f'Orthogonality Error (condition={condition})')
        axes[i, 1].set_xlabel('Matrix Size')
        axes[i, 1].set_ylabel('Error (max norm)')
        axes[i, 1].set_xscale('log')
        axes[i, 1].set_yscale('log')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
        
        # Reconstruction error
        axes[i, 2].plot(sizes, results['recon_error_enhanced'][condition], '-o', label='Enhanced QR')
        axes[i, 2].plot(sizes, results['recon_error_native'][condition], '-o', label='Native QR')
        axes[i, 2].set_title(f'Reconstruction Error (condition={condition})')
        axes[i, 2].set_xlabel('Matrix Size')
        axes[i, 2].set_ylabel('Error (max norm)')
        axes[i, 2].set_xscale('log')
        axes[i, 2].set_yscale('log')
        axes[i, 2].legend()
        axes[i, 2].grid(True)
        
        # Solution difference
        axes[i, 3].plot(sizes, results['q_diff'][condition], '-o', label='Q Difference')
        axes[i, 3].plot(sizes, results['r_diff'][condition], '-o', label='R Difference')
        axes[i, 3].set_title(f'Solution Difference (condition={condition})')
        axes[i, 3].set_xlabel('Matrix Size')
        axes[i, 3].set_ylabel('Difference (max norm)')
        axes[i, 3].set_xscale('log')
        axes[i, 3].set_yscale('log')
        axes[i, 3].legend()
        axes[i, 3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    print("Enhanced QR Benchmark")
    print("=" * 80)
    
    # Define test parameters
    sizes = [8, 16, 32, 64, 128, 256]
    conditions = [1.0, 10.0, 100.0]
    
    # Run benchmarks
    results = run_benchmark(sizes, conditions, num_runs=3)
    
    # Plot results
    plot_results(results, save_path="qr_benchmark_results.png")
    
    print("\nBenchmark completed. Results saved to qr_benchmark_results.png")