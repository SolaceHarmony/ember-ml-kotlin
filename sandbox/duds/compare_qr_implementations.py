import mlx.core as mx
import time
import numpy as np
import matplotlib.pyplot as plt
from tiled_hpc_qr_experiment import tiled_hpc_qr
from sandbox.duds.enhanced_tiled_hpc_qr import enhanced_tiled_hpc_qr

def compare_implementations(sizes=[(50, 50), (100, 100), (200, 200), (500, 500)], repeat=3):
    """
    Compare the original and enhanced QR implementations across different matrix sizes.
    
    Args:
        sizes: List of (m, n) tuples for matrix sizes to test
        repeat: Number of times to repeat each test for averaging
    """
    results = []
    
    for size in sizes:
        m, n = size
        print(f"\nComparing with matrix size {m}x{n}...")
        
        # Generate random matrix
        A = mx.random.normal((m, n), dtype=mx.float32)
        
        # Test original implementation
        original_times = []
        for i in range(repeat):
            start_time = time.time()
            Q_original, R_original = tiled_hpc_qr(A)
            end_time = time.time()
            original_times.append(end_time - start_time)
            
        avg_original_time = sum(original_times) / repeat
        print(f"Original QR completed in {avg_original_time:.4f} seconds (avg of {repeat} runs)")
        
        # Check orthogonality and reconstruction for original
        original_ortho_error = mx.mean(mx.abs(mx.matmul(Q_original.T, Q_original) - mx.eye(Q_original.shape[0]))).item()
        original_recon_error = mx.mean(mx.abs(mx.matmul(Q_original, R_original) - A)).item()
        print(f"Original orthogonality error: {original_ortho_error:.6e}")
        print(f"Original reconstruction error: {original_recon_error:.6e}")
        
        # Test enhanced implementation
        enhanced_times = []
        for i in range(repeat):
            start_time = time.time()
            Q_enhanced, R_enhanced = enhanced_tiled_hpc_qr(A)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
            
        avg_enhanced_time = sum(enhanced_times) / repeat
        print(f"Enhanced QR completed in {avg_enhanced_time:.4f} seconds (avg of {repeat} runs)")
        
        # Check orthogonality and reconstruction for enhanced
        enhanced_ortho_error = mx.mean(mx.abs(mx.matmul(Q_enhanced.T, Q_enhanced) - mx.eye(Q_enhanced.shape[0]))).item()
        enhanced_recon_error = mx.mean(mx.abs(mx.matmul(Q_enhanced, R_enhanced) - A)).item()
        print(f"Enhanced orthogonality error: {enhanced_ortho_error:.6e}")
        print(f"Enhanced reconstruction error: {enhanced_recon_error:.6e}")
        
        # Compare the two implementations
        speedup = avg_original_time / avg_enhanced_time
        ortho_improvement = original_ortho_error / enhanced_ortho_error if enhanced_ortho_error > 0 else float('inf')
        recon_improvement = original_recon_error / enhanced_recon_error if enhanced_recon_error > 0 else float('inf')
        
        print(f"Speedup: {speedup:.2f}x")
        print(f"Orthogonality improvement: {ortho_improvement:.2f}x")
        print(f"Reconstruction improvement: {recon_improvement:.2f}x")
        
        # Compare with native MLX QR if available
        try:
            start_time = time.time()
            Q_native, R_native = mx.linalg.qr(A, stream=mx.cpu)
            end_time = time.time()
            native_time = end_time - start_time
            
            native_ortho_error = mx.mean(mx.abs(mx.matmul(Q_native.T, Q_native) - mx.eye(Q_native.shape[0]))).item()
            native_recon_error = mx.mean(mx.abs(mx.matmul(Q_native, R_native) - A)).item()
            
            print(f"Native MLX QR completed in {native_time:.4f} seconds")
            print(f"Native orthogonality error: {native_ortho_error:.6e}")
            print(f"Native reconstruction error: {native_recon_error:.6e}")
            
            result = {
                "size": size,
                "original_time": avg_original_time,
                "enhanced_time": avg_enhanced_time,
                "native_time": native_time,
                "original_ortho_error": original_ortho_error,
                "enhanced_ortho_error": enhanced_ortho_error,
                "native_ortho_error": native_ortho_error,
                "original_recon_error": original_recon_error,
                "enhanced_recon_error": enhanced_recon_error,
                "native_recon_error": native_recon_error,
                "speedup": speedup,
                "ortho_improvement": ortho_improvement,
                "recon_improvement": recon_improvement
            }
        except Exception as e:
            print(f"Native MLX QR failed: {e}")
            result = {
                "size": size,
                "original_time": avg_original_time,
                "enhanced_time": avg_enhanced_time,
                "original_ortho_error": original_ortho_error,
                "enhanced_ortho_error": enhanced_ortho_error,
                "original_recon_error": original_recon_error,
                "enhanced_recon_error": enhanced_recon_error,
                "speedup": speedup,
                "ortho_improvement": ortho_improvement,
                "recon_improvement": recon_improvement
            }
        
        results.append(result)
        print("-" * 40)
    
    return results

def plot_comparison_results(results):
    """
    Plot the comparison results.
    
    Args:
        results: List of result dictionaries from compare_implementations
    """
    sizes = [f"{r['size'][0]}x{r['size'][1]}" for r in results]
    
    # Performance comparison
    plt.figure(figsize=(12, 8))
    
    # Plot execution times
    plt.subplot(2, 2, 1)
    original_times = [r["original_time"] for r in results]
    enhanced_times = [r["enhanced_time"] for r in results]
    native_times = [r.get("native_time", 0) for r in results]
    
    x = np.arange(len(sizes))
    width = 0.25
    
    plt.bar(x - width, original_times, width, label='Original')
    plt.bar(x, enhanced_times, width, label='Enhanced')
    if all(t > 0 for t in native_times):
        plt.bar(x + width, native_times, width, label='Native MLX')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison')
    plt.xticks(x, sizes)
    plt.legend()
    
    # Plot speedup
    plt.subplot(2, 2, 2)
    speedups = [r["speedup"] for r in results]
    plt.bar(x, speedups, width*2)
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup (x)')
    plt.title('Enhanced vs Original Speedup')
    plt.xticks(x, sizes)
    
    # Plot orthogonality error
    plt.subplot(2, 2, 3)
    original_ortho = [r["original_ortho_error"] for r in results]
    enhanced_ortho = [r["enhanced_ortho_error"] for r in results]
    native_ortho = [r.get("native_ortho_error", 0) for r in results]
    
    plt.bar(x - width, original_ortho, width, label='Original')
    plt.bar(x, enhanced_ortho, width, label='Enhanced')
    if all(e > 0 for e in native_ortho):
        plt.bar(x + width, native_ortho, width, label='Native MLX')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Orthogonality Error')
    plt.title('Orthogonality Error Comparison')
    plt.xticks(x, sizes)
    plt.yscale('log')
    plt.legend()
    
    # Plot reconstruction error
    plt.subplot(2, 2, 4)
    original_recon = [r["original_recon_error"] for r in results]
    enhanced_recon = [r["enhanced_recon_error"] for r in results]
    native_recon = [r.get("native_recon_error", 0) for r in results]
    
    plt.bar(x - width, original_recon, width, label='Original')
    plt.bar(x, enhanced_recon, width, label='Enhanced')
    if all(e > 0 for e in native_recon):
        plt.bar(x + width, native_recon, width, label='Native MLX')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Comparison')
    plt.xticks(x, sizes)
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('qr_comparison_results.png')
    plt.show()

def test_ill_conditioned_matrices():
    """
    Test both implementations on ill-conditioned matrices to compare numerical stability.
    """
    condition_numbers = [10, 100, 1000, 10000, 100000]
    results = []
    
    print("\nTesting numerical stability with ill-conditioned matrices...")
    
    for cond in condition_numbers:
        print(f"\nTesting with condition number {cond}...")
        
        # Create a matrix with specified condition number
        m, n = 100, 100
        U = mx.random.normal((m, m), dtype=mx.float32)
        U, _ = mx.linalg.qr(U, stream=mx.cpu)  # Orthogonalize
        
        V = mx.random.normal((n, n), dtype=mx.float32)
        V, _ = mx.linalg.qr(V, stream=mx.cpu)  # Orthogonalize
        
        # Create singular values with specified condition number
        s = mx.array([1.0 * (cond ** (-i / (min(m, n) - 1))) for i in range(min(m, n))], dtype=mx.float32)
        S = mx.zeros((m, n), dtype=mx.float32)
        for i in range(min(m, n)):
            S = S.at[i, i].add(s[i])
        
        # Create matrix A = U*S*V^T with specified condition number
        A = mx.matmul(mx.matmul(U, S), V.T)
        
        # Test original implementation
        Q_original, R_original = tiled_hpc_qr(A)
        original_ortho_error = mx.mean(mx.abs(mx.matmul(Q_original.T, Q_original) - mx.eye(Q_original.shape[0]))).item()
        original_recon_error = mx.mean(mx.abs(mx.matmul(Q_original, R_original) - A)).item()
        
        # Test enhanced implementation
        Q_enhanced, R_enhanced = enhanced_tiled_hpc_qr(A)
        enhanced_ortho_error = mx.mean(mx.abs(mx.matmul(Q_enhanced.T, Q_enhanced) - mx.eye(Q_enhanced.shape[0]))).item()
        enhanced_recon_error = mx.mean(mx.abs(mx.matmul(Q_enhanced, R_enhanced) - A)).item()
        
        print(f"Original orthogonality error: {original_ortho_error:.6e}")
        print(f"Enhanced orthogonality error: {enhanced_ortho_error:.6e}")
        print(f"Original reconstruction error: {original_recon_error:.6e}")
        print(f"Enhanced reconstruction error: {enhanced_recon_error:.6e}")
        
        ortho_improvement = original_ortho_error / enhanced_ortho_error if enhanced_ortho_error > 0 else float('inf')
        recon_improvement = original_recon_error / enhanced_recon_error if enhanced_recon_error > 0 else float('inf')
        
        print(f"Orthogonality improvement: {ortho_improvement:.2f}x")
        print(f"Reconstruction improvement: {recon_improvement:.2f}x")
        
        results.append({
            "condition": cond,
            "original_ortho_error": original_ortho_error,
            "enhanced_ortho_error": enhanced_ortho_error,
            "original_recon_error": original_recon_error,
            "enhanced_recon_error": enhanced_recon_error,
            "ortho_improvement": ortho_improvement,
            "recon_improvement": recon_improvement
        })
        
        print("-" * 40)
    
    return results

def plot_stability_results(results):
    """
    Plot the stability test results.
    
    Args:
        results: List of result dictionaries from test_ill_conditioned_matrices
    """
    conditions = [r["condition"] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot orthogonality error
    plt.subplot(1, 2, 1)
    original_ortho = [r["original_ortho_error"] for r in results]
    enhanced_ortho = [r["enhanced_ortho_error"] for r in results]
    
    plt.plot(conditions, original_ortho, 'o-', label='Original')
    plt.plot(conditions, enhanced_ortho, 's-', label='Enhanced')
    
    plt.xlabel('Condition Number')
    plt.ylabel('Orthogonality Error')
    plt.title('Orthogonality Error vs Condition Number')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot reconstruction error
    plt.subplot(1, 2, 2)
    original_recon = [r["original_recon_error"] for r in results]
    enhanced_recon = [r["enhanced_recon_error"] for r in results]
    
    plt.plot(conditions, original_recon, 'o-', label='Original')
    plt.plot(conditions, enhanced_recon, 's-', label='Enhanced')
    
    plt.xlabel('Condition Number')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Condition Number')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig('qr_stability_results.png')
    plt.show()

if __name__ == "__main__":
    print("=" * 80)
    print("Comparing Original vs Enhanced Tiled HPC QR Implementations")
    print("=" * 80)
    
    # Compare performance and accuracy
    results = compare_implementations(sizes=[(50, 50), (100, 100), (200, 200), (400, 400)])
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Size':>10} | {'Orig Time':>10} | {'Enh Time':>10} | {'Speedup':>10} | {'Ortho Imp':>10} | {'Recon Imp':>10}")
    print("-" * 80)
    
    for r in results:
        size_str = f"{r['size'][0]}x{r['size'][1]}"
        print(f"{size_str:>10} | {r['original_time']:>10.4f} | {r['enhanced_time']:>10.4f} | {r['speedup']:>10.2f}x | {r['ortho_improvement']:>10.2f}x | {r['recon_improvement']:>10.2f}x")
    
    print("-" * 80)
    
    # Test numerical stability
    stability_results = test_ill_conditioned_matrices()
    
    # Print stability summary
    print("\nNumerical Stability Summary:")
    print("-" * 80)
    print(f"{'Condition':>10} | {'Orig Ortho':>12} | {'Enh Ortho':>12} | {'Ortho Imp':>10} | {'Orig Recon':>12} | {'Enh Recon':>12} | {'Recon Imp':>10}")
    print("-" * 80)
    
    for r in stability_results:
        print(f"{r['condition']:>10} | {r['original_ortho_error']:>12.2e} | {r['enhanced_ortho_error']:>12.2e} | {r['ortho_improvement']:>10.2f}x | {r['original_recon_error']:>12.2e} | {r['enhanced_recon_error']:>12.2e} | {r['recon_improvement']:>10.2f}x")
    
    print("-" * 80)
    
    try:
        # Generate plots if matplotlib is available
        plot_comparison_results(results)
        plot_stability_results(stability_results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\nComparison completed!")