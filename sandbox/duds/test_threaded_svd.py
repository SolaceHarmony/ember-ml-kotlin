"""
Test file for running the SVD kernel in multiple threads simultaneously.
"""
import time
import threading
import mlx.core as mx
import numpy as np
from ember_ml.backend.mlx.linearalg.svd_ops import svd

def run_svd(thread_id, n, results, times):
    """
    Run the SVD function and store the result and time taken.
    
    Args:
        thread_id: ID of the thread
        n: Size of the matrix
        results: Dictionary to store the results
        times: Dictionary to store the times
    """
    print(f"Thread {thread_id}: Starting SVD on {n}x{n} matrix")
    
    # Create a matrix with known singular values
    mx.random.seed(42 + thread_id)  # Different seed for each thread
    
    # Create orthogonal matrices using QR
    from ember_ml.backend.mlx.linearalg.qr_ops import qr
    U_rand = mx.random.normal((n, n), dtype=mx.float32)
    U, _ = qr(U_rand)
    
    # Create diagonal matrix with decreasing singular values
    s_values = mx.linspace(n, 1, n, dtype=mx.float32)
    
    V_rand = mx.random.normal((n, n), dtype=mx.float32)
    V, _ = qr(V_rand)
    
    # Create matrix A = U @ diag(s) @ V.T
    s_diag = mx.zeros((n, n), dtype=mx.float32)
    for i in range(n):
        indices = mx.array([[i, i]])
        from ember_ml.backend.mlx.tensor.ops.indexing import scatter
        s_diag = scatter(indices, s_values[i], s_diag.shape)
    
    A = mx.matmul(U, mx.matmul(s_diag, mx.transpose(V)))
    
    # Run SVD
    start_time = time.time()
    try:
        u, s, vh = svd(A)
        end_time = time.time()
        
        # Verify result: u @ diag(s) @ vh should be close to A
        s_diag = mx.zeros((u.shape[1], vh.shape[0]), dtype=mx.float32)
        for i in range(min(u.shape[1], vh.shape[0])):
            indices = mx.array([i, i])
            from ember_ml.backend.mlx.tensor.ops.indexing import scatter
            s_diag = scatter(indices, s[i], s_diag.shape)
        
        reconstruction = mx.matmul(u, mx.matmul(s_diag, vh))
        reconstruction_error = mx.mean(mx.abs(reconstruction - A)).item()
        
        results[thread_id] = {
            "success": True,
            "error": reconstruction_error
        }
        times[thread_id] = end_time - start_time
        
        print(f"Thread {thread_id}: SVD completed in {times[thread_id]:.4f}s, Reconstruction error: {reconstruction_error:.2e}")
    except Exception as e:
        results[thread_id] = {
            "success": False,
            "error": str(e)
        }
        times[thread_id] = time.time() - start_time
        print(f"Thread {thread_id}: SVD failed after {times[thread_id]:.4f}s with error: {e}")

def test_threaded_svd():
    """
    Test running SVD in multiple threads simultaneously.
    """
    print("Testing SVD in multiple threads")
    print("=" * 80)
    
    # Matrix sizes for each thread
    sizes = [32, 64]
    
    # Dictionaries to store results and times
    results = {}
    times = {}
    
    # Create threads
    threads = []
    for i, n in enumerate(sizes):
        thread = threading.Thread(target=run_svd, args=(i, n, results, times))
        threads.append(thread)
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    all_success = True
    for i, n in enumerate(sizes):
        if results[i]["success"]:
            print(f"Thread {i} ({n}x{n} matrix): Success in {times[i]:.4f}s, Error: {results[i]['error']:.2e}")
        else:
            print(f"Thread {i} ({n}x{n} matrix): Failed in {times[i]:.4f}s, Error: {results[i]['error']}")
            all_success = False
    
    if all_success:
        print("\nAll threads completed successfully!")
    else:
        print("\nSome threads failed!")

def test_sequential_svd():
    """
    Test running SVD sequentially for comparison.
    """
    print("\nTesting SVD sequentially")
    print("=" * 80)
    
    # Matrix sizes
    sizes = [32, 64]
    
    # Dictionaries to store results and times
    results = {}
    times = {}
    
    # Run SVD sequentially
    for i, n in enumerate(sizes):
        run_svd(i, n, results, times)
    
    # Print summary
    print("\nSummary:")
    print("-" * 80)
    all_success = True
    for i, n in enumerate(sizes):
        if results[i]["success"]:
            print(f"Run {i} ({n}x{n} matrix): Success in {times[i]:.4f}s, Error: {results[i]['error']:.2e}")
        else:
            print(f"Run {i} ({n}x{n} matrix): Failed in {times[i]:.4f}s, Error: {results[i]['error']}")
            all_success = False
    
    if all_success:
        print("\nAll runs completed successfully!")
    else:
        print("\nSome runs failed!")

if __name__ == "__main__":
    # First run SVD in multiple threads
    test_threaded_svd()
    
    # Then run SVD sequentially for comparison
    test_sequential_svd()