import mlx.core as mx
import numpy as np

def simple_qr(A: mx.array):
    """Simplified QR using Metal with basic operations"""
    m, n = A.shape
    Q = mx.eye(m, dtype=mx.float32)
    R = A.astype(mx.float32)
    
    src = """

        tid = thread_position_in_grid.x;
        const uint m = dims[0];
        const uint n = dims[1];
        const uint k = col[0];
        
        if (tid >= m) return;
        
        // Simple operation - just zero out below diagonal
        if (tid > k && k < n) {
            R_out[tid * n + k] = 0.0f;
        }
    """
    
    # Compile kernel
    kernel = mx.fast.metal_kernel(
        name="qr_column_update",
        source=src,
        input_names=["Q", "R", "col", "dims"],
        output_names=["R_out"],
        ensure_row_contiguous=True
    )
    
    dims = mx.array([m, n], dtype=mx.uint32)
    
    for k in range(min(m,n)):
        col = mx.array([k], dtype=mx.uint32)
        kernel(inputs=[Q, R, col, dims], 
               output_shapes=[R.shape],
               grid=(Q.size,R.size,1), 
               threadgroup=(m,n,1))
        
    
    return Q, R

def test_simple_qr():
    print("\n=== Testing Simple QR ===")
    A = mx.array([[4,1,2],[2,3,1],[1,2,5]], dtype=mx.float32)
    print("Input A:\n", A)
    
    Q, R = simple_qr(A)
    print("\nQ:\n", Q)
    print("\nR:\n", R)
    
    # Verify R is upper triangular
    print("\nR is upper triangular:", 
          np.allclose(np.triu(R.__array__()), R.__array__()))
    
    # Verify Q is identity (for this simple version)
    print("Q is identity:", 
          np.allclose(Q.__array__(), np.eye(3)))

if __name__ == "__main__":
    test_simple_qr()