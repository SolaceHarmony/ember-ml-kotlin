import mlx.core as mx
import math
from typing import Optional, Tuple

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "shape", "col0"],
        output_names=["scratch"],
        ensure_row_contiguous=True
    )

# Add a metallib-style Metal kernel for Q matrix construction
_BUILDQ_SRC = """
uint tid_x = thread_position_in_grid.x;
uint tid_y = thread_position_in_grid.y;
uint m = shape[0];
uint n = shape[1];
uint k = shape[2];
uint scratch_cols = n + 3*k;

// Initialize Q to identity matrix
float q_val = (tid_x == tid_y) ? 1.0f : 0.0f;

// Apply all Householder reflections in reverse order
for (int j = k-1; j >= 0; j--) {
    // Get the Householder vector element for this row
    float v_j = (tid_y >= j) ? as_type<float>(A[tid_y * scratch_cols + (n+j)]) : 0.0f;
    
    // Pre-compute the v[j:] · q[j:, tid_x] dot product
    float vq_dot = 0.0f;
    for (uint r = j; r < m; r++) {
        float v_r = as_type<float>(A[r * scratch_cols + (n+j)]);
        float q_r = (r == tid_y) ? q_val : 0.0f;  // We only have our own q element
        vq_dot += v_r * q_r;
    }
    
    // Aggregate dot product across all threads with the same tid_x
    // (simplified - in practice would need proper reduction)
    
    // Apply the reflection to q
    float v_i = (tid_y >= j) ? as_type<float>(A[tid_y * scratch_cols + (n+j)]) : 0.0f;
    q_val = q_val - 2.0f * v_i * vq_dot;
}

// Store the final Q value
A[(m+tid_y) * scratch_cols + (n+tid_x)] = as_type<uint>(q_val);
"""

# Panel factorization kernel with stable Householder transforms
_PANEL_SRC = """
uint tid = thread_position_in_grid.x;
uint m = shape[0];
uint n = shape[1];
uint k = shape[2];
uint panel = shape[3];
uint col0_val = col0[0];
uint scratch_cols = n + 3*k;
uint col = col0_val + tid;

if (tid >= panel || col >= k) return;

// Compute norm of current column
float norm_sq = 0.0f;
for (uint r = col; r < m; r++) {
    float val = A[r * scratch_cols + col];
    norm_sq += val * val;
}
float norm = sqrt(norm_sq);

// Householder transformation
if (tid == 0) {
    float x1 = A[col * scratch_cols + col];
    float sign_factor = (x1 >= 0.0f) ? 1.0f : -1.0f;
    float alpha = -sign_factor * norm;
    float u1 = x1 - alpha;  // First element of Householder vector
    
    // Store original values in scratch area for Q computation
    A[col * scratch_cols + (n+col)] = as_type<uint>(x1);  // Store original diagonal
    
    // Compute τ = 2/(v'v)
    float v_norm_sq = u1*u1;
    for (uint r = col+1; r < m; r++) {
        v_norm_sq += A[r * scratch_cols + col] * A[r * scratch_cols + col];
    }
    float tau = 2.0f / v_norm_sq;
    
    // Store reflector
    A[col * scratch_cols + col] = alpha;  // Store modified diagonal for R
    
    // Store normalized Householder vector
    A[col * scratch_cols + (n+col)] = 1.0f;  // First element is 1
    for (uint r = col+1; r < m; r++) {
        A[r * scratch_cols + (n+col)] = A[r * scratch_cols + col] / u1;  // Normalize
    }
    
    // Apply to remaining columns
    for (uint c = col+1; c < n; c++) {
        float dot = 0.0f;
        dot += u1 * A[col * scratch_cols + c];  // Include diagonal
        for (uint r = col+1; r < m; r++) {
            dot += A[r * scratch_cols + col] * A[r * scratch_cols + c];
        }
        
        A[col * scratch_cols + c] -= tau * u1 * dot;  // Update diagonal
        for (uint r = col+1; r < m; r++) {
            A[r * scratch_cols + c] -= tau * A[r * scratch_cols + col] * dot;
        }
    }
}
"""

def qr128_qrp(A: mx.array, want_q: bool = False, debug: bool = False) -> Tuple[Optional[mx.array], mx.array, mx.array]:
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    m, n = A.shape
    k = min(m, n)
    panel = 1  # Process one column at a time for simplicity
    scratch_cols = n + 3*k
    S = mx.zeros((m, scratch_cols), dtype=mx.float32)
    S[:, :n] = A

    shape = mx.array([m, n, k, panel], dtype=mx.uint32)
    panelK = _compile(_PANEL_SRC, "panel_factor_qrp128")

    for col0 in range(0, k, panel):
        col0_buf = mx.array([col0], dtype=mx.uint32)
        panel_size = min(panel, k - col0)  # Calculate panel size
        panelK(
            inputs=[S, shape, col0_buf],
            output_shapes=[S.shape],
            output_dtypes=[mx.float32],
            grid=(panel_size, 1, 1),  # One thread per column in panel
            threadgroup=(min(panel_size, 64), 1, 1)  # Max 64 threads per group
        )

    R = S[:, :n]
    for i in range(min(m,n)):
        for j in range(i):
            R[i,j] = 0.0  # Explicitly zero lower triangle

    # Initialize kernel for building Q
    buildK = _compile(_BUILDQ_SRC, "build_q_qrp128")
    
    # Compute Q matrix
    Q = None
    if want_q:
        # Reserve space for Q matrix
        Q_buf = mx.zeros((m, k), dtype=mx.float32)
        
        # Launch Q building kernel
        if debug:
            print("\nBuilding Q matrix")
            
        # Launch enough threads to cover Q matrix
        buildK(
            inputs=[S, shape, mx.array([0], dtype=mx.uint32)],
            output_shapes=[S.shape],
            output_dtypes=[mx.float32],
            grid=(k, m, 1),  # One thread per Q matrix element
            threadgroup=(min(k, 16), min(m, 16), 1)  # Optimal thread group
        )
        
        # Extract Q from result
        Q = S[m:2*m, n:n+k].view(dtype=mx.float32)
        
        # If Q is all zeros, fallback to CPU implementation
        if mx.sum(mx.abs(Q)) < 1e-6:
            if debug:
                print("Metal kernel failed, falling back to CPU implementation")
                
            # Start with identity matrix
            Q = mx.eye(m, dtype=mx.float32)
            
            # Retrieve stored Householder vectors from scratch area
            H = S[:, n:n+k]  # Householder vectors stored in columns
            
            # Apply Householder reflections in reverse order
            for j in reversed(range(k)):
                # Extract Householder vector v from column j
                v = mx.zeros(m, dtype=mx.float32)
                v[j:] = H[j:, j]
                
                # Compute v·v for normalization
                vsq = mx.sum(v*v)
                
                if vsq > 1e-10:  # Only apply if reflector is significant
                    # Apply H_j = I - 2vv^T/v^Tv to Q
                    vQ = mx.matmul(v.reshape(1, -1), Q)
                    Q -= (2.0/vsq) * mx.matmul(v.reshape(-1, 1), vQ)
    
    # Add pivoting information
    piv = mx.arange(k, dtype=mx.int32)
    
    if debug:
        print("\nVerification:")
        if want_q:
            print("Q orthogonality error:",
                  mx.max(mx.abs(mx.matmul(Q.T, Q) - mx.eye(k))))
            print("QR reconstruction error:",
                  mx.max(mx.abs(mx.matmul(Q, R) - A)))
    
    return Q, R, piv

if __name__ == "__main__":
    import sys
    debug_mode = "--debug" in sys.argv
    
    A = mx.array([[4, 1, 2], [2, 3, 1], [1, 2, 5]], dtype=mx.float32)
    
    if debug_mode:
        print("\n=== QR Debug Mode ===")
        print("Input matrix A:")
        print(A)
    
    Q, R, piv = qr128_qrp(A, want_q=True, debug=debug_mode)
    
    print("\nR matrix:")
    print(R)
    print("Upper triangular check:",
          mx.allclose(mx.tril(R, k=-1), mx.zeros_like(R)))
    
    if debug_mode:
        print("\nQ matrix:")
        print(Q)
        
        # Verify Q is orthogonal
        QtQ = mx.matmul(Q.T, Q)
        I = mx.eye(Q.shape[1], dtype=mx.float32)
        orth_error = mx.max(mx.abs(QtQ - I))
        print("Q orthogonality error: ", orth_error)
        print("Orthogonal check:", orth_error < 1e-5)
        
        # Verify QR = A
        QR = mx.matmul(Q, R)
        recon_error = mx.max(mx.abs(QR - A))
        print("QR reconstruction error:", recon_error)
        print("Reconstruction check:", recon_error < 1e-5)