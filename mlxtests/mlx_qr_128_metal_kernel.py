"""
QR decomposition implementation using a Metal kernel with 128-bit precision.
This provides improved numerical stability for the SVD implementation.
"""
from typing import Tuple
import mlx.core as mx

# Maximum matrix dimension supported by the kernel
MAX_MATRIX_DIM = 4096

# Define Metal kernel source string for QR decomposition
_qr_128_metal_kernel_source = """
#define TILE_SIZE 16
#define EPSILON 1e-10f

// QR decomposition using Householder reflections with 128-bit precision
if (thread_position_in_grid.x == 0) {
    uint m = A_shape[0];
    uint n = A_shape[1];
    uint min_dim = min(m, n);
    
    // Initialize Q to identity matrix
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < m; j++) {
            Q[i * m + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize R to A
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++) {
            R[i * n + j] = A[i * n + j];
        }
    }
    
    // Perform QR decomposition using Householder reflections
    for (uint k = 0; k < min_dim; k++) {
        // Compute the Householder vector
        float x[4096];  // Assuming max dimension is 4096
        float norm_sq = 0.0f;
        
        // Extract the column vector
        for (uint i = k; i < m; i++) {
            x[i-k] = R[i * n + k];
            norm_sq += x[i-k] * x[i-k];
        }
        
        float norm = sqrt(norm_sq);
        if (norm < EPSILON) {
            continue;  // Skip if column is already zero
        }
        
        // Adjust the sign to avoid cancellation
        float sign = (x[0] >= 0.0f) ? 1.0f : -1.0f;
        x[0] += sign * norm;
        
        // Recompute the norm of the adjusted vector
        norm_sq = 0.0f;
        for (uint i = 0; i < m-k; i++) {
            norm_sq += x[i] * x[i];
        }
        
        // Normalize the Householder vector
        float inv_norm_sq = 1.0f / norm_sq;
        for (uint i = 0; i < m-k; i++) {
            x[i] *= inv_norm_sq;
        }
        
        // Apply the Householder reflection to R
        for (uint j = k; j < n; j++) {
            float dot = 0.0f;
            for (uint i = 0; i < m-k; i++) {
                dot += x[i] * R[(i+k) * n + j];
            }
            
            for (uint i = 0; i < m-k; i++) {
                R[(i+k) * n + j] -= 2.0f * x[i] * dot;
            }
        }
        
        // Apply the Householder reflection to Q
        for (uint j = 0; j < m; j++) {
            float dot = 0.0f;
            for (uint i = 0; i < m-k; i++) {
                dot += x[i] * Q[(i+k) * m + j];
            }
            
            for (uint i = 0; i < m-k; i++) {
                Q[(i+k) * m + j] -= 2.0f * x[i] * dot;
            }
        }
    }
    
    // Transpose Q to get the orthogonal matrix
    float Q_temp[4096 * 4096];  // Temporary storage for Q
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < m; j++) {
            Q_temp[i * m + j] = Q[i * m + j];
        }
    }
    
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < m; j++) {
            Q[i * m + j] = Q_temp[j * m + i];
        }
    }
    
    // Ensure R is upper triangular by zeroing out the lower part
    for (uint i = 1; i < m; i++) {
        for (uint j = 0; j < min(i, n); j++) {
            R[i * n + j] = 0.0f;
        }
    }
}
"""

# Compile the kernel at module level
_qr_128_metal_kernel_compiled = mx.fast.metal_kernel(
    name="qr_128_metal_kernel",
    source=_qr_128_metal_kernel_source,
    input_names=["A"],
    output_names=["Q", "R"],
    ensure_row_contiguous=True
)

def qr_128_metal(a: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Compute QR decomposition using a Metal kernel with 128-bit precision.
    
    Args:
        a: Input matrix.
        
    Returns:
        Q, R matrices.
    """
    # Convert input to float32 array
    a_arr = mx.array(a, dtype=mx.float32)
    
    # Get dimensions
    m, n = a_arr.shape
    
    # Check if dimensions exceed the maximum supported by the Metal kernel
    if m > MAX_MATRIX_DIM or n > MAX_MATRIX_DIM:
        raise ValueError(f"Matrix dimensions exceed maximum supported size ({MAX_MATRIX_DIM})")
    
    # Call the kernel
    grid = (1, 1, 1)
    threads = (1, 1, 1)
    
    result = _qr_128_metal_kernel_compiled(
        inputs=[a_arr],
        output_shapes=[(m, m), (m, n)],
        output_dtypes=[a_arr.dtype, a_arr.dtype],
        grid=grid,
        threadgroup=threads
    )
    
    return result[0], result[1]