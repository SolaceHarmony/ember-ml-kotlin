import mlx.core as mx
import math
import time
from typing import Optional, Tuple

# Helper functions
def next_pow2(x):
    return 1 << (x - 1).bit_length()

def _compile(src, name):
    """Compile a Metal kernel with the given source code and name."""
    try:
        # MLX will wrap our code in a kernel function with appropriate parameters
        kernel = mx.fast.metal_kernel(
            name=name,
            source=src,
            input_names=["A", "shape", "col0_buf"],
            output_names=["A"],  # Output is the same as input A
            ensure_row_contiguous=True
        )
        print(f"Successfully compiled kernel: {name}")
        return kernel
    except Exception as e:
        print(f"Error compiling kernel {name}: {e}")
        raise

# Metal kernel sources - Note: MLX will wrap this code in a kernel function
_PANEL_SRC = """
#include <metal_stdlib>
using namespace metal;
#define TG_SIZE   64u
#define LIMBS     4u
#define LIMB_RADIX 4294967296.0f

// MLX will automatically create the kernel function and parameters
// We just need to provide the body of the function
const uint tid = tid_in.x;
const uint ltid = lid.x;
const uint m = shape[0], n = shape[1];
const uint k = shape[2], panel = shape[3];
const uint scratch_cols = shape[5];
const uint col0 = *col0_buf;

device uint* colA = A;
device uint* colV = A + m * scratch_cols + n;
device uint* tauBuf = A + m * scratch_cols + n + k;
device uint* pivBuf = tauBuf + k;

if (tid >= panel || tid + col0 >= k) return;

threadgroup uint sh[LIMBS][TG_SIZE];
threadgroup float shf[TG_SIZE];
threadgroup float tg_inv;

const uint col = col0 + tid;

// Copy A to V
for (uint r = ltid; r < m; r += TG_SIZE)
    colV[r * scratch_cols + col] = colA[r * scratch_cols + col];

threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute norm (128-bit via limbs + FP32 fast path)
uint loc[LIMBS] = {0};
float fp32 = 0;
for (uint r = ltid; r < m; r += TG_SIZE) {
    float v = as_type<float>(colV[r * scratch_cols + col]);
    fp32 = fma(v, v, fp32);
    ulong p = ulong(as_type<uint>(v)) * ulong(as_type<uint>(v));
    loc[0] += uint(p);
    uint c = p >> 32;
    for (uint i = 1; i < LIMBS; ++i) {
        uint t = loc[i] + c;
        c = (t < loc[i]);
        loc[i] = t;
        if (!c) break;
    }
}
for (uint i = 0; i < LIMBS; ++i) sh[i][ltid] = loc[i];
shf[ltid] = fp32;
threadgroup_barrier(mem_flags::mem_threadgroup);

// SIMD reduce fp32 for pivot selection
float simdf = shf[ltid];
for (uint off = TG_SIZE >> 1; off; off >>= 1) {
    simdf += (ltid < off) ? shf[ltid + off] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid < off) shf[ltid] = simdf;
}
if (ltid == 0) shf[0] = simdf;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Tree reduce limbs
for (uint l = 0; l < LIMBS; ++l) {
    for (uint off = TG_SIZE >> 1; off; off >>= 1) {
        if (ltid < off) sh[l][ltid] += sh[l][ltid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
float norm = sqrt(max(({
    float acc = 0, sc = 1;
    for (uint i = 0; i < LIMBS; ++i) {
        acc += float(sh[i][0]) * sc;
        sc *= LIMB_RADIX;
    }
    acc;
}), 1.0e-18f));

if (ltid == 0) {
    tg_inv = 1.0f / norm;
    tauBuf[col] = as_type<uint>(norm);
    pivBuf[col] = col;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Normalize v
float inv = tg_inv;
for (uint r = ltid; r < m; r += TG_SIZE) {
    float v = as_type<float>(colV[r * scratch_cols + col]) * inv;
    colV[r * scratch_cols + col] = as_type<uint>(v);
}
"""

_APPLY_SRC = """
#include <metal_stdlib>
using namespace metal;
#define BLK 128u
#define SIMD_SIZE 8u

// MLX will automatically create the kernel function and parameters
const uint m = shape[0], n = shape[1], k = shape[2], panel = shape[3];
const uint scratch_cols = shape[5];
const uint col0 = *col0_buf;
const uint blk_i = gid.y, blk_j = gid.x;
const uint row0 = blk_i * BLK + lid.y * SIMD_SIZE + sid.y;
const uint col0_global = blk_j * BLK + lid.x * SIMD_SIZE + sid.x + col0 + panel;

if (row0 >= m || col0_global >= n) return;

threadgroup float v_cache[BLK][SIMD_SIZE];
threadgroup float tau_cache[SIMD_SIZE];

if (lid.y == 0 && sid.y == 0) {
    for (uint p = sid.x; p < panel; p += SIMD_SIZE) {
        tau_cache[p % SIMD_SIZE] = as_type<float>(A[n + k + col0 + p]);
    }
}

for (uint p = lid.x; p < panel; p += SIMD_SIZE) {
    if (sid.x < SIMD_SIZE) {
        v_cache[lid.y * SIMD_SIZE + sid.y][p % SIMD_SIZE] = as_type<float>(A[row0 * scratch_cols + (n + col0 + p)]);
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

float acc = 0.0f;
for (uint p = 0; p < panel; p += SIMD_SIZE) {
    float v[SIMD_SIZE];
    float tau[SIMD_SIZE];
    for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
        v[i] = v_cache[lid.y * SIMD_SIZE + sid.y][i];
        tau[i] = tau_cache[i];
    }

    float a = as_type<float>(A[row0 * scratch_cols + col0_global]);
    for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
        acc += v[i] * a * tau[i];
    }
}

float newA = as_type<float>(A[row0 * scratch_cols + col0_global]) - 2.0f * acc;
A[row0 * scratch_cols + col0_global] = as_type<uint>(newA);
"""

_BUILDQ_SRC = """
#include <metal_stdlib>
using namespace metal;
#define TG 32u

// MLX will automatically create the kernel function and parameters
const uint m = shape[0], k = shape[2];
const uint scratch_cols = shape[5];
if (tid_in.x >= k || tid_in.y >= m) return;

float q = (tid_in.x == tid_in.y) ? 1.0f : 0.0f;
for (int p = k - 1; p >= 0; --p) {
    float v = as_type<float>(A[tid_in.y * scratch_cols + (m + p)]);
    float tau = as_type<float>(A[m * scratch_cols + m + p]);
    q -= 2.0f * tau * v * q;
}
A[m * scratch_cols + m + tid_in.y * scratch_cols + tid_in.x] = as_type<uint>(q);
"""

# Compile kernels
panelK = _compile(_PANEL_SRC, "panel_factor_qrp128")
applyK = _compile(_APPLY_SRC, "apply_update_qrp128")
buildK = _compile(_BUILDQ_SRC, "build_q_qrp128")

def qr128_qrp(A: mx.array, want_q: bool = False) -> Tuple[Optional[mx.array], mx.array, mx.array]:
    """
    Compute the QR decomposition with column pivoting of matrix A using MLX and Metal acceleration.
    
    Args:
        A: Input matrix of shape (m, n) with dtype float32
        want_q: Whether to explicitly form the Q matrix
        
    Returns:
        Q: Orthogonal matrix (if want_q=True, otherwise None)
        R: Upper triangular matrix
        piv: Pivot indices
    """
    # Input validation
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    # Get dimensions
    m, n = map(int, A.shape)
    k = min(m, n)
    panel = 64  # Panel size for blocking
    limbs = 4   # Number of limbs for high-precision arithmetic
    
    # Allocate scratch space
    scratch_cols = n + k + k + k  # A | V | τ | piv
    S = mx.zeros((m, scratch_cols), dtype=mx.uint32)
    S[:, :n] = A.view(dtype=mx.uint32)  # Copy A
    
    # Setup shape information
    shape = mx.array([m, n, k, panel, limbs, scratch_cols], dtype=mx.uint32)
    
    # P₀: Panel factorization
    start_time = time.time()
    for col0 in range(0, k, panel):
        col0_buf = mx.array([col0], dtype=mx.uint32)
        # For panel factorization, use one thread per column in the panel
        panel_size = min(panel, k - col0)
        grid = (1, 1, 1)
        threadgroup = (panel_size, 1, 1)
        
        # Print debug info
        print(f"Panel {col0}: grid={grid}, threadgroup={threadgroup}")
        
        # Execute the panel factorization kernel
        try:
            result = panelK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32],
                           grid=grid, threadgroup=threadgroup)
            
            # The kernel returns a list of arrays, get the first one
            S = result[0] if isinstance(result, list) else result
            
            # Debug: Check if S has been modified
            print(f"  Panel {col0} max value in S: {float(mx.max(S))}")
        except Exception as e:
            print(f"Error in panel factorization at col0={col0}: {e}")
            raise
        
        # P₁: Trailing update
        right0 = col0 + panel
        if right0 < n:
            # For trailing update, use 2D grid of blocks
            blocks = (math.ceil((n - right0) / 128), math.ceil(m / 128), 1)
            
            # Print debug info
            print(f"Trailing update {col0}: grid={blocks}, threadgroup=(8, 8, 1)")
            
            # Execute the trailing update kernel
            try:
                result = applyK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32],
                               grid=blocks, threadgroup=(8, 8, 1))
                
                # The kernel returns a list of arrays, get the first one
                S = result[0] if isinstance(result, list) else result
                
                # Debug: Check if S has been modified
                print(f"  Trailing {col0} max value in S: {float(mx.max(S))}")
            except Exception as e:
                print(f"Error in trailing update at col0={col0}: {e}")
                raise
    
    factorization_time = time.time() - start_time
    
    # P₂: Build explicit Q
    if want_q:
        start_time = time.time()
        col0_buf = mx.array([0], dtype=mx.uint32)  # Dummy value
        q_grid = (math.ceil(k / 32), math.ceil(m / 32), 1)
        
        # Print debug info
        print(f"Building Q: grid={q_grid}, threadgroup=(32, 1, 1)")
        
        # Execute the build Q kernel
        try:
            result = buildK(inputs=[S, shape, col0_buf], output_shapes=[S.shape], output_dtypes=[mx.uint32],
                           grid=q_grid, threadgroup=(32, 1, 1))
            
            # The kernel returns a list of arrays, get the first one
            S = result[0] if isinstance(result, list) else result
            
            # Debug: Check if S has been modified
            print(f"  Build Q max value in S: {float(mx.max(S))}")
        except Exception as e:
            print(f"Error in building Q: {e}")
            raise
        # Extract Q from S
        Q = S[:, n:n+k].view(dtype=A.dtype)
        
        # Debug: Check Q values
        print(f"  Q shape: {Q.shape}, min: {float(mx.min(Q))}, max: {float(mx.max(Q))}")
        q_time = time.time() - start_time
    else:
        Q = None
        q_time = 0
    
    # Extract results
    R = S[:, :n].view(dtype=A.dtype)
    piv = S[0, n+2*k:n+3*k].view(dtype=mx.int32)
    
    # Debug: Check R and piv values
    print(f"  R shape: {R.shape}, min: {float(mx.min(R))}, max: {float(mx.max(R))}")
    print(f"  piv shape: {piv.shape}, values: {piv[:min(10, k)].tolist()}")
    
    print(f"QR Factorization time: {factorization_time:.4f}s")
    if want_q:
        print(f"Q formation time: {q_time:.4f}s")
    
    return Q, R, piv

def norm(x, ord=None):
    """Compute the matrix or vector norm."""
    if ord is None:
        # Default to Frobenius norm for matrices, L2 norm for vectors
        if x.ndim > 1:
            ord = 'fro'
        else:
            ord = 2
            
    if ord == 'fro':
        # Frobenius norm
        return mx.sqrt(mx.sum(mx.square(x)))
    elif ord == 2 and x.ndim == 1:
        # L2 norm for vectors
        return mx.sqrt(mx.sum(mx.square(x)))
    else:
        # For other norms, we would need to implement them
        # This is a simplified version
        return mx.sqrt(mx.sum(mx.square(x)))

# Test and validate
if __name__ == "__main__":
    print("Testing QR decomposition with column pivoting...")
    
    # Test with different matrix sizes
    sizes = [(512, 512), (1024, 512), (512, 1024)]
    
    for m, n in sizes:
        print(f"\nTesting {m}x{n} matrix:")
        
        # Generate random test matrix
        A = mx.random.normal((m, n), dtype=mx.float32)
        
        # Debug: Check input matrix
        print(f"Input matrix A: shape={A.shape}, min={mx.min(A)}, max={mx.max(A)}")
        
        # Compute QR decomposition
        start_time = time.time()
        Q, R, piv = qr128_qrp(A, want_q=True)
        total_time = time.time() - start_time
        
        print(f"Total time: {total_time:.4f}s")
        
        # Create permutation matrix based on pivot indices
        P = mx.zeros((n, n), dtype=A.dtype)
        for i, p in enumerate(piv.tolist()):
            if i < n and p < n:
                P[i, p] = 1.0
        
        # Debug: Check permutation matrix
        print(f"Permutation matrix P: shape={P.shape}, sum={mx.sum(P)}")
        
        # Apply permutation to R
        R_perm = mx.matmul(R, P)
        
        # Validate reconstruction
        # For non-square matrices, we need to handle the dimensions carefully
        k = min(m, n)  # Define k here for validation
        if m >= n:
            # Tall matrix: Q is m x k, R_perm is k x n
            recon_error = norm(mx.matmul(Q, R_perm[:k, :]) - A) / norm(A)
        else:
            # Wide matrix: Q is m x m, R_perm is m x n
            recon_error = norm(mx.matmul(Q, R_perm) - A) / norm(A)
        print(f"‖QR−A‖/‖A‖ = {recon_error}")
        
        # Validate Q orthogonality
        # Q should be m x min(m,n)
        QtQ = mx.matmul(Q.T, Q)
        eye = mx.eye(min(m, n), dtype=mx.float32)
        orth_error = norm(QtQ - eye) / norm(eye)
        print(f"‖QᵀQ−I‖/‖I‖ = {orth_error}")
        
        # Check if R is upper triangular
        tril_sum = mx.sum(mx.abs(mx.tril(R, -1)))
        print(f"Sum of below-diagonal elements in R: {tril_sum}")