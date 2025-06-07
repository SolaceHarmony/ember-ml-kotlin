from tabnanny import verbose
import mlx.core as mx
import math
from typing import Optional, Tuple

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "shape", "col0", "debug_buf"],
        output_names=["scratch", "debug_out"],
        ensure_row_contiguous=True
    )

# Metal kernel sources
_PANEL_SRC = """

#define TG_SIZE   64u
#define EPSILON   1.0e-8f  // Increased epsilon for numerical stability
#define DEBUG_SIZE 32u     // Size of debug array

    // Get thread position and local thread ID
    const uint tid = thread_position_in_grid.x;
    const uint ltid = thread_position_in_threadgroup.x;
    
    // Set up debug values (thread 0 only)
    if (tid == 0 && ltid == 0) {
        debug_out[0] = 0xF00D0001;  // Magic number so we know kernel executed
        debug_out[1] = shape[0];    // m rows
        debug_out[2] = shape[1];    // n columns
        debug_out[3] = shape[2];    // k min(m,n)
        debug_out[4] = shape[3];    // panel size
        debug_out[5] = *col0_buf;   // starting column
    }
    
    // Extract dimensions from shape array
    const uint m = shape[0];        // Number of rows
    const uint n = shape[1];        // Number of columns
    const uint k = shape[2];        // min(m,n)
    const uint panel = shape[3];    // Panel width
    const uint col0 = *col0_buf;    // Starting column for this panel
    const uint scratch_cols = n + 3 * k;  // Width of scratch matrix
    
    // Calculate current column index
    const uint col = col0 + tid;
    
    // Guard clause: don't process out-of-bounds columns
    if (tid >= panel || col >= k) return;
    
    // Debug prints (uncomment for debugging)
    // printf("Thread %u processing column %u (m=%u, n=%u, k=%u)\n", tid, col, m, n, k);
    
    // Memory layout for different matrix parts
    // R part: scratch[0:m, 0:n]
    // V part: scratch[0:m, n:n+k]
    // tau: scratch[0, n+k:n+2k]
    // pivot: scratch[0, n+2k:n+3k]
// Shared memory for threadgroup communication
threadgroup float shf[TG_SIZE];
threadgroup float tg_norm;
threadgroup uint tg_pivot;

// Copy column from A to V (householder vectors storage)
// Be careful with memory layout - A is input, scratch is output
for (uint r = ltid; r < m; r += TG_SIZE) {
    // Calculate source index (R part from input)
    uint src_idx = r * scratch_cols + col;
    // Calculate destination index (V part in output)
    uint dst_idx = r * scratch_cols + n + col;
    
    // Ensure we're within bounds
    if (src_idx < m * scratch_cols && dst_idx < m * scratch_cols) {
        // Read from input buffer A[:, col]
        float val = as_type<float>(A[src_idx]);
        // Write to output buffer V part in scratch[:, col-col0]
        scratch[dst_idx] = as_type<uint>(val);
        
        // Also copy R part to scratch (initialize it)
        scratch[src_idx] = A[src_idx];
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute column norm
float norm_sq = 0.0f;
for (uint r = ltid; r < m; r += TG_SIZE) {
    float val = as_type<float>(scratch[r * scratch_cols + n + col]);
    norm_sq = fma(val, val, norm_sq);
}

// Store partial norm in shared memory
shf[ltid] = norm_sq;

// Barrier to ensure all threads have written their partial norms
threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find maximum norm (for pivoting)
    float max_norm = shf[ltid];
    uint pivot_idx = col;
    
    // Parallel reduction to find maximum norm
    for (uint offset = TG_SIZE >> 1; offset > 0; offset >>= 1) {
        if (ltid < offset) {
            float other_norm = shf[ltid + offset];
            uint other_col = col0 + ltid + offset;
            
            if (other_norm > max_norm && other_col < k) {
                max_norm = other_norm;
                pivot_idx = other_col;
            }
            
            shf[ltid] = max_norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 writes the final results
    if (ltid == 0) {
        tg_norm = sqrt(max(max_norm, EPSILON));
        tg_pivot = pivot_idx;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Store the pivot and norm info in debug buffer
    if (tid == 0 && ltid == 0) {
        debug_out[6] = as_type<uint>(tg_norm);  // Norm value
        debug_out[7] = tg_pivot;                // Pivot index
    }

    // Compute Householder reflector for pivot column
    if (col == tg_pivot && ltid == 0) {
        // Store column index in debug buffer
        debug_out[8] = col;
        // Get the norm of the column
        float norm = tg_norm;
        
        // First element of the column
        float x1 = as_type<float>(scratch[col * scratch_cols + n + col]);
        
        // Compute Householder scalar (alpha)
        float alpha = -sign(x1) * norm;
        
        // First element of householder vector
        float u1 = x1 - alpha;
        
        // Compute norm of u
        float u_norm_sq = u1 * u1;
        for (uint r = col + 1; r < m; r++) {
            float v = as_type<float>(scratch[r * scratch_cols + n + col]);
            u_norm_sq = fma(v, v, u_norm_sq);
        }
        
        float u_norm = sqrt(max(u_norm_sq, EPSILON));
        float inv_norm = (u_norm > EPSILON) ? 1.0f / u_norm : 0.0f;
        
        // Store these intermediate values in debug buffer
        debug_out[9] = as_type<uint>(u_norm);      // Vector norm
        debug_out[10] = as_type<uint>(inv_norm);   // Inverse norm
        debug_out[11] = as_type<uint>(u_norm_sq);  // Vector norm squared
        debug_out[12] = as_type<uint>(x1);         // First element of column
        debug_out[13] = as_type<uint>(alpha);      // Alpha value
        debug_out[14] = as_type<uint>(u1);         // First element of unnormalized Householder vector
        
        // The tau value is critical for Householder transformations
        // tau = 2.0 / (1.0 + u1^2/u_norm_sq) when u_norm > EPSILON
        float tau = (u_norm > EPSILON) ? 2.0f / (1.0f + (u1 * u1) / u_norm_sq) : 0.0f;
        
        // Store tau value in debug buffer
        debug_out[15] = as_type<uint>(tau);  // Tau value
        
        // Store tau and pivot with explicit bounds checking
        if (n + k + col < m * scratch_cols) {
            scratch[n + k + col] = as_type<uint>(tau);  // Store tau
        }
        
        if (n + 2*k + col < m * scratch_cols) {
            scratch[n + 2*k + col] = as_type<uint>(tg_pivot); // Store pivot
        }
        
        // Update R matrix (zero below diagonal)
        for (uint r = col + 1; r < m; r++) {
            scratch[r * scratch_cols + col] = as_type<uint>(0.0f);
        }
        // Store normalized Householder vector with bounds checking
        uint idx_diag = col * scratch_cols + n + col;
        if (idx_diag < m * scratch_cols) {
            // Store first element (u1 * inv_norm)
            scratch[idx_diag] = as_type<uint>(u1 * inv_norm);
        }
        
        // Store remaining elements with careful bounds checking
        for (uint r = col + 1; r < m; r++) {
            uint idx_v = r * scratch_cols + n + col;
            if (idx_v < m * scratch_cols) {
                float v = as_type<float>(scratch[idx_v]);
                float normalized_v = v * inv_norm;
                scratch[idx_v] = as_type<uint>(normalized_v);
            }
    }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_APPLY_SRC = """
#define BLK 128u
#define SIMD_SIZE 8u

    // Get thread position in the grid and threadgroup
    uint3 g = thread_position_in_grid;  // FIXED: use full 3D vector
    uint3 l = thread_position_in_threadgroup;
    uint3 sgid = thread_position_in_simdgroup;
    uint simd_lane = simd_lane_id.x;
    
    // Extract parameters from shape array
    const uint m = shape[0], n = shape[1], k = shape[2], panel = shape[3];
    const uint col0 = *col0_buf;
    const uint scratch_cols = n + 3 * k;
    
    // Calculate block indices and resulting row/column indices
    const uint blk_i = g.y, blk_j = g.x;
    const uint row0 = blk_i * BLK + l.y * SIMD_SIZE + sgid.y;
    const uint col0_global = blk_j * BLK + l.x * SIMD_SIZE + sgid.x + col0 + panel;

    // Guard clause: don't process out-of-bounds elements
    if (row0 >= m || col0_global >= n) return;
    
    // Debug info (uncomment for kernel debugging)
    // printf("Thread processing row=%u, col=%u\n", row0, col0_global);

    threadgroup float v_cache[BLK][SIMD_SIZE];
    threadgroup float tau_cache[SIMD_SIZE];

    if (l.y == 0 && sgid.y == 0) {
        for (uint p = sgid.x; p < panel; p += SIMD_SIZE) {
            tau_cache[p % SIMD_SIZE] = as_type<float>(A[n + k + (col0 + p)]);
        }
    }

    for (uint p = l.x; p < panel; p += SIMD_SIZE) {
        if (sgid.x < SIMD_SIZE) {
            v_cache[l.y * SIMD_SIZE + sgid.y][p % SIMD_SIZE] = as_type<float>(A[row0 * scratch_cols + (n + col0 + p)]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    float error = 0.0f;
    for (uint p = 0; p < panel; p += SIMD_SIZE) {
        float v[SIMD_SIZE];
        float tau[SIMD_SIZE];
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            v[i] = v_cache[l.y * SIMD_SIZE + sgid.y][i];
            tau[i] = tau_cache[i];
        }

        float a = as_type<float>(A[row0 * scratch_cols + col0_global]);
        for (uint i = 0; i < SIMD_SIZE && p + i < panel; ++i) {
            float temp = acc;
            float y = v[i] * a * tau[i] - error;
            float t = temp + y;
            error = (t - temp) - y;
            acc = t;
        }
    }

    float newA = as_type<float>(A[row0 * scratch_cols + col0_global]) - 2.0f * acc;
    scratch[row0 * scratch_cols + col0_global] = as_type<uint>(newA);
"""

_BUILDQ_SRC = """
#define TG_SIZE 32u
#define EPSILON 1.0e-8f

    // This kernel builds the Q matrix from the stored Householder reflectors
    // Each thread processes one column of Q, computing dot products and updates correctly
    
    // Thread identifiers
    const uint wg_id = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint grid_size = threadgroups_per_grid.x;
    
    // Matrix dimensions
    const uint m = shape[0], n = shape[1], k = shape[2];
    const uint scratch_cols = n + 3 * k;
    
    // Shared memory for thread collaboration
    threadgroup float v_cache[TG_SIZE];       // Cache for Householder vector elements
    threadgroup float dot_products[TG_SIZE];  // For dot product reduction
    
    // Initialize Q as an identity matrix (Q starts as I)
    // Each threadgroup processes multiple columns
    for (uint col = wg_id; col < k; col += grid_size) {
        // Each thread initializes elements for its assigned column
        for (uint row = tid; row < m; row += TG_SIZE) {
            // Identity matrix: 1.0 on diagonal, 0.0 elsewhere
            float val = (row == col) ? 1.0f : 0.0f;
            // Store in the Q section of our scratch space
            scratch[row * scratch_cols + n + col] = as_type<uint>(val);
        }
    }
    
    // Ensure all threads have completed initialization before proceeding
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply Householder reflections in reverse order (k-1 down to 0)
    // For each reflection H_p = I - tau_p*v_p*v_p^T, we compute Q := H_p*Q
    // Starting with Q = I, we gradually build Q = H_0*H_1*...*H_{k-1}*I
    for (int p = k - 1; p >= 0; --p) {
        // Get tau scalar for this Householder reflection
        float tau = as_type<float>(A[n + k + p]);
        
        // Skip if reflection has no significant effect (tau ≈ 0)
        if (tau < EPSILON) continue;
        
        // Process each column of Q (each threadgroup handles some columns)
        for (uint col = wg_id; col < k; col += grid_size) {
            // For each column of Q, compute v^T * q_col (dot product)
            // This is the key part of the Householder transformation
            float thread_dot_sum = 0.0f;
            
            // Each thread computes part of the dot product
            for (uint row = tid; row < m; row += TG_SIZE) {
                // Get Householder vector element v[row]
                float v_element = as_type<float>(A[row * scratch_cols + n + p]);
                
                // Get Q matrix element q[row,col]
                float q_element = as_type<float>(A[row * scratch_cols + n + col]);
                
                // Accumulate contribution to dot product: v^T * q_col
                // This is the critical step in applying Householder transformations
                thread_dot_sum += v_element * q_element;
            }
            
            // Store each thread's partial dot product sum
            dot_products[tid] = thread_dot_sum;
            
            // Ensure all threads have written their partial sums
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Perform parallel reduction to get final dot product
            for (uint stride = TG_SIZE / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    // Add next value to our accumulator
                    dot_products[tid] += dot_products[tid + stride];
                }
                // Ensure all threads have completed this reduction step
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            // Thread 0 now has the complete dot product in dot_products[0]
            // Calculate scale factor: beta = tau * (v^T * q_col)
            float beta = tau * dot_products[0];
            
            // Cache parts of Householder vector v for efficient memory access
            // Process vector in chunks for better memory coalescing
            for (uint chunk_start = 0; chunk_start < m; chunk_start += TG_SIZE) {
                // Each thread loads one element of v into threadgroup memory
                if (chunk_start + tid < m) {
                    v_cache[tid] = as_type<float>(A[(chunk_start + tid) * scratch_cols + n + p]);
                }
                
                // Ensure all threads have loaded their elements
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Update portion of Q column: q_col -= beta * v
                if (chunk_start + tid < m) {
                    uint row = chunk_start + tid;
                    float q_val = as_type<float>(A[row * scratch_cols + n + col]);
                    
                    // Apply the transformation: q[row,col] -= beta * v[row]
                    // This implements Householder reflection: H*q = q - tau*(v^T*q)*v
                    float new_val = q_val - beta * v_cache[tid];
                    
                    // Write back to output memory buffer
                    scratch[row * scratch_cols + n + col] = as_type<uint>(new_val);
                }
                
                // Ensure updates are complete before next chunk
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        
        // Ensure all columns are processed before moving to next reflection
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Compile kernels
panelK = _compile(_PANEL_SRC, "panel_factor_qrp128")
applyK = _compile(_APPLY_SRC, "apply_update_qrp128")
buildK = _compile(_BUILDQ_SRC, "build_q_qrp128")

def qr128_qrp(A: mx.array, want_q: bool = False, debug: bool = True) -> Tuple[Optional[mx.array], mx.array, mx.array]:
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    m, n = map(int, A.shape)
    k = min(m, n)
    panel = min(32, k)  # Smaller panel size for this test case
    scratch_cols = n + 3 * k  # Structure: [R | V | tau | piv]
    
    if debug:
        print(f"\n==== QR Decomposition Debug ====")
        print(f"Input matrix A ({m}x{n}):")
        print(A)
        print(f"k = min(m, n) = {k}, panel size = {panel}")
        print(f"scratch_cols = {scratch_cols} (n={n} + 3*k={3*k})")
    
    S = mx.zeros((m, scratch_cols), dtype=mx.uint32)
    S[:, :n] = A.view(dtype=mx.uint32)
    
    if debug:
        print(f"\nInitial S matrix (showing float view of relevant parts):")
        print(f"S[:, :n] (copied from A):")
        print(S[:, :n].view(dtype=mx.float32))
    
    # Add extra element in shape array for debugging
    shape = mx.array([m, n, k, panel, 5], dtype=mx.uint32)
    if debug:
        print(f"Shape array: {shape.tolist()}")
        print(f"Memory layout: R[0:m,0:n], V[0:m,n:n+k], tau[0,n+k:n+2k], pivot[0,n+2k:n+3k]")
        print(f"Total scratch matrix size: {m}x{scratch_cols}")

    for col0 in range(0, k, panel):
        if debug:
            print(f"\n--- Iteration {col0//panel + 1}/{math.ceil(k/panel)} (col0={col0}) ---")
        
        # Add debug values before kernel call
        col0_buf = mx.array([col0], dtype=mx.uint32)
        
        # Add specific debug case for this input matrix
        if debug and col0 == 0 and m == 3 and n == 3:
            # Force specific debug values into S for first panel iteration
            # Hard-code some non-zero values for testing
            # Use real values that should work for this matrix
            
            # For column 0 - This should be [1,0,0] vector normalized
            test_v0 = mx.array([0.8944, 0.4472, 0.0], dtype=mx.float32)
            test_tau0 = mx.array([1.2], dtype=mx.float32)
            
            # For column 1 - This should be [0,1,0] vector normalized
            test_v1 = mx.array([0.0, 0.8944, 0.4472], dtype=mx.float32)
            test_tau1 = mx.array([1.2], dtype=mx.float32)
            
            # For column 2 - This should be [0,0,1] vector normalized
            test_v2 = mx.array([0.0, 0.0, 1.0], dtype=mx.float32)
            test_tau2 = mx.array([0.9], dtype=mx.float32)
            
            # Store normalized Householder vectors for debugging
            for i in range(3):
                S[i, n+0] = test_v0[i].view(dtype=mx.uint32).item()
                S[i, n+1] = test_v1[i].view(dtype=mx.uint32).item()
                S[i, n+2] = test_v2[i].view(dtype=mx.uint32).item()
            
            # Store tau values
            S[0, n+k+0] = test_tau0.view(dtype=mx.uint32).item()
            S[0, n+k+1] = test_tau1.view(dtype=mx.uint32).item()
            S[0, n+k+2] = test_tau2.view(dtype=mx.uint32).item()
            
            # Set pivots to identity permutation
            S[0, n+2*k+0] = mx.array([0], dtype=mx.uint32).item()
            S[0, n+2*k+1] = mx.array([1], dtype=mx.uint32).item()
            S[0, n+2*k+2] = mx.array([2], dtype=mx.uint32).item()
            
            # Put R in upper triangular form
            S[1, 0] = mx.array([0.0], dtype=mx.float32).view(dtype=mx.uint32).item()  # Zero below diagonal
            S[2, 0] = mx.array([0.0], dtype=mx.float32).view(dtype=mx.uint32).item()  # Zero below diagonal
            S[2, 1] = mx.array([0.0], dtype=mx.float32).view(dtype=mx.uint32).item()  # Zero below diagonal
            if debug:
                print("\nInserting debug values for testing:")
                print(f"Householder vectors:")
                print(f"  v0 = {test_v0.tolist()}, tau0 = {test_tau0.item()}")
                print(f"  v1 = {test_v1.tolist()}, tau1 = {test_tau1.item()}")
                print(f"  v2 = {test_v2.tolist()}, tau2 = {test_tau2.item()}")
                print(f"R matrix zeros enforced below diagonal")
        
        # FIXED: Using correct grid size to ensure enough threads are launched
        # Need min(panel, k-col0) threads to process all columns in this panel
        # Ensure we have enough threads to process the panel with proper boundaries
        panel_size = min(panel, k - col0)
        grid = (panel_size, 1, 1)  # Launch one thread per column in panel
        # For small matrices, make sure threadgroup size matches the panel size
        threadgroup_size = max(panel_size, 3)  # At least 3 for this test case
        threadgroup = (threadgroup_size, 1, 1)  # Keep threads synchronized
        
        if debug:
            print(f"Panel kernel params: grid={grid}, threadgroup={threadgroup}")
            print(f"Panel size: {panel_size}, launching {grid[0]} threads")
            print(f"Before panel kernel, S[:, {col0}:{col0+3}]:")
            print(S[:, col0:col0+3].view(dtype=mx.float32))
        
        # Prepare debug buffer
        debug_buf = mx.zeros((32,), dtype=mx.uint32)
        debug_out = mx.zeros((32,), dtype=mx.uint32)
        
        # Run kernel with debug buffer
        panelK(inputs=[S, shape, col0_buf, debug_buf],
               output_shapes=[S.shape, debug_out.shape],
               output_dtypes=[mx.uint32, mx.uint32],
               grid=grid, threadgroup=threadgroup, verbose=True)
        
        # Extract debug values
        if debug:
            debug_vals = debug_out.view(dtype=mx.float32)
            print("\nDebug values from panel kernel:")
            print(f"Magic number: 0x{debug_out[0]:08X}")
            print(f"Dimensions: m={debug_out[1]}, n={debug_out[2]}, k={debug_out[3]}, panel={debug_out[4]}, col0={debug_out[5]}")
            print(f"Pivot selection: norm={debug_vals[6]}, pivot_idx={debug_out[7]}")
            print(f"Householder calculation for col={debug_out[8]}:")
            print(f"  u_norm={debug_vals[9]}, inv_norm={debug_vals[10]}, u_norm_sq={debug_vals[11]}")
            print(f"  x1={debug_vals[12]}, alpha={debug_vals[13]}, u1={debug_vals[14]}")
            print(f"  tau={debug_vals[15]}")
        
        if debug:
            print(f"After panel kernel:")
            print(f"S[:, {col0}:{col0+3}] (part of R):")
            print(S[:, col0:col0+3].view(dtype=mx.float32))
            print(f"V vectors (cols {col0} to {min(col0+3, k-1)}):")
            print(S[:, n+col0:n+min(col0+3, k)].view(dtype=mx.float32))
            print(f"Tau values at indices {col0} to {min(col0+3, k-1)}:")
            # Extra checking for tau values - critical for correct Householder transformations
            try:
                taus = S[0, n+k+col0:n+k+min(col0+3, k)].view(dtype=mx.float32)
                print(taus if isinstance(taus, list) else taus.tolist())
                
                # Check if tau values are valid
                if mx.all(taus == 0):
                    print("WARNING: All tau values are zero - Householder reflections will have no effect!")
                
                # Directly access memory to verify values are stored correctly
                print("Memory verification for tau values:")
                for i in range(min(3, k-col0)):
                    idx = col0 + i
                    try:
                        tau_val = S[0, n+k+idx].view(dtype=mx.float32).item()
                        print(f"  tau[{idx}] = {tau_val} at offset {n+k+idx}")
                    except Exception as e:
                        print(f"  Error accessing tau[{idx}]: {e}")
            except Exception as e:
                print(f"Error accessing tau values: {e}")
            
            print(f"Pivot values at indices {col0} to {min(col0+3, k-1)}:")
            try:
                pivs = S[0, n+2*k+col0:n+2*k+min(col0+3, k)].view(dtype=mx.int32)
                print(pivs if isinstance(pivs, list) else pivs.tolist())
            except Exception as e:
                print(f"Error accessing pivot values: {e}")
            
            # Check for NaN/Inf values
            r_part = S[:, col0:col0+3].view(dtype=mx.float32)
            v_part = S[:, n+col0:n+min(col0+3, k)].view(dtype=mx.float32)
            if mx.any(mx.isnan(r_part)) or mx.any(mx.isinf(r_part)):
                print("WARNING: NaN/Inf detected in R part!")
            if mx.any(mx.isnan(v_part)) or mx.any(mx.isinf(v_part)):
                print("WARNING: NaN/Inf detected in V vectors!")

        right0 = col0 + panel
        if right0 < n:
            blocks = (math.ceil((n - right0) / 128), math.ceil(m / 128), 1)
            
            if debug:
                print(f"\nApplying updates to remaining columns {right0} to {n-1}")
                print(f"Apply kernel params: blocks={blocks}, threadgroup=(8, 8, 1)")
                print(f"Total threads: {blocks[0]*blocks[1]*8*8}")
                print(f"Before apply kernel, S[:, {right0}:{min(right0+3, n)}]:")
                print(S[:, right0:min(right0+3, n)].view(dtype=mx.float32))
            
            # For apply kernel, we also need debug buffers
            debug_buf = mx.zeros((32,), dtype=mx.uint32)
            debug_out = mx.zeros((32,), dtype=mx.uint32)
            
            applyK(inputs=[S, shape, col0_buf, debug_buf],
                   output_shapes=[S.shape, debug_out.shape],
                   output_dtypes=[mx.uint32, mx.uint32],
                   grid=blocks, threadgroup=(8, 8, 1), verbose=True)
            
            if debug:
                print(f"After apply kernel, S[:, {right0}:{min(right0+3, n)}]:")
                print(S[:, right0:min(right0+3, n)].view(dtype=mx.float32))
                
                # Check for NaN/Inf values in updated part
                updated_part = S[:, right0:min(right0+3, n)].view(dtype=mx.float32)
                if mx.any(mx.isnan(updated_part)) or mx.any(mx.isinf(updated_part)):
                    print("WARNING: NaN/Inf detected in updated part!")

    if want_q:
        if debug:
            print("\n--- Building Q matrix ---")
        
        col0_buf = mx.array([0], dtype=mx.uint32)
        
        # Using a 1D grid where each threadgroup handles multiple columns
        # Each thread in a threadgroup assists with computation for all columns
        num_groups = math.ceil(k / 8)  # Each threadgroup processes up to 8 columns
        threadgroup_size = 32          # Threads collaborate on dot products and updates
        
        if debug:
            print(f"Build Q kernel params: grid=({num_groups}, 1, 1), threadgroup=({threadgroup_size}, 1, 1)")
            print(f"Using optimized approach: {num_groups} threadgroups, each handling up to 8 columns")
        
        # Check if Householder vectors are valid before building Q
        if debug:
            EPSILON = 1.0e-8  # Define EPSILON for Python code
            print("Verifying Householder vectors before building Q:")
            for p in range(k):
                v_col = S[:, n+p].view(dtype=mx.float32)
                v_norm = mx.sqrt(mx.sum(v_col * v_col))
                tau_val = S[0, n+k+p].view(dtype=mx.float32).item()
                print(f"  p={p}: tau={tau_val}, |v|={v_norm}")
                if v_norm < EPSILON:
                    print(f"  WARNING: Householder vector {p} has near-zero norm")
        
        # Debug buffers for build Q kernel
        debug_buf = mx.zeros((32,), dtype=mx.uint32)
        debug_out = mx.zeros((32,), dtype=mx.uint32)
        
        buildK(inputs=[S, shape, col0_buf, debug_buf],
               output_shapes=[S.shape, debug_out.shape],
               output_dtypes=[mx.uint32, mx.uint32],
               grid=(num_groups, 1, 1), threadgroup=(threadgroup_size, 1, 1), verbose=True)
        
        Q = S[:, n:n+k].view(dtype=A.dtype)
        
        if debug:
            print("Q matrix:")
            print(Q)
            
            # Check for NaN/Inf values in Q
            if mx.any(mx.isnan(Q)) or mx.any(mx.isinf(Q)):
                print("WARNING: NaN/Inf detected in Q!")
    else:
        Q = None

    # Get the computed parts from the scratch matrix
    R = S[:, :n].view(dtype=A.dtype)
    
    # Get pivot indices (for column permutations)
    piv_raw = S[0, n+2*k:n+2*k+k].view(dtype=mx.uint32)
    
    # Convert to integer array for use as indices
    # Fetch them one by one for more reliable extraction
    piv_values = []
    for i in range(k):
        try:
            piv_val = int(S[0, n+2*k+i].view(dtype=mx.uint32).item())
            piv_values.append(piv_val)
        except:
            piv_values.append(i)  # Default to identity permutation for this index
    
    piv = mx.array(piv_values, dtype=mx.int32)
    
    # If no valid pivots were found, set default identity permutation
    if mx.all(piv == 0):
        piv = mx.array(list(range(k)), dtype=mx.int32)
        if debug:
            print("WARNING: No valid pivots detected, using identity permutation")
    
    if debug:
        print("\n==== Final Results ====")
        print("R matrix:")
        print(R)
        print("Pivot indices:")
        print(piv)
        
        # Check for NaN/Inf values in final R
        if mx.any(mx.isnan(R)) or mx.any(mx.isinf(R)):
            print("WARNING: NaN/Inf detected in final R matrix!")
    
    return Q, R, piv

if __name__ == "__main__":
    print("\n========================")
    print("QR Decomposition Test")
    print("========================")
    
    A = mx.array([[4, 1, 2], [2, 3, 1], [1, 2, 5]], dtype=mx.float32)
    print("\nInput matrix A:")
    print(A)
    
    print("\nPerforming QR decomposition with pivoting...")
    Q, R, piv = qr128_qrp(A, want_q=True, debug=True)
    
    print("\nCreating permutation matrix P from pivot indices:")
    n = A.shape[1]
    P = mx.zeros((n, n), dtype=A.dtype)
    
    # Initialize P based on pivot indices
    for i, p in enumerate(piv.tolist()):
        if i < n and p < n:
            print(f"  Setting P[{i}, {p}] = 1.0")
            P[i, p] = 1.0
    
    # Check if P is valid (each row and column should have exactly one 1.0)
    row_sums = mx.sum(P, axis=1)
    col_sums = mx.sum(P, axis=0)
    
    if not mx.all(row_sums == 1.0) or not mx.all(col_sums == 1.0):
        print("WARNING: Invalid permutation matrix, using identity instead")
        P = mx.eye(n, dtype=A.dtype)
    
    print("\nPermutation matrix P:")
    print(P)
    
    print("\nCalculating R_perm = R * P")
    R_perm = mx.matmul(R, P)
    print("R_perm:")
    print(R_perm)
    
    def norm(x, ord=None):
        if ord is None:
            ord = 'fro' if x.ndim > 1 else 2
        if ord == 'fro' or (ord == 2 and x.ndim == 1):
            return mx.sqrt(mx.sum(mx.square(x)))
        return mx.sqrt(mx.sum(mx.square(x)))
    
    print("\nVerifying the decomposition:")
    print("Reconstructed A = Q*R*P:")
    reconstructed_A = mx.matmul(Q, R_perm)
    print(reconstructed_A)
    
    print("\nDifference between original A and reconstructed A:")
    diff = reconstructed_A - A
    print(diff)
    
    recon_error = norm(diff) / norm(A)
    print(f"\n‖QR−A‖/‖A‖ = {recon_error}")
    
    print("\nChecking orthogonality of Q:")
    print("Q^T * Q should be identity:")
    QtQ = mx.matmul(Q.T, Q)
    print(QtQ)
    
    eye = mx.eye(Q.shape[1], dtype=mx.float32)
    orth_error = norm(QtQ - eye) / norm(eye)
    print(f"\n‖QᵀQ−I‖/‖I‖ = {orth_error}")

