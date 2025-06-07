import mlx.core as mx
import numpy as np
import math
from typing import Optional, Tuple

def _compile(src, name):
    return mx.fast.metal_kernel(
        name=name,
        source=src,
        input_names=["A", "shape", "col0", "debug_buf"],
        output_names=["scratch", "debug_out", "thread_log"],
        ensure_row_contiguous=True
    )

# Metal kernel with detailed logging
_PANEL_SRC = """
#define TG_SIZE_X  8u
#define TG_SIZE_Y  8u
#define EPSILON    1.0e-8f
#define LOG_SIZE   100u  // Size of thread execution log

    // Get thread position in all dimensions
    const uint tid_x = thread_position_in_grid.x;
    const uint tid_y = thread_position_in_grid.y;
    const uint tid_z = thread_position_in_grid.z;
    const uint ltid_x = thread_position_in_threadgroup.x;
    const uint ltid_y = thread_position_in_threadgroup.y;
    const uint ltid = ltid_y * TG_SIZE_X + ltid_x;
    
    // Extract dimensions from shape array
    const uint m = shape[0];        // Number of rows
    const uint n = shape[1];        // Number of columns
    const uint k = shape[2];        // min(m,n)
    const uint panel = shape[3];    // Panel width
    const uint col0 = *col0_buf;    // Starting column for this panel
    const uint scratch_cols = n + 3 * k;
    
    // Calculate current column index
    const uint col = col0 + tid_x;
    
    // THREAD LOGGING SYSTEM
    // Each thread will record its execution in the thread_log buffer
    // Format: [tid_x, tid_y, stage, value]
    // Where stage is:
    //   1 = thread started
    //   2 = after column copying
    //   3 = after computing norm
    //   4 = after reduction
    //   5 = after vector normalization
    
    // Find a slot in the log array for this thread
    uint log_idx = (tid_x * gridDim.y + tid_y) * 4;
    
    // Stage 1: Thread started
    if (log_idx < LOG_SIZE) {
        thread_log[log_idx] = tid_x;
        thread_log[log_idx+1] = tid_y;
        thread_log[log_idx+2] = 1;
        thread_log[log_idx+3] = as_type<uint>(1.0f);
    }
    
    // Thread 0,0 writes debug header with magic number and dimensions
    if (tid_x == 0 && tid_y == 0) {
        debug_out[0] = 0xF00D0001;  // Magic number
        debug_out[1] = m;           // m rows
        debug_out[2] = n;           // n columns
        debug_out[3] = k;           // min(m,n)
        debug_out[4] = panel;       // panel size
        debug_out[5] = col0;        // starting column
        
        // Log extra thread info
        debug_out[20] = gridDim.x;  // Grid dimensions
        debug_out[21] = gridDim.y;
        debug_out[22] = threadgroup_position_in_grid.x; // Thread group position
        debug_out[23] = threadgroup_position_in_grid.y;
    }
    
    // Guard clause: don't process out-of-bounds columns
    if (tid_x >= panel || col >= k) {
        if (log_idx < LOG_SIZE) {
            thread_log[log_idx+2] = 999; // Mark as skipped
        }
        return;
    }
    
    // Memory layout: [R | V | tau | piv]
    // R part: scratch[0:m, 0:n]
    // V part: scratch[0:m, n:n+k]
    // tau: scratch[0, n+k:n+2k]
    // pivot: scratch[0, n+2k:n+3k]
    
    // Shared memory for threadgroup
    threadgroup float norm_sq_partial[TG_SIZE_X * TG_SIZE_Y];
    threadgroup float tg_norm;
    threadgroup uint tg_pivot;
    
    // 1. Copy column from A to V (householder vectors storage)
    //    Each thread handles a different row of its assigned column
    for (uint row = tid_y; row < m; row += gridDim.y) {
        // Calculate indices for A and scratch
        uint a_idx = row * n + col;                // Source in A
        uint r_idx = row * scratch_cols + col;     // Target in R part
        uint v_idx = row * scratch_cols + n + col; // Target in V part
        
        if (a_idx < m * n && r_idx < m * scratch_cols && v_idx < m * scratch_cols) {
            // Copy from input A to both R and V parts of scratch
            float val = as_type<float>(A[a_idx]);
            scratch[r_idx] = as_type<uint>(val);
            scratch[v_idx] = as_type<uint>(val);
            
            // Log this copy operation for one value (first element only)
            if (row == 0 && log_idx < LOG_SIZE) {
                thread_log[log_idx+2] = 2;
                thread_log[log_idx+3] = as_type<uint>(val);
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Compute column norm - each thread works on a subset of rows
    float thread_norm_sq = 0.0f;
    for (uint row = tid_y; row < m; row += gridDim.y) {
        uint v_idx = row * scratch_cols + n + col; // Index in V part
        if (v_idx < m * scratch_cols) {
            float val = as_type<float>(scratch[v_idx]);
            thread_norm_sq = fma(val, val, thread_norm_sq);
        }
    }
    
    // Store partial norm in shared memory
    norm_sq_partial[ltid] = thread_norm_sq;
    
    // Log computed norm contribution
    if (log_idx < LOG_SIZE) {
        thread_log[log_idx+2] = 3;
        thread_log[log_idx+3] = as_type<uint>(thread_norm_sq);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Reduce partial norms within threadgroup
    if (ltid_y == 0) {
        float sum = norm_sq_partial[ltid_x]; // Start with this thread's value
        for (uint i = 1; i < TG_SIZE_Y; i++) {
            uint idx = ltid_x + i * TG_SIZE_X;
            if (idx < TG_SIZE_X * TG_SIZE_Y) {
                sum += norm_sq_partial[idx];
            }
        }
        norm_sq_partial[ltid_x] = sum;
        
        // Log the sum for the first thread
        if (ltid_x == 0 && log_idx < LOG_SIZE) {
            thread_log[log_idx+2] = 4;
            thread_log[log_idx+3] = as_type<uint>(sum);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 4. Find maximum norm and set pivot - done by thread (0,0) for simplicity
    if (tid_x == 0 && tid_y == 0) {
        float max_norm_sq = 0.0f;
        uint pivot_idx = 0;
        
        // Find maximum norm across all columns in this panel
        for (uint i = 0; i < min(panel, TG_SIZE_X); i++) {
            float col_norm = norm_sq_partial[i];
            if (col_norm > max_norm_sq) {
                max_norm_sq = col_norm;
                pivot_idx = col0 + i;
            }
        }
        
        // Store in threadgroup memory for other threads
        tg_norm = sqrt(max(max_norm_sq, EPSILON));
        tg_pivot = pivot_idx;
        
        // Store in debug buffer
        debug_out[6] = as_type<uint>(tg_norm);      // Norm value
        debug_out[7] = tg_pivot;                    // Pivot index
        debug_out[24] = as_type<uint>(max_norm_sq); // Squared norm
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 5. Compute Householder reflector for pivot column
    //    Only the thread handling the pivot column does this part
    if (col == tg_pivot && tid_y == 0) {
        // Store column index in debug buffer
        debug_out[8] = col;
        // Get the norm of the column
        float norm = tg_norm;
        
        // First element of the column (at diagonal)
        float x1 = as_type<float>(scratch[col * scratch_cols + n + col]);
        
        // Compute Householder scalar (alpha)
        float alpha = -sign(x1) * norm;
        
        // First element of unnormalized Householder vector
        float u1 = x1 - alpha;
        
        // Compute norm of u vector
        float u_norm_sq = u1 * u1;
        for (uint r = col + 1; r < m; r++) {
            float v = as_type<float>(scratch[r * scratch_cols + n + col]);
            u_norm_sq = fma(v, v, u_norm_sq);
        }
        
        float u_norm = sqrt(max(u_norm_sq, EPSILON));
        float inv_norm = (u_norm > EPSILON) ? 1.0f / u_norm : 0.0f;
        
        // Store intermediate values in debug buffer
        debug_out[9] = as_type<uint>(u_norm);      // Vector norm
        debug_out[10] = as_type<uint>(inv_norm);   // Inverse norm
        debug_out[11] = as_type<uint>(u_norm_sq);  // Vector norm squared
        debug_out[12] = as_type<uint>(x1);         // First element of column
        debug_out[13] = as_type<uint>(alpha);      // Alpha value
        debug_out[14] = as_type<uint>(u1);         // First element of unnormalized vector
        
        // Tau value for Householder reflection
        float tau = (u_norm > EPSILON) ? 2.0f / (1.0f + (u1 * u1) / u_norm_sq) : 0.0f;
        debug_out[15] = as_type<uint>(tau);
        
        // Store tau value
        if (n + k + col < m * scratch_cols) {
            scratch[n + k + col] = as_type<uint>(tau);
        }
        
        // Store pivot index
        if (n + 2*k + col < m * scratch_cols) {
            scratch[n + 2*k + col] = tg_pivot;
        }
        
        // Zero out elements below diagonal in R
        for (uint r = col + 1; r < m; r++) {
            scratch[r * scratch_cols + col] = as_type<uint>(0.0f);
        }
        
        // Normalize and store Householder vector
        uint idx_diag = col * scratch_cols + n + col;
        if (idx_diag < m * scratch_cols) {
            scratch[idx_diag] = as_type<uint>(u1 * inv_norm);
        }
        
        for (uint r = col + 1; r < m; r++) {
            uint idx_v = r * scratch_cols + n + col;
            if (idx_v < m * scratch_cols) {
                float v = as_type<float>(scratch[r * scratch_cols + n + col]);
                float normalized_v = v * inv_norm;
                scratch[idx_v] = as_type<uint>(normalized_v);
            }
        }
        
        // Log completion of Householder computation
        if (log_idx < LOG_SIZE) {
            thread_log[log_idx+2] = 5;
            thread_log[log_idx+3] = as_type<uint>(tau);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# Compile panel kernel only
panelK = _compile(_PANEL_SRC, "panel_factor_qrp128_debug")

def debug_panel_kernel(A: mx.array) -> dict:
    """Run just the panel kernel with detailed logging and return diagnostic info."""
    assert A.ndim == 2, "Input must be 2D matrix"
    assert A.dtype == mx.float32, "Only float32 supported"
    
    m, n = map(int, A.shape)
    k = min(m, n)
    panel = min(32, k)
    scratch_cols = n + 3 * k  # Structure: [R | V | tau | piv]
    
    print(f"\n==== Panel Kernel Debug ====")
    print(f"Input matrix A ({m}x{n}):")
    print(A)
    print(f"k = min(m, n) = {k}, panel size = {panel}")
    print(f"scratch_cols = {scratch_cols} (n={n} + 3*k={3*k})")
    
    # Create scratch matrix and initialize it
    S = mx.zeros((m, scratch_cols), dtype=mx.uint32)
    
    # Set up shape array
    shape = mx.array([m, n, k, panel, 5], dtype=mx.uint32)
    col0_buf = mx.array([0], dtype=mx.uint32)
    
    # Create debug buffer
    debug_buf = mx.zeros((32,), dtype=mx.uint32)
    debug_out = mx.zeros((32,), dtype=mx.uint32)
    
    # Thread execution log - stores [tid_x, tid_y, stage, value] for each thread
    thread_log_size = 100  # Up to 25 threads (4 values per thread)
    thread_log = mx.zeros((thread_log_size,), dtype=mx.uint32)
    
    # Configure kernel launch parameters for 2D grid
    grid = (panel, m, 1)       # One thread per column and row
    threadgroup = (8, 8, 1)    # 8x8 threadgroup
    
    print(f"Panel kernel params: grid={grid}, threadgroup={threadgroup}")
    
    # Run the kernel
    panelK(inputs=[A, shape, col0_buf, debug_buf],
           output_shapes=[S.shape, debug_out.shape, thread_log.shape],
           output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
           grid=grid, threadgroup=threadgroup, verbose=True)
    
    # Extract debug values
    debug_vals = debug_out.view(dtype=mx.float32)
    
    print("\nDebug values from panel kernel:")
    print(f"Magic number: 0x{debug_out[0]:08X}")
    print(f"Dimensions: m={debug_out[1]}, n={debug_out[2]}, k={debug_out[3]}, panel={debug_out[4]}, col0={debug_out[5]}")
    print(f"Grid dimensions: {debug_out[20]}x{debug_out[21]}")
    print(f"Thread group position: ({debug_out[22]}, {debug_out[23]})")
    print(f"Pivot selection: norm={debug_vals[6]}, norm_sq={debug_vals[24]}, pivot_idx={debug_out[7]}")
    print(f"Householder calculation for col={debug_out[8]}:")
    print(f"  u_norm={debug_vals[9]}, inv_norm={debug_vals[10]}, u_norm_sq={debug_vals[11]}")
    print(f"  x1={debug_vals[12]}, alpha={debug_vals[13]}, u1={debug_vals[14]}")
    print(f"  tau={debug_vals[15]}")
    
    # Extract thread execution log
    print("\nThread Execution Log:")
    print("tid_x  tid_y  stage  value")
    print("-----  -----  -----  -----")
    
    log_entries = []
    for i in range(0, thread_log_size, 4):
        if i+3 < thread_log_size and (thread_log[i] > 0 or thread_log[i+1] > 0 or thread_log[i+2] > 0):
            tid_x = thread_log[i]
            tid_y = thread_log[i+1]
            stage = thread_log[i+2]
            value = thread_log[i+3].view(dtype=mx.float32).item()
            log_entries.append((tid_x, tid_y, stage, value))
    
    # Sort by thread ID and then by stage
    log_entries.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Print the log entries
    for entry in log_entries:
        tid_x, tid_y, stage, value = entry
        stage_desc = {
            1: "started",
            2: "copy",
            3: "norm",
            4: "reduce",
            5: "householder",
            999: "skipped"
        }.get(stage, str(stage))
        print(f"{tid_x:5d}  {tid_y:5d}  {stage_desc:5s}  {value:.6f}")
    
    # Extract results from scratch matrix
    R = S[:, :n].view(dtype=A.dtype)
    V = S[:, n:n+k].view(dtype=A.dtype)
    tau_vals = S[0, n+k:n+k+k].view(dtype=A.dtype)
    pivot_vals = S[0, n+2*k:n+2*k+k].view(dtype=mx.int32)
    
    print("\nResults from panel kernel:")
    print("R matrix (should have zeros below diagonal):")
    print(R)
    print("\nV vectors (Householder vectors):")
    print(V)
    print("\nTau values:")
    print(tau_vals)
    print("\nPivot indices:")
    print(pivot_vals)
    
    # Check if results make sense
    if mx.all(V == 0):
        print("\nWARNING: All Householder vectors are zero!")
    
    if mx.all(tau_vals == 0):
        print("WARNING: All tau values are zero - no Householder reflections!")
    
    if mx.all(pivot_vals == 0):
        print("WARNING: No pivoting performed!")
    
    if not mx.all(mx.tril(R, -1) == 0):
        print("WARNING: R matrix should have zeros below diagonal!")
    
    return {
        "debug_out": debug_out,
        "thread_log": log_entries,
        "R": R,
        "V": V,
        "tau": tau_vals,
        "pivots": pivot_vals
    }

if __name__ == "__main__":
    print("\n========================")
    print("QR Panel Kernel Debug")
    print("========================")
    
    A = mx.array([[4, 1, 2], [2, 3, 1], [1, 2, 5]], dtype=mx.float32)
    
    # Run diagnostic
    diagnostics = debug_panel_kernel(A)