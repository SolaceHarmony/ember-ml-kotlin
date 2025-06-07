from doctest import debug
import mlx.core as mx
import time

# This script contains an experimental tiled QR implementation
# that incorporates limb-based precision (HPC) within a Metal kernel.

def tiled_hpc_qr(A: mx.array) -> tuple[mx.array, mx.array]:
    """
    Experimental tiled QR decomposition using a Metal kernel with limb-based precision.
    
    This implementation uses tiling for better cache utilization and parallelism,
    along with limb-based precision arithmetic for improved numerical stability.
    
    Args:
        a: Input matrix (M x N) as float32.
        
    Returns:
        Tuple of (Q, R) matrices (M x M and M x N) as float32.
    """
    m, n = A.shape
    min_dim = min(m, n)

    # Define Metal kernel source string for tiled, limb-based QR decomposition
    metal_kernel_source = """
/*****************************************************************************
 *  tiled_hpc_qr_kernel – 128-bit limb (hpc16×8) QR for MLX / Metal GPU
 *  outputs:  Q_out  (m×m),  R_out (m×n),  debug[0…11]
 *            debug[0]..[3]  = first-fault flags
 *            debug[4]       = last ‖v‖²   ,  debug[5] = last ‖v‖
 *            debug[6]       = last vᵀv    ,  debug[7] = last 1/(vᵀv)
 *            debug[8]       = last dot_R  ,  debug[9] = last dot_Q
 *            debug[10]      = column scale, debug[11] = threadgroup size
 *****************************************************************************/
#define EPSILON      1e-10f
#define NUM_LIMBS    8
#define BIT_MASK     0xFFFFu
#define LIMB_RADIX   65536.0f          /* 2¹⁶                                     */

//device float *debug          [[buffer(4)]];
threadgroup float  tg_inv_vt_v;
threadgroup float  tg_scale;

/* ------------------------------------------------------------------------- */
{
    const uint m        = shapeParams[0];
    const uint n        = shapeParams[1];
    const uint min_dim  = (m < n ? m : n);

    /* ============= 0. copy A→R, eye→Q ==================================== */
    for (uint row = thread_position_in_grid.x;
         row < m;
         row += threads_per_threadgroup.x)
    {
        for (uint col = 0; col < n; ++col)
            R_out[row*n + col] = A[row*n + col];

        for (uint col = 0; col < m; ++col)
            Q_out[row*m + col] = (row == col) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    debug[11] = (float)threads_per_threadgroup.x;

    /* ============= 1. panel loop ======================================== */
    for (uint k = 0; k < min_dim; ++k)
    {
    /* ---- 1a. column scaling ------------------------------------------- */
        if (thread_position_in_threadgroup.x == 0)
        {
            float maxAbs = 0.0f;
            for (uint i = k; i < m; ++i)
                maxAbs = fmax(maxAbs, fabs(R_out[i*n + k]));

            float scale = (maxAbs > EPSILON) ? 1.0f / maxAbs : 1.0f;
            scale = clamp(scale, 1e-4f, 1e4f);
            tg_scale = scale;
            for (uint i = k; i < m; ++i)
                R_out[i*n + k] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float scale = tg_scale;
        debug[10] = scale;

    /* ---- 1b. ||v||² with 8 limbs -------------------------------------- */
        ushort     local[NUM_LIMBS] = {0};
        threadgroup atomic_uint shared[NUM_LIMBS];
        if (thread_position_in_threadgroup.x == 0)
            for (uint l=0;l<NUM_LIMBS;++l)
                atomic_store_explicit(&shared[l],0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = k + thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup.x)
        {
            float v  = R_out[i*n + k];
            uint  b  = as_type<uint>(v);
            ushort lo = b & BIT_MASK;
            ushort hi = (b >> 16) & BIT_MASK;

            uint p0 = (uint)(lo*lo);
            uint p1 = (uint)(hi*hi);
            uint pc = ((uint)(lo*hi))<<1;

            local[0]+= (ushort)(p0 & BIT_MASK);
            local[1]+= (ushort)((p0>>16) + (pc & BIT_MASK));
            local[2]+= (ushort)((pc>>16) + (p1 & BIT_MASK));
            local[3]+= (ushort)(p1>>16);
            /* limbs 4-7 remain 0 for float32², but keep for headroom      */
        }
        for (uint l=0;l<NUM_LIMBS;++l)
            atomic_fetch_add_explicit(&shared[l],(uint)local[l],memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_position_in_threadgroup.x == 0)
            for (uint l=0;l<NUM_LIMBS-1;++l){
                uint v  = atomic_load_explicit(&shared[l],memory_order_relaxed);
                uint c  = v>>16;
                atomic_store_explicit(&shared[l],v & BIT_MASK,memory_order_relaxed);
                atomic_fetch_add_explicit(&shared[l+1],c,memory_order_relaxed);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm2 = 0.0f, radix = 1.0f;
        for (uint l=0;l<NUM_LIMBS;++l){
            uint v  = atomic_load_explicit(&shared[l],memory_order_relaxed);
            norm2  += (float)v * radix;
            radix  *= LIMB_RADIX;
        }
        if (isnan(norm2) || norm2<=EPSILON){ debug[0]=1.0f; norm2 = EPSILON; }
        debug[4] = norm2;
        float norm = sqrt(norm2);
        debug[5] = norm;

    /* ---- 2. Householder head update ----------------------------------- */
        if (thread_position_in_threadgroup.x == 0)
        {
            float sign = (R_out[k*n + k] >= 0.0f ? 1.0f : -1.0f);
            R_out[k*n + k] += sign * norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);


    /* ---- 4. vᵀv inverse (8-limb) -------------------------------------- */
        ushort     vloc[NUM_LIMBS] = {0};
        threadgroup atomic_uint vshr[NUM_LIMBS];
        if (thread_position_in_threadgroup.x == 0)
            for (uint l=0;l<NUM_LIMBS;++l)
                atomic_store_explicit(&vshr[l],0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = k + thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup.x)
        {
            float v  = R_out[i*n + k];
            uint  b  = as_type<uint>(v);
            ushort lo = b & BIT_MASK;
            ushort hi = (b >> 16) & BIT_MASK;

            uint p0 = (uint)(lo*lo);
            uint p1 = (uint)(hi*hi);
            uint pc = ((uint)(lo*hi))<<1;

            vloc[0]+= (ushort)(p0 & BIT_MASK);
            vloc[1]+= (ushort)((p0>>16) + (pc & BIT_MASK));
            vloc[2]+= (ushort)((pc>>16) + (p1 & BIT_MASK));
            vloc[3]+= (ushort)(p1>>16);
        }
        for (uint l=0;l<NUM_LIMBS;++l)
            atomic_fetch_add_explicit(&vshr[l],(uint)vloc[l],memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_position_in_threadgroup.x == 0)
            for (uint l=0;l<NUM_LIMBS-1;++l){
                uint v  = atomic_load_explicit(&vshr[l],memory_order_relaxed);
                uint c  = v>>16;
                atomic_store_explicit(&vshr[l],v & BIT_MASK,memory_order_relaxed);
                atomic_fetch_add_explicit(&vshr[l+1],c,memory_order_relaxed);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float vtv = 0.0f; radix = 1.0f;
        for (uint l=0;l<NUM_LIMBS;++l){
            uint v=atomic_load_explicit(&vshr[l],memory_order_relaxed);
            vtv += (float)v * radix;
            radix *= LIMB_RADIX;
        }
        if (isnan(vtv)||vtv<=EPSILON){ debug[0]=2.0f; vtv=EPSILON; }
        debug[6]=vtv;
        if (thread_position_in_threadgroup.x==0)
        {
            float inv = 1.0f / vtv;
            if (isnan(inv)||isinf(inv)) inv = 0.0f;
            tg_inv_vt_v = inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_vtv = tg_inv_vt_v;
        debug[7]=inv_vtv;

    /* ---- 5. reflection on R ------------------------------------------- */
        for (uint j = k; j < n; ++j)
        {
            ushort dloc[NUM_LIMBS]={0};
            threadgroup atomic_uint dshr[NUM_LIMBS];
            if (thread_position_in_threadgroup.x==0)
                for(uint l=0;l<NUM_LIMBS;++l)
                    atomic_store_explicit(&dshr[l],0u,memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for(uint i=k + thread_position_in_grid.x;
                i<m;
                i+=threads_per_threadgroup.x)
            {
                float vi=R_out[i*n+k];
                float vj=R_out[i*n+j];
                uint  a=as_type<uint>(vi);
                uint  b=as_type<uint>(vj);
                ushort vi_lo=a & BIT_MASK, vi_hi=(a>>16)&BIT_MASK;
                ushort vj_lo=b & BIT_MASK, vj_hi=(b>>16)&BIT_MASK;

                uint p0=(uint)(vi_lo*vj_lo);
                uint p1=(uint)(vi_hi*vj_hi);
                uint pc=(uint)(vi_lo*vj_hi+vi_hi*vj_lo);

                dloc[0]+= (ushort)(p0 & BIT_MASK);
                dloc[1]+= (ushort)((p0>>16)+(pc & BIT_MASK));
                dloc[2]+= (ushort)((pc>>16)+(p1 & BIT_MASK));
                dloc[3]+= (ushort)(p1>>16);
            }
            for(uint l=0;l<NUM_LIMBS;++l)
                atomic_fetch_add_explicit(&dshr[l],(uint)dloc[l],memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if(thread_position_in_threadgroup.x==0)
                for(uint l=0;l<NUM_LIMBS-1;++l){
                    uint v=atomic_load_explicit(&dshr[l],memory_order_relaxed);
                    uint c=v>>16;
                    atomic_store_explicit(&dshr[l],v & BIT_MASK,memory_order_relaxed);
                    atomic_fetch_add_explicit(&dshr[l+1],c,memory_order_relaxed);
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot=0.0f; radix=1.0f;
            for(uint l=0;l<NUM_LIMBS;++l){
                uint v=atomic_load_explicit(&dshr[l],memory_order_relaxed);
                dot+= (float)v*radix;
                radix*=LIMB_RADIX;
            }
            dot /= scale;                       /* undo prior column scaling */
            dot = clamp(dot,-1e5f,1e5f);
            if(isnan(dot)||isinf(dot)){ debug[0]=3.0f; dot=0.0f; }
            debug[8]=dot;

            for(uint i=k + thread_position_in_grid.x;
                i<m;
                i+=threads_per_threadgroup.x)
            {
                float v  = R_out[i*n+k];
                float upd= 2.0f*v*dot*inv_vtv;
                if(isnan(upd)||isinf(upd)){ debug[1]=4.0f; continue; }
                R_out[i*n+j]-=upd;
                if(isnan(R_out[i*n+j])){ debug[2]=1.0f; R_out[i*n+j]=0.0f; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    /* ---- 6. reflection on Q ------------------------------------------- */
        for (uint j = 0; j < m; ++j)
        {
            ushort dloc[NUM_LIMBS]={0};
            threadgroup atomic_uint dshr[NUM_LIMBS];
            if (thread_position_in_threadgroup.x==0)
                for(uint l=0;l<NUM_LIMBS;++l)
                    atomic_store_explicit(&dshr[l],0u,memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for(uint i=k + thread_position_in_grid.x;
                i<m;
                i+=threads_per_threadgroup.x)
            {
                float vi=R_out[i*n+k];
                float qi=Q_out[i*m+j];
                uint  a=as_type<uint>(vi);
                uint  b=as_type<uint>(qi);
                ushort vi_lo=a & BIT_MASK, vi_hi=(a>>16)&BIT_MASK;
                ushort qi_lo=b & BIT_MASK, qi_hi=(b>>16)&BIT_MASK;

                uint p0=(uint)(vi_lo*qi_lo);
                uint p1=(uint)(vi_hi*qi_hi);
                uint pc=(uint)(vi_lo*qi_hi+vi_hi*qi_lo);

                dloc[0]+= (ushort)(p0 & BIT_MASK);
                dloc[1]+= (ushort)((p0>>16)+(pc & BIT_MASK));
                dloc[2]+= (ushort)((pc>>16)+(p1 & BIT_MASK));
                dloc[3]+= (ushort)(p1>>16);
            }
            for(uint l=0;l<NUM_LIMBS;++l)
                atomic_fetch_add_explicit(&dshr[l],(uint)dloc[l],memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if(thread_position_in_threadgroup.x==0)
                for(uint l=0;l<NUM_LIMBS-1;++l){
                    uint v=atomic_load_explicit(&dshr[l],memory_order_relaxed);
                    uint c=v>>16;
                    atomic_store_explicit(&dshr[l],v & BIT_MASK,memory_order_relaxed);
                    atomic_fetch_add_explicit(&dshr[l+1],c,memory_order_relaxed);
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot=0.0f; radix=1.0f;
            for(uint l=0;l<NUM_LIMBS;++l){
                uint v=atomic_load_explicit(&dshr[l],memory_order_relaxed);
                dot+= (float)v*radix;
                radix*=LIMB_RADIX;
            }
            dot /= scale;                       /* undo prior column scaling */
            dot = clamp(dot,-1e5f,1e5f);
            if(isnan(dot)||isinf(dot)){ debug[3]=1.0f; dot=0.0f; }
            debug[9]=dot;

            for(uint i=k + thread_position_in_grid.x;
                i<m;
                i+=threads_per_threadgroup.x)
            {
                float v   = R_out[i*n+k];
                float upd = 2.0f*v*dot*inv_vtv;
                if(isnan(upd)||isinf(upd)){ debug[1]=5.0f; continue; }
                Q_out[i*m+j]-=upd;
                if(isnan(Q_out[i*m+j])){ debug[3]=2.0f; Q_out[i*m+j]=0.0f; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        /* ---- 7. restore original scaling for column k -------------------- */
        for (uint i = k + thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup.x)
            R_out[i*n + k] /= scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } /* ── next column k ── */
} /* end kernel body */
        """

    try:
        # Compile the Metal kernel
        compiled_kernel = mx.fast.metal_kernel(
            name="tiled_hpc_qr_kernel",
            source=metal_kernel_source,
            input_names=["A","shapeParams"],
            output_names=["Q_out","R_out","debug"],
            ensure_row_contiguous=True
        )
    except Exception as e:
        print(f"Failed to compile Metal kernel: {e}")
        print("Please check the Metal kernel source for errors.")
        return mx.eye(m), mx.array(A)

    # Determine grid and threadgroup sizes based on matrix dimensions
    threads_per_group = min(32, max(1, m // 8))      # small TGs work best
    num_groups        = (m + threads_per_group - 1) // threads_per_group
    grid_size         = (num_groups, 1, 1)
    tg_size           = (threads_per_group, 1, 1)
    shape_params      = mx.array([m, n], dtype=mx.uint32)
    
    dbg = mx.zeros((4,), dtype=mx.uint32)

    try:
        # Execute the kernel
        Q, R, dbg = compiled_kernel(
            inputs         =[A, shape_params],
            output_shapes  =[(m,m),(m,n),(12,)],
            output_dtypes  =[A.dtype,A.dtype,mx.float32],
            grid           =grid_size,
            threadgroup    =tg_size
        )
        print("Debug flags:", dbg)
        return Q,R
    except Exception as e:
        print(f"Metal kernel execution failed: {e}")
        return mx.eye(m), mx.array(A)

if __name__ == "__main__":
    # Test with a smaller matrix first
    print("Testing with small matrix (10x10)...")
    A_small = mx.random.normal((10, 10), dtype=mx.float32)
    start_time = time.time()
    Q_small, R_small = tiled_hpc_qr(A_small)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_small.T, Q_small) - mx.eye(Q_small.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_small, R_small) - A_small)).item())
    print("-" * 20)
    alpha = mx.mean(mx.abs(Q_enhanced.T @ A_medium)) / mx.mean(mx.abs(R_medium))
    print('empirical α:', alpha.item())
    print('recon after rescale:',
        mx.mean(mx.abs(Q_enhanced @ (R_medium/alpha) - A_medium)).item())
    # Test with original matrix
    print("Testing with original matrix (100x150)...")
    A_example = mx.random.normal((100, 150), dtype=mx.float32)
    start_time = time.time()
    Q_example, R_example = tiled_hpc_qr(A_example)
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.4f} seconds.")
    print("Orthogonality check (Q.T @ Q - I):", mx.mean(mx.abs(mx.matmul(Q_example.T, Q_example) - mx.eye(Q_example.shape[0]))).item())
    print("Reconstruction check (Q @ R - A):", mx.mean(mx.abs(mx.matmul(Q_example, R_example) - A_example)).item())
    print("-" * 20)

    # Compare with native MLX QR
    print("Comparing with native MLX QR...")
    start_time = time.time()
    Q_native, R_native = mx.linalg.qr(A_example, stream=mx.cpu)
    end_time = time.time()
    print(f"Native MLX QR completed in {end_time - start_time:.4f} seconds.")
    print("Difference in Q:", mx.mean(mx.abs(Q_example - Q_native)).item())
    print("Difference in R:", mx.mean(mx.abs(R_example - R_native)).item())
    print("-" * 20)