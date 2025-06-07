const uint m = shapeParams[0];
    const uint n = shapeParams[1];
    const uint tile_size = shapeParams[2]; // Use tile size from shapeParams
    const uint min_dim = (m < n ? m : n);
    
    /* Inline helper functions for limb-based arithmetic */
    auto add_to_limbs = [](threadgroup atomic_uint* shared, thread ushort* local) {
        for (uint l = 0; l < NUM_LIMBS; ++l)
            atomic_fetch_add_explicit(&shared[l], (uint)local[l], memory_order_relaxed);
    };
    
    auto propagate_carries = [](threadgroup atomic_uint* shared) {
        for (uint l = 0; l < NUM_LIMBS-1; ++l) {
            uint v = atomic_load_explicit(&shared[l], memory_order_relaxed);
            uint c = v >> 16;
            atomic_store_explicit(&shared[l], v & BIT_MASK, memory_order_relaxed);
            atomic_fetch_add_explicit(&shared[l+1], c, memory_order_relaxed);
        }
    };
    
    auto compute_limb_value = [](threadgroup atomic_uint* shared) -> float {
        float result = 0.0f;
        float radix = 1.0f;
        
        for (uint l = 0; l < NUM_LIMBS; ++l) {
            uint v = atomic_load_explicit(&shared[l], memory_order_relaxed);
            result += (float)v * radix;
            radix *= LIMB_RADIX;
        }
        
        return result;
    };
    
    /* Enhanced 16-bit multiplication with better precision */
    auto multiply_and_accumulate = [](float a, float b, thread ushort* result) {
        uint  a_bits = as_type<uint>(a);
        uint  b_bits = as_type<uint>(b);
        
        ushort a_lo = a_bits & BIT_MASK;
        ushort a_hi = (a_bits >> 16) & BIT_MASK;
        ushort b_lo = b_bits & BIT_MASK;
        ushort b_hi = (b_bits >> 16) & BIT_MASK;
        
        // Full 32-bit products
        uint p_lo_lo = (uint)(a_lo * b_lo);
        uint p_hi_hi = (uint)(a_hi * b_hi);
        uint p_lo_hi = (uint)(a_lo * b_hi);
        uint p_hi_lo = (uint)(a_hi * b_lo);
        uint p_mid = p_lo_hi + p_hi_lo;
        
        // Handle potential carry from middle terms
        uint carry = (p_mid < p_lo_hi) ? (1u << 16) : 0;
        
        // Accumulate into limbs with carries
        result[0] += (ushort)(p_lo_lo & BIT_MASK);
        uint temp = (p_lo_lo >> 16) + (p_mid & BIT_MASK);
        result[1] += (ushort)(temp & BIT_MASK);
        temp = (temp >> 16) + (p_mid >> 16) + (p_hi_hi & BIT_MASK) + carry;
        result[2] += (ushort)(temp & BIT_MASK);
        result[3] += (ushort)((temp >> 16) + (p_hi_hi >> 16));
    /* keep the carry-chain alive */
    result[4] += (ushort)(p_hi_hi & BIT_MASK);
    result[5] += (ushort)(p_hi_hi >> 16);
        
        // Note: For full 128-bit precision, we would extend to all 8 limbs
        // but for float32 x float32, the upper limbs remain 0
    };
    
    /* Initialize debug array */
    if (thread_position_in_grid.x == 0) {
        for (uint i = 0; i < 16; ++i)
            debug[i] = 0.0f;
        debug[11] = (float)threads_per_threadgroup;
        debug[12] = (float)tile_size;
    }
    
    /* ============= 0. copy A→R, eye→Q ==================================== */
    for (uint row = thread_position_in_grid.x;
         row < m;
         row += threads_per_threadgroup)
    {
        for (uint col = 0; col < n; ++col)
            R_out[row*n + col] = A[row*n + col];
        
        for (uint col = 0; col < m; ++col)
            Q_out[row*m + col] = (row == col) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    /* ============= 1. panel loop ======================================== */
    for (uint k = 0; k < min_dim; ++k)
    {
        /* ---- 1a. Dynamic column scaling for better numerical stability --- */
        if (thread_position_in_threadgroup.x == 0)
        {
            float maxAbs = 0.0f;
            float sumSq = 0.0f;
            
            for (uint i = k; i < m; ++i) {
                float val = R_out[i*n + k];
                maxAbs = fmax(maxAbs, fabs(val));
                sumSq += val * val;
            }
            
            // Use a combination of max and RMS for better scaling
            float rms = sqrt(sumSq / (m - k));
            float scale_factor = fmax(maxAbs, rms);
            
            float scale = (scale_factor > EPSILON) ? 1.0f / scale_factor : 1.0f;
            scale = clamp(scale, 1e-6f, 1e6f);
            tg_scale = scale;
            
            for (uint i = k; i < m; ++i)
                R_out[i*n + k] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float scale = tg_scale;
        debug[10] = scale;
        
        /* ---- 1b. ||v||² with optimized 8 limbs -------------------------- */
        thread ushort local[NUM_LIMBS] = {0};
        threadgroup atomic_uint shared[NUM_LIMBS];
        
        if (thread_position_in_threadgroup.x == 0)
            for (uint l = 0; l < NUM_LIMBS; ++l)
                atomic_store_explicit(&shared[l], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process in tiles for better cache utilization
        for (uint tile_start = k; tile_start < m; tile_start += tile_size) {
            uint tile_end = min(tile_start + tile_size, m);
            
            for (uint i = tile_start + thread_position_in_grid.x;
                 i < tile_end;
                 i += threads_per_threadgroup)
            {
                float v = R_out[i*n + k];
                multiply_and_accumulate(v, v, local);
            }
        }
        
        add_to_limbs(shared, local);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (thread_position_in_threadgroup.x == 0)
            propagate_carries(shared);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float norm2 = compute_limb_value(shared);
        if (isnan(norm2) || isinf(norm2) || norm2 <= EPSILON) { 
            if (debug[0] == 0.0f) debug[0] = 1.0f; // First fault flag
            debug[13] = (float)k; // Store column index of fault
            norm2 = EPSILON; 
        }
        debug[4] = norm2;
        
        float norm = sqrt(norm2);
        if (isnan(norm) || isinf(norm)) {
             if (debug[0] == 0.0f) debug[0] = 1.1f; // First fault flag
             debug[13] = (float)k; // Store column index of fault
             norm = sqrt(EPSILON); // Use sqrt(EPSILON) as a fallback
        }
        debug[5] = norm;
        
        if (thread_position_in_threadgroup.x == 0) {
            tg_norm = norm;
            float head_val = R_out[k*n + k];
            tg_sign = (head_val >= 0.0f ? 1.0f : -1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        /* ---- 2. Householder head update ----------------------------------- */
        if (thread_position_in_threadgroup.x == 0) {
            R_out[k*n + k] += tg_sign * tg_norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
/* ---- 3. vᵀv computation with optimized limbs --------------------- */
        for (uint l = 0; l < NUM_LIMBS; ++l)
            local[l] = 0;
            
        if (thread_position_in_threadgroup.x == 0)
            for (uint l = 0; l < NUM_LIMBS; ++l)
                atomic_store_explicit(&shared[l], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process in tiles for better cache utilization
        for (uint tile_start = k; tile_start < m; tile_start += tile_size) {
            uint tile_end = min(tile_start + tile_size, m);
            
            for (uint i = tile_start + thread_position_in_grid.x;
                 i < tile_end;
                 i += threads_per_threadgroup)
            {
                float v = R_out[i*n + k];
                multiply_and_accumulate(v, v, local);
            }
        }
        
        add_to_limbs(shared, local);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (thread_position_in_threadgroup.x == 0)
            propagate_carries(shared);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float vtv = compute_limb_value(shared);
        if (isnan(vtv) || isinf(vtv) || vtv <= EPSILON) { 
            if (debug[0] == 0.0f) debug[0] = 2.0f; // First fault flag
            debug[13] = (float)k; // Store column index of fault
            vtv = EPSILON; 
        }
        debug[6] = vtv;
        
        if (thread_position_in_threadgroup.x == 0) {
            float inv = 1.0f / vtv;  // keep raw reciprocal, we'll multiply by 2 only once
            if (isnan(inv) || isinf(inv)) {
                 if (debug[0] == 0.0f) debug[0] = 2.1f; // First fault flag
                 debug[13] = (float)k; // Store column index of fault
                 inv = 0.0f; // Fallback to 0.0
            }
            tg_inv_vt_v = inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_vtv = tg_inv_vt_v;
        debug[7] = inv_vtv;
        
        /* ---- 4. reflection on R (tiled for better cache utilization) ----- */
        for (uint j_tile = k; j_tile < n; j_tile += tile_size) {
            uint j_end = min(j_tile + tile_size, n);
            
            for (uint j = j_tile; j < j_end; ++j) {
                for (uint l = 0; l < NUM_LIMBS; ++l)
                    local[l] = 0;
                    
                threadgroup atomic_uint dshr[NUM_LIMBS];
                if (thread_position_in_threadgroup.x == 0)
                    for (uint l = 0; l < NUM_LIMBS; ++l)
                        atomic_store_explicit(&dshr[l], 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Process in tiles for better cache utilization
                for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                    uint i_end = min(i_tile + tile_size, m);
                    
                    for (uint i = i_tile + thread_position_in_grid.x;
                         i < i_end;
                         i += threads_per_threadgroup)
                    {
                        float vi = R_out[i*n + k];
                        float vj = R_out[i*n + j];
                        multiply_and_accumulate(vi, vj, local);
                    }
                }
                
                add_to_limbs(dshr, local);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                if (thread_position_in_threadgroup.x == 0)
                    propagate_carries(dshr);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                float dot = compute_limb_value(dshr);
                dot /= scale;             /* undo prior column‑k scaling */
                dot = clamp(dot, -1e8f, 1e8f); // Clamp to prevent extreme values
                
                if (isnan(dot) || isinf(dot)) { 
                    if (debug[0] == 0.0f) debug[0] = 3.0f; // First fault flag
                    debug[13] = (float)k; // Store column index of fault
                    debug[14] = (float)j; // Store column index of fault
                    dot = 0.0f; 
                }
                debug[8] = dot;
    
    // Apply Householder reflection: x' = x - 2(v·x/v·v)v
                for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                    uint i_end = min(i_tile + tile_size, m);
                    
                    for (uint i = i_tile + thread_position_in_grid.x;
                         i < i_end;
                         i += threads_per_threadgroup)
                    {
                        float v = R_out[i*n + k];
                        // Apply scaling compensation correctly: (2 * v * dot) / vtv
                        float upd = 2.0f * v * dot * inv_vtv; 
                        
                        if (isnan(upd) || isinf(upd)) { 
                            if (debug[1] == 0.0f) debug[1] = 4.0f; // First fault flag
                            debug[13] = (float)i; // Store row index of fault
                            debug[14] = (float)j; // Store column index of fault
                            continue; 
                        }
                        
                        R_out[i*n + j] -= upd;
                        
                        if (isnan(R_out[i*n + j]) || isinf(R_out[i*n + j])) { 
                            if (debug[2] == 0.0f) debug[2] = 1.0f; // First fault flag
                            debug[13] = (float)i; // Store row index of fault
                            debug[14] = (float)j; // Store column index of fault
                            R_out[i*n + j] = 0.0f; 
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        
        /* ---- 5. reflection on Q (tiled for better cache utilization) ----- */
        for (uint j_tile = 0; j_tile < m; j_tile += tile_size) {
            uint j_end = min(j_tile + tile_size, m);
            
            for (uint j = j_tile; j < j_end; ++j) {
                for (uint l = 0; l < NUM_LIMBS; ++l)
                    local[l] = 0;
                    
                threadgroup atomic_uint dshr[NUM_LIMBS];
                if (thread_position_in_threadgroup.x == 0)
                    for (uint l = 0; l < NUM_LIMBS; ++l)
                        atomic_store_explicit(&dshr[l], 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Process in tiles for better cache utilization
                for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                    uint i_end = min(i_tile + tile_size, m);
                    
                    for (uint i = i_tile + thread_position_in_grid.x;
                         i < i_end;
                         i += threads_per_threadgroup)
                    {
                        float vi = R_out[i*n + k];
                        float qi = Q_out[i*m + j];
                        multiply_and_accumulate(vi, qi, local);
                    }
                }
                
                add_to_limbs(dshr, local);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                if (thread_position_in_threadgroup.x == 0)
                    propagate_carries(dshr);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                float dot = compute_limb_value(dshr);
                /* dot already in physical units for Q path */
                dot /= scale;                    /* undo column‑k scaling on v */
                dot = clamp(dot, -1e8f, 1e8f); // Clamp to prevent extreme values
                
                if (isnan(dot) || isinf(dot)) { 
                    if (debug[3] == 0.0f) debug[3] = 1.0f; // First fault flag
                    debug[13] = (float)k; // Store column index of fault
                    debug[14] = (float)j; // Store column index of fault
                    dot = 0.0f; 
                }
                debug[9] = dot;
                // Apply Householder reflection: x' = x - 2(v·x/v·v)v
                for (uint i_tile = k; i_tile < m; i_tile += tile_size) {
                    uint i_end = min(i_tile + tile_size, m);
                    
                    for (uint i = i_tile + thread_position_in_grid.x;
                         i < i_end;
                         i += threads_per_threadgroup)
                    {
                        float v = R_out[i*n + k];
                        // Apply scaling compensation correctly: (2 * v * dot) / vtv
                        float upd = 2.0f * v * dot * inv_vtv; 
                        
                        if (isnan(upd) || isinf(upd)) { 
                            if (debug[1] == 0.0f) debug[1] = 5.0f; // First fault flag
                            debug[13] = (float)i; // Store row index of fault
                            debug[14] = (float)j; // Store column index of fault
                            continue; 
                        }
                        
                        Q_out[i*m + j] -= upd;
                        
                        if (isnan(Q_out[i*m + j]) || isinf(Q_out[i*m + j])) { 
                            if (debug[3] == 0.0f) debug[3] = 2.0f; // First fault flag
                            debug[13] = (float)i; // Store row index of fault
                            debug[14] = (float)j; // Store column index of fault
                            Q_out[i*m + j] = 0.0f; 
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        
        /* ---- 6. restore original scaling for column k -------------------- */
        for (uint i = k + thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup)
            R_out[i*n + k] /= scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        /* ---- 7. Zero out elements below diagonal in R -------------------- */
        for (uint i = k + 1 + thread_position_in_grid.x;
             i < m;
             i += threads_per_threadgroup)
        {
            R_out[i*n + k] = 0.0f;  // Explicitly zero out sub-diagonal elements
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } /* ── next column k ── */