# Writing Metal Kernels for MLX: A Comprehensive Guide

## Introduction

MLX provides a high-level interface for machine learning on Apple Silicon, but sometimes you need to write custom Metal kernels for performance-critical operations. This guide covers the process of writing, debugging, and optimizing Metal kernels for MLX, with a focus on practical examples and the specific dispatch mechanics used by the framework.

## Basic Structure of a Metal Kernel in MLX

### Kernel Definition

```python
import mlx.core as mx

# Define the kernel source code as a string
# Note: Only the body of the kernel function is needed here.
# MLX automatically generates the function signature based on input_names, output_names, and used attributes.
kernel_source = """
// Your Metal kernel code here
// Access thread ID: uint tid = thread_position_in_grid.x;
// Access threadgroup size: uint3 tpg = threads_per_threadgroup;
// Access grid size (total threads): uint3 gpg = grid_size; // Note: grid_size.x may behave unexpectedly in MLX dispatch
"""

# Compile the kernel
compiled_kernel = mx.fast.metal_kernel(
    name="your_kernel_name",
    source=kernel_source,
    input_names=["input1", "input2"], # Names of input buffers
    output_names=["output"], # Names of output buffers
    ensure_row_contiguous=True # Ensures inputs are row-contiguous (can be set to False for strided access)
)

# Use the kernel
def your_function(input1, input2):
    # Prepare inputs as MLX arrays
    input1_mlx = mx.array(input1)
    input2_mlx = mx.array(input2)
    
    # Determine output shape and dtype
    output_shape = input1_mlx.shape # Example output shape
    output_dtype = input1_mlx.dtype # Example output dtype
    
    # Configure kernel dispatch
    # In MLX, the 'grid' parameter specifies the TOTAL NUMBER OF THREADS to dispatch.
    # The 'threadgroup' parameter specifies the threads per threadgroup.
    total_threads = input1_mlx.size # Example: one thread per element
    threadgroup_size = 32 # Example: a common threadgroup size (multiple of SIMD width)
    
    # Ensure total_threads is a multiple of threadgroup_size for simplicity in some kernels
    # total_threads = (total_threads + threadgroup_size - 1) // threadgroup_size * threadgroup_size
    
    grid = (total_threads, 1, 1) # Total threads in x-dimension
    threadgroup = (threadgroup_size, 1, 1) # Threads per threadgroup in x-dimension
    
    outputs = compiled_kernel(
        inputs=[input1_mlx, input2_mlx],
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        grid=grid, # Total threads
        threadgroup=threadgroup # Threads per threadgroup
    )
    return outputs[0]
```

## Metal Kernel Syntax and Built-in Variables in MLX

Understanding how MLX dispatches Metal kernels is crucial for correctly using built-in variables and writing efficient kernel logic.

### Thread Dispatch Mechanics in MLX

Unlike standard Metal programming where `dispatchThreadgroups` is commonly used and `grid_size` represents the number of threadgroups, MLX's `mlx.fast.metal_kernel` uses a dispatch model closer to `dispatchThreads`.

- **`grid` parameter in `mlx.fast.metal_kernel`**: This specifies the **total number of threads** to be launched in each dimension of the grid.
- **`threadgroup` parameter in `mlx.fast.metal_kernel`**: This specifies the dimensions of the threadgroups. MLX automatically calculates the number of threadgroups needed based on the total threads and the threadgroup size.

### Built-in Variable Semantics in MLX Kernels

Within the Metal kernel source code, the standard Metal built-in variables are available, but their values are populated based on MLX's dispatch:

- **`thread_position_in_grid`**: This provides the linear index of the current thread within the *total number of threads dispatched*. For a 1D grid, `thread_position_in_grid.x` will range from 0 to `total_threads - 1`. This is the primary variable to use for linear indexing of data.
- **`threads_per_threadgroup`**: This correctly reflects the `threadgroup` size specified in the `mlx.fast.metal_kernel` call.
- **`grid_size`**: **Crucially, in MLX's dispatch model, `grid_size` within the kernel does NOT represent the number of threadgroups.** Our testing indicates that `grid_size.x` consistently reads as 0, regardless of the configured total threads or threadgroup size. Relying on `grid_size` for calculating global indices or loop strides is not reliable in MLX.

### Correct Thread Indexing and Loop Patterns

Since `thread_position_in_grid.x` provides the linear index within the total threads, you can use it directly for element-wise operations when the total number of threads equals the number of elements to process.

For operations where the total number of elements is greater than the number of launched threads (e.g., processing a large matrix with a limited number of threads), you need to use a grid-stride loop. However, because `grid_size.x` is unreliable, you cannot calculate the total number of threads within the kernel using `grid_size.x * threads_per_threadgroup.x`.

Instead, you should:

1.  **Calculate the total number of threads to launch in your Python code.** This is the value you pass to the `grid` parameter.
2.  **Pass the total number of threads as an explicit input to your Metal kernel.**
3.  **Use this explicit input in your Metal kernel to calculate the stride for the grid-stride loop.**

### Example: Grid-Stride Loop with Explicit Total Threads

```python
# In your Python code:
total_elements = m * n # Example total elements
threads_per_group = 32
total_threads_to_launch = (total_elements + threads_per_group - 1) // threads_per_group * threads_per_group # Ensure multiple of threadgroup size
grid = (total_threads_to_launch, 1, 1)
threadgroup = (threads_per_group, 1, 1)
total_threads_input = mx.array([total_threads_to_launch], dtype=mx.uint32)

compiled_kernel = mx.fast.metal_kernel(
    # ... other parameters
    input_names=["input", "total_threads_input"], # Add total_threads_input
    # ... other parameters
)

outputs = compiled_kernel(
    inputs=[input_mlx, total_threads_input], # Pass total_threads_input
    grid=grid,
    threadgroup=threadgroup,
    # ... other parameters
)
```

```metal
// In your Metal kernel source:
// Get thread ID
uint tid = thread_position_in_grid.x;

// Get total threads from input
const uint total_threads = total_threads_input[0];

// Process elements using a grid-stride loop with explicit total_threads
for (uint i = tid; i < total_elements; i += total_threads) {
    // Your kernel logic using index 'i'
    output[i] = input[i] * 2.0f; // Example operation
}
```

This approach ensures that the grid-stride loop uses the correct stride based on the actual number of threads launched by MLX, even though `grid_size.x` is unreliable.

## Shared Memory Management

### Declaration and Size Limits

| Aspect | Details | Pitfalls |
|--------|---------|----------|
| Declaration | `threadgroup float shared_mem[SIZE];` | Must be declared at kernel scope |
| Size Limit | 32KB (32,768 bytes) total per threadgroup | Exceeding this will cause compilation failure |
| Data Types | Each `float` is 4 bytes | A 1000-element float array uses 4000 bytes |
| Dynamic Sizing | Not directly supported | Use constants or preprocessor defines |

### Example: Calculating Shared Memory Size

```metal
// For a float array of 1000 elements:
// 1000 * 4 bytes = 4000 bytes (well under the 32KB limit)
threadgroup float shared_mem[1000];

// For a 2D matrix of 100x100 floats:
// 100 * 100 * 4 bytes = 40,000 bytes (exceeds the 32KB limit!)
// Instead, use a smaller size:
threadgroup float shared_matrix[80 * 80]; // 25,600 bytes (under the limit)
```

### Synchronization

Always use barriers when accessing shared memory to avoid race conditions:

```metal
// Write to shared memory
shared_mem[thread_position_in_threadgroup.x] = input[thread_position_in_grid.x];

// Ensure all threads in the threadgroup have written to shared memory
threadgroup_barrier(mem_flags::mem_device);

// Now read from shared memory
float value = shared_mem[other_thread_in_group_idx];
```

## Common Patterns for Matrix Operations

### Thread Allocation for Multiple Matrices

When working with multiple matrices (like in QR decomposition), calculate the total workload by summing the elements of all matrices:

```python
# Calculate total elements for initialization
q_elements = m * m  # Elements in Q matrix
r_elements = m * n  # Elements in R matrix
total_init_elements = q_elements + r_elements  # Total elements to process

# Configure kernel dispatch
threadgroup_size = 32  # Use a reasonable threadgroup size
# Ensure total_threads_to_launch is a multiple of threadgroup_size
total_threads_to_launch = (total_init_elements + threadgroup_size - 1) // threadgroup_size * threadgroup_size

grid = (total_threads_to_launch, 1, 1)
threadgroup = (threadgroup_size, 1, 1)
```

In the kernel, you can then use the thread ID to determine which matrix element to process:

```metal
// Initialize multiple matrices in parallel
if (tid < total_init_elements) {
    // Initialize first matrix (e.g., Q matrix in QR decomposition)
    if (tid < q_elements) {
        uint row = tid / m;
        uint col = tid % m;
        Q_out[tid] = (row == col) ? 1.0f : 0.0f;  // Identity matrix
    }
    // Initialize second matrix (e.g., R matrix in QR decomposition)
    else {
        uint r_idx = tid - q_elements;
        R_out[r_idx] = A[r_idx];  // Copy from input
    }
}
```

### Element-wise Operations

```metal
uint tid = thread_position_in_grid.x;
uint total_elements = total_threads_input[0]; // Get total threads from input

// Process elements in parallel using grid-stride loop
for (uint idx = tid; idx < total_elements; idx += total_threads) {
    output[idx] = func(input[idx]);
}
```

### Matrix Multiplication

For matrix multiplication, you typically use shared memory and tiling to improve performance. The indexing will involve mapping the linear thread ID to 2D or 3D indices corresponding to the matrix elements being computed.

```metal
// Example (simplified, without full tiling logic)
uint tid = thread_position_in_grid.x;
uint total_threads = total_threads_input[0];

const uint m = A_shape[0];
const uint n = B_shape[1];
const uint k = A_shape[1]; // or B_shape[0]

// Each thread computes one element of the output matrix C
if (tid < m * n) {
    uint row = tid / n;
    uint col = tid % n;
    
    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += A[row * k + i] * B[i * n + col];
    }
    
    C[row * n + col] = sum;
}
```

### Reduction Operations

For reductions, you typically use a combination of thread-local accumulation, SIMD operations for fast within-threadgroup reduction, and potentially shared memory for inter-threadgroup reduction.

#### Critical: Barrier Placement in Parallel Reduction

When implementing parallel reduction using shared memory, the placement of barriers is critical. Always place a barrier **before** reading from shared memory to ensure all threads have completed their writes:

```metal
// Example: Parallel reduction with correct barrier placement
threadgroup float shmem[32]; // Shared memory for reduction
    
// Each thread calculates its partial sum
float partial_sum = 0.0f;
if (tid < m) {
    float v_i = R_out[tid * n];
    partial_sum = v_i * v_i;
}
// Store partial sum in shared memory
shmem[thread_position_in_threadgroup.x] = partial_sum;

// Synchronize before reduction
threadgroup_barrier(mem_flags::mem_threadgroup);

// Reduce within threadgroup - CORRECT barrier placement
for (uint stride = threads_per_threadgroup.x/2; stride > 0; stride >>= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier BEFORE reading
    if (thread_position_in_threadgroup.x < stride) {
        shmem[thread_position_in_threadgroup.x] += shmem[thread_position_in_threadgroup.x + stride];
    }
}
```

Incorrect barrier placement (after the read/write operation) can lead to race conditions and undefined behavior:

```metal
// INCORRECT barrier placement - can cause race conditions
for (uint stride = threads_per_threadgroup.x/2; stride > 0; stride >>= 1) {
    if (thread_position_in_threadgroup.x < stride) {
        shmem[thread_position_in_threadgroup.x] += shmem[thread_position_in_threadgroup.x + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier AFTER reading/writing
}
```

#### SIMD-based Reduction Example

```metal
// Example (simplified, within a single threadgroup)
uint tid = thread_position_in_grid.x;
uint num_threads_in_group = threads_per_threadgroup.x;

threadgroup float shared_storage[/* size based on threadgroup_x */];

float thread_partial_sum = 0.0f;
// Accumulate thread-local sum (using grid-stride loop if needed)
// ...

// Store thread-local sum in shared memory
shared_storage[thread_position_in_threadgroup.x] = thread_partial_sum;

// Synchronize within threadgroup
threadgroup_barrier(mem_flags::mem_device);

// Perform reduction within threadgroup using shared memory
// ... (e.g., parallel reduction in shared memory)

// Use SIMD operations for faster reduction within SIMD groups
float simd_sum_val = simd_sum(thread_partial_sum);

// Only one thread per SIMD group needs to write to shared memory or global memory
// ...
```

## Avoiding Parameter Conflicts

### Shape Parameter Handling

MLX automatically generates shape parameters for tensor inputs. For example, if you have an input named "A", MLX will automatically generate a parameter named "A_shape" of type `const constant int*`.

To avoid conflicts, do not explicitly include shape parameters in your `input_names` list that match this pattern. Instead, use the automatically generated shape parameters:

```python
# CORRECT: Let MLX handle the shape parameter
compiled_kernel = mx.fast.metal_kernel(
    name="your_kernel_name",
    source=kernel_source,
    input_names=["A"],  # Do not include "A_shape"
    output_names=["output"],
    ensure_row_contiguous=True
)
```

In your kernel code, you can access the shape parameter directly:

```metal
// Access the automatically generated shape parameter
const uint m = A_shape[0];
const uint n = A_shape[1];
```

If you need to pass a shape parameter with a different type (e.g., uint32), use a different name:

```python
# CORRECT: Use a different name for custom shape parameter
shape_uint32 = mx.array(A.shape, dtype=mx.uint32)
compiled_kernel = mx.fast.metal_kernel(
    name="your_kernel_name",
    source=kernel_source,
    input_names=["A", "shape_uint32"],  # Different name
    output_names=["output"],
    ensure_row_contiguous=True
)
```

## Debugging Strategies

### 1. Incremental Testing

Start with a minimal kernel that just writes the thread ID to an output buffer to verify basic execution and thread indexing. Gradually add complexity, testing each step.

### 2. Controlled Inputs

Use small, fixed inputs with known expected outputs to make debugging easier.

### 3. Debug Buffer

Include a debug buffer in your outputs to store intermediate values and flags:

```python
# Include a debug buffer in outputs
output_shapes = [(m, n), (16,)]  # Main output and debug buffer
output_dtypes = [mx.float32, mx.float32]

outputs = compiled_kernel(
    inputs=[input_mlx],
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    grid=grid,
    threadgroup=threadgroup
)

result, debug_info = outputs
```

In your kernel, write to the debug buffer at key points:

```metal
// Set debug values
if (tid == 0) {
    dbg[0] = 1.0f;  // Execution flag
    dbg[1] = float(m);  // Dimensions
    dbg[2] = float(n);
    // Store intermediate results
    dbg[7] = norm;
    dbg[8] = dot_product;
}
```

### 3. Error Message Analysis

Pay close attention to compilation errors from the Metal compiler. Use `verbose=True` in `mlx.fast.metal_kernel` to see the generated Metal source code, which can help pinpoint syntax errors or incorrect variable usage.

### 4. Sandbox Testing

Create isolated Python files to test specific kernel functionalities or configurations.

### 5. Metal Validation Layers

Enable Metal validation layers using environment variables (`MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`) for enhanced diagnostics.

## Performance Optimization

### 1. Thread Configuration

- **Total Threads (`grid`)**: Set the total number of threads to be at least the total number of elements you need to process.
- **Threadgroup Size (`threadgroup`)**: Choose a threadgroup size that is a multiple of the SIMD width (typically 32) for better SIMD utilization. Common threadgroup sizes are 32, 64, 128, 256, 512, or 1024. The total number of threads in a threadgroup (`threadgroup.x * threadgroup.y * threadgroup.z`) must not exceed `maxTotalThreadsPerThreadgroup` (typically 1024).
- **Balance**: Aim for a balance between the number of threadgroups and the threadgroup size to keep the GPU busy.

### 2. Memory Access Patterns

- **Coalesced Access**: Design your kernel to ensure that threads within a threadgroup access memory in a coalesced pattern (adjacent threads access adjacent memory locations). This is crucial for performance.
- **Shared Memory**: Use shared memory to reduce global memory access, but be mindful of the 32KB limit and bank conflicts.
- **Data Types**: Use appropriate data types (e.g., `half` for float16) to reduce memory bandwidth and improve performance.

### 3. Algorithmic Optimizations

- **Tiling**: Implement tiling for matrix operations to improve data locality and reduce global memory access.
- **SIMD Operations**: Leverage SIMD operations (`simd_sum`, `simd_max`, etc.) for fast parallel operations within SIMD groups.
- **Work Distribution**: Ensure work is distributed evenly among threads to avoid idle threads.

## Conclusion and Recommendations

Writing custom Metal kernels in MLX requires a clear understanding of the framework's specific dispatch mechanics and how built-in variables are populated. The key is to recognize that the `grid` parameter specifies the total number of threads and to use `thread_position_in_grid.x` as the primary linear index.

While `grid_size.x` is not reliably available as the number of threadgroups, passing the total number of threads as an explicit input allows for correct implementation of grid-stride loops and efficient processing of larger datasets.

Developers should adopt a "total threads" mindset when configuring kernel dispatch in MLX and carefully implement indexing and loop patterns based on `thread_position_in_grid.x`. Incremental testing, controlled inputs, and analysis of generated code are essential debugging strategies.

By following these guidelines and best practices, developers can effectively leverage the power of Metal for performance-critical operations within the MLX framework.