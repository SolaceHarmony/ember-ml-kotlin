# Metal Kernel Integration for Ember ML Kotlin

## Overview

Ember ML Kotlin integrates with Metal, Apple's GPU programming framework, to accelerate tensor operations on Apple platforms. This integration allows us to leverage the power of Apple GPUs for machine learning tasks while maintaining a consistent API across all platforms.

## Motivation

Metal integration is particularly important for several reasons:

1. **Performance**: Metal provides direct access to the GPU, enabling significant performance improvements for tensor operations.
2. **Float64 Limitations**: Apple's Metal and MLX frameworks don't natively support Float64 operations, necessitating our bitwise workaround approach.
3. **Platform Optimization**: Proper Metal integration allows us to fully utilize Apple hardware capabilities.
4. **Consistent API**: By integrating Metal behind our backend abstraction, we can provide a consistent API regardless of the underlying hardware.

## Architecture

The Metal kernel integration consists of several components:

### 1. Metal Kernel Bindings

Kotlin Native provides the ability to interoperate with native libraries, including Metal. We use this capability to create bindings to Metal:

```kotlin
@ExperimentalForeignApi
class MetalContext(val device: MTLDevice, val commandQueue: MTLCommandQueue) {
    companion object {
        fun create(): MetalContext? {
            val device = MTLCreateSystemDefaultDevice() ?: return null
            val commandQueue = device.newCommandQueue() ?: return null
            return MetalContext(device, commandQueue)
        }
    }

    fun createBuffer(size: Int): MTLBuffer? {
        return device.newBuffer(size.toULong(), MTLResourceOptions.MTLResourceStorageModeShared)
    }

    fun createComputePipelineState(function: String): MTLComputePipelineState? {
        val library = device.newDefaultLibrary() ?: return null
        val function = library.newFunction(function) ?: return null
        return device.newComputePipelineState(function)
    }

    fun execute(
        pipelineState: MTLComputePipelineState,
        buffers: List<MTLBuffer>,
        threadgroupSize: MTLSize,
        gridSize: MTLSize
    ) {
        val commandBuffer = commandQueue.commandBuffer() ?: return
        val computeEncoder = commandBuffer.computeCommandEncoder() ?: return

        computeEncoder.setComputePipelineState(pipelineState)

        buffers.forEachIndexed { index, buffer ->
            computeEncoder.setBuffer(buffer, 0u, index.toUInt())
        }

        computeEncoder.dispatchThreadgroups(gridSize, threadgroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
```

### 2. Metal Kernel Library

We provide a library of Metal kernels for common tensor operations:

```metal
#include <metal_stdlib>
using namespace metal;

// Basic element-wise operations
kernel void add_float(device const float* a,
                     device const float* b,
                     device float* result,
                     uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

kernel void multiply_float(device const float* a,
                          device const float* b,
                          device float* result,
                          uint index [[thread_position_in_grid]]) {
    result[index] = a[index] * b[index];
}

// Matrix operations
kernel void matmul_float(device const float* a,
                        device const float* b,
                        device float* result,
                        device const uint* dims,
                        uint2 position [[thread_position_in_grid]]) {
    uint m = dims[0];
    uint n = dims[1];
    uint k = dims[2];

    uint row = position.x;
    uint col = position.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (uint i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        result[row * n + col] = sum;
    }
}

// Advanced operations like SVD
kernel void svd_step_float(device const float* matrix,
                          device float* u,
                          device float* s,
                          device float* vt,
                          device const uint* dims,
                          uint2 position [[thread_position_in_grid]]) {
    // SVD implementation
    // ...
}
```

### 3. Metal Backend Implementation

We implement a Metal backend that uses these kernels:

```kotlin
class MetalBackend : Backend {
    private val context: MetalContext? = MetalContext.create()
    private val kernelCache = mutableMapOf<String, MTLComputePipelineState>()

    override fun name(): String = "metal"

    override fun isAvailable(): Boolean = context != null

    override fun priority(): Int = 200  // Higher priority than CPU

    override fun matmul(a: EmberTensor, b: EmberTensor): EmberTensor {
        if (context == null) throw IllegalStateException("Metal not available")

        // Unwrap tensors to get raw data
        val aData = (a.data as FloatArray)
        val bData = (b.data as FloatArray)

        // Create output tensor
        val m = a.shape.dimensions[0]
        val n = b.shape.dimensions[1]
        val k = a.shape.dimensions[1]
        val resultShape = EmberShape(m, n)
        val resultData = FloatArray(m * n)

        // Create Metal buffers
        val aBuffer = context.createBuffer(aData.size * 4)!!
        val bBuffer = context.createBuffer(bData.size * 4)!!
        val resultBuffer = context.createBuffer(resultData.size * 4)!!
        val dimsBuffer = context.createBuffer(3 * 4)!!

        // Copy data to buffers
        memcpy(aBuffer.contents(), aData, aData.size * 4)
        memcpy(bBuffer.contents(), bData, bData.size * 4)

        val dims = UIntArray(3)
        dims[0] = m.toUInt()
        dims[1] = n.toUInt()
        dims[2] = k.toUInt()
        memcpy(dimsBuffer.contents(), dims, dims.size * 4)

        // Get or create compute pipeline
        val pipelineState = getComputePipeline("matmul_float")

        // Calculate threadgroup and grid sizes
        val threadgroupSize = MTLSizeMake(16u, 16u, 1u)
        val gridSize = MTLSizeMake(
            (m + 15u) / 16u,
            (n + 15u) / 16u,
            1u
        )

        // Execute kernel
        context.execute(
            pipelineState,
            listOf(aBuffer, bBuffer, resultBuffer, dimsBuffer),
            threadgroupSize,
            gridSize
        )

        // Copy result back
        memcpy(resultData, resultBuffer.contents(), resultData.size * 4)

        // Create and return tensor
        return EmberTensor(resultData, resultShape, a.dtype, a.device, a.requiresGrad)
    }

    // Other operations...

    private fun getComputePipeline(function: String): MTLComputePipelineState {
        return kernelCache.getOrPut(function) {
            context!!.createComputePipelineState(function)
                ?: throw IllegalStateException("Failed to create compute pipeline for $function")
        }
    }
}
```

### 4. SVD Implementation

One of the key operations we port from the Python implementation is the SVD (Singular Value Decomposition) algorithm. This is particularly important for many machine learning tasks:

```kotlin
class MetalSVD {
    private val context: MetalContext? = MetalContext.create()

    fun svd(matrix: EmberTensor): Triple<EmberTensor, EmberTensor, EmberTensor> {
        if (context == null) throw IllegalStateException("Metal not available")

        // Unwrap tensor to get raw data
        val matrixData = (matrix.data as FloatArray)

        // Get dimensions
        val m = matrix.shape.dimensions[0]
        val n = matrix.shape.dimensions[1]

        // Create output tensors
        val uData = FloatArray(m * m)
        val sData = FloatArray(minOf(m, n))
        val vtData = FloatArray(n * n)

        // Create Metal buffers
        val matrixBuffer = context.createBuffer(matrixData.size * 4)!!
        val uBuffer = context.createBuffer(uData.size * 4)!!
        val sBuffer = context.createBuffer(sData.size * 4)!!
        val vtBuffer = context.createBuffer(vtData.size * 4)!!
        val dimsBuffer = context.createBuffer(2 * 4)!!

        // Copy data to buffers
        memcpy(matrixBuffer.contents(), matrixData, matrixData.size * 4)

        val dims = UIntArray(2)
        dims[0] = m.toUInt()
        dims[1] = n.toUInt()
        memcpy(dimsBuffer.contents(), dims, dims.size * 4)

        // Implement SVD algorithm using multiple kernel calls
        // ...

        // Copy results back
        memcpy(uData, uBuffer.contents(), uData.size * 4)
        memcpy(sData, sBuffer.contents(), sData.size * 4)
        memcpy(vtData, vtBuffer.contents(), vtData.size * 4)

        // Create and return tensors
        val u = EmberTensor(uData, EmberShape(m, m), matrix.dtype, matrix.device, matrix.requiresGrad)
        val s = EmberTensor(sData, EmberShape(minOf(m, n)), matrix.dtype, matrix.device, matrix.requiresGrad)
        val vt = EmberTensor(vtData, EmberShape(n, n), matrix.dtype, matrix.device, matrix.requiresGrad)

        return Triple(u, s, vt)
    }
}
```

## Platform-Specific Considerations

### macOS and iOS

On Apple platforms, we use the Metal framework directly:

```kotlin
actual object PlatformBackend {
    actual fun createNativeBackend(): Backend {
        return if (MetalBackend().isAvailable()) {
            MetalBackend()
        } else {
            CPUBackend()
        }
    }
}
```

### Other Platforms

On non-Apple platforms, we fall back to other backends:

```kotlin
actual object PlatformBackend {
    actual fun createNativeBackend(): Backend {
        return when {
            VulkanBackend().isAvailable() -> VulkanBackend()
            else -> CPUBackend()
        }
    }
}
```

## Performance Considerations

Metal provides significant performance benefits, but there are several considerations:

1. **Data Transfer**: Transferring data between CPU and GPU can be a bottleneck. We minimize this by:
   - Keeping data on the GPU as long as possible
   - Using shared memory when appropriate
   - Batching operations to reduce transfer frequency

2. **Kernel Compilation**: Metal kernels are compiled at runtime, which can introduce latency. We address this by:
   - Caching compiled kernels
   - Pre-warming the cache for common operations
   - Using a background thread for compilation

3. **Memory Management**: GPU memory is a limited resource. We manage it by:
   - Implementing a tensor cache to reuse memory
   - Releasing unused tensors promptly
   - Monitoring memory usage and adapting accordingly

## Example: Matrix Multiplication

Here's a complete example of matrix multiplication using Metal:

```kotlin
fun matmulMetal(a: EmberTensor, b: EmberTensor): EmberTensor {
    // Get Metal context
    val context = MetalContext.create() ?: throw IllegalStateException("Metal not available")

    // Unwrap tensors
    val aData = (a.data as FloatArray)
    val bData = (b.data as FloatArray)

    // Get dimensions
    val m = a.shape.dimensions[0]
    val n = b.shape.dimensions[1]
    val k = a.shape.dimensions[1]

    // Create result tensor
    val resultShape = EmberShape(m, n)
    val resultData = FloatArray(m * n)

    // Create Metal buffers
    val aBuffer = context.createBuffer(aData.size * 4)!!
    val bBuffer = context.createBuffer(bData.size * 4)!!
    val resultBuffer = context.createBuffer(resultData.size * 4)!!
    val dimsBuffer = context.createBuffer(3 * 4)!!

    // Copy data to buffers
    memcpy(aBuffer.contents(), aData, aData.size * 4)
    memcpy(bBuffer.contents(), bData, bData.size * 4)

    val dims = UIntArray(3)
    dims[0] = m.toUInt()
    dims[1] = n.toUInt()
    dims[2] = k.toUInt()
    memcpy(dimsBuffer.contents(), dims, dims.size * 4)

    // Create compute pipeline
    val library = context.device.newDefaultLibrary() ?: throw IllegalStateException("Failed to create Metal library")
    val function = library.newFunction("matmul_float") ?: throw IllegalStateException("Failed to find matmul_float function")
    val pipelineState = context.device.newComputePipelineState(function) ?: throw IllegalStateException("Failed to create compute pipeline")

    // Calculate threadgroup and grid sizes
    val threadgroupSize = MTLSizeMake(16u, 16u, 1u)
    val gridSize = MTLSizeMake(
        (m + 15u) / 16u,
        (n + 15u) / 16u,
        1u
    )

    // Create command buffer and encoder
    val commandBuffer = context.commandQueue.commandBuffer() ?: throw IllegalStateException("Failed to create command buffer")
    val computeEncoder = commandBuffer.computeCommandEncoder() ?: throw IllegalStateException("Failed to create compute encoder")

    // Set pipeline state and buffers
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(aBuffer, 0u, 0u)
    computeEncoder.setBuffer(bBuffer, 0u, 1u)
    computeEncoder.setBuffer(resultBuffer, 0u, 2u)
    computeEncoder.setBuffer(dimsBuffer, 0u, 3u)

    // Dispatch threadgroups
    computeEncoder.dispatchThreadgroups(gridSize, threadgroupSize)
    computeEncoder.endEncoding()

    // Execute and wait
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Copy result back
    memcpy(resultData, resultBuffer.contents(), resultData.size * 4)

    // Create and return tensor
    return EmberTensor(resultData, resultShape, a.dtype, "metal", a.requiresGrad)
}
```

## Future Directions

1. **Kernel Fusion**: Implement kernel fusion to reduce memory transfers and improve performance
2. **Custom Kernels**: Develop specialized kernels for common neural network operations
3. **Automatic Kernel Generation**: Generate Metal kernels from high-level operation descriptions
4. **Multi-GPU Support**: Support multiple GPUs for parallel computation
5. **Cross-Platform Abstraction**: Create a unified API that works across Metal, Vulkan, and other GPU frameworks
