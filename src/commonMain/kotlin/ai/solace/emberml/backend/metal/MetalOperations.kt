package ai.solace.emberml.backend.metal

import ai.solace.emberml.tensor.common.EmberDType

/**
 * Metal operations for tensor computations using GPU kernels.
 * 
 * This object provides implementations of common tensor operations using Metal kernels
 * for high-performance GPU computation.
 */
object MetalOperations {
    
    /**
     * Element-wise addition of two tensors.
     */
    fun add(a: MetalTensor, b: MetalTensor, context: MetalContext): MetalTensor {
        requireSameShape(a, b)
        
        val result = MetalTensor.zeros(a.shape, a.dtype, context)
        
        val pipelineState = context.createComputePipelineState("add_float")
        val buffers = listOf(a.buffer, b.buffer, result.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((a.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Element-wise subtraction of two tensors.
     */
    fun subtract(a: MetalTensor, b: MetalTensor, context: MetalContext): MetalTensor {
        requireSameShape(a, b)
        
        val result = MetalTensor.zeros(a.shape, a.dtype, context)
        
        val pipelineState = context.createComputePipelineState("subtract_float")
        val buffers = listOf(a.buffer, b.buffer, result.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((a.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Element-wise multiplication of two tensors.
     */
    fun multiply(a: MetalTensor, b: MetalTensor, context: MetalContext): MetalTensor {
        requireSameShape(a, b)
        
        val result = MetalTensor.zeros(a.shape, a.dtype, context)
        
        val pipelineState = context.createComputePipelineState("multiply_float")
        val buffers = listOf(a.buffer, b.buffer, result.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((a.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Element-wise division of two tensors.
     */
    fun divide(a: MetalTensor, b: MetalTensor, context: MetalContext): MetalTensor {
        requireSameShape(a, b)
        
        val result = MetalTensor.zeros(a.shape, a.dtype, context)
        
        val pipelineState = context.createComputePipelineState("divide_float")
        val buffers = listOf(a.buffer, b.buffer, result.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((a.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Matrix multiplication of two tensors.
     */
    fun matmul(a: MetalTensor, b: MetalTensor, context: MetalContext): MetalTensor {
        require(a.shape.size == 2 && b.shape.size == 2) { 
            "Matrix multiplication requires 2D tensors" 
        }
        require(a.shape[1] == b.shape[0]) { 
            "Inner dimensions must match for matrix multiplication" 
        }
        
        val m = a.shape[0]
        val n = b.shape[1]
        val k = a.shape[1]
        
        val resultShape = intArrayOf(m, n)
        val result = MetalTensor.zeros(resultShape, a.dtype, context)
        
        // Create dimensions buffer
        val dimsData = intArrayOf(m, n, k)
        val dimsBuffer = context.createBuffer(dimsData.map { it.toFloat() }.toFloatArray())
        
        val pipelineState = context.createComputePipelineState("matmul_float")
        val buffers = listOf(a.buffer, b.buffer, result.buffer, dimsBuffer)
        
        val threadsPerThreadgroup = MetalSize(16, 16)
        val threadgroupsPerGrid = MetalSize(
            (m + 15) / 16,
            (n + 15) / 16
        )
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Casts a tensor to a different data type.
     */
    fun cast(tensor: MetalTensor, dtype: EmberDType, context: MetalContext): MetalTensor {
        if (tensor.dtype == dtype) {
            return tensor.copy()
        }
        
        val result = MetalTensor.zeros(tensor.shape, dtype, context)
        
        val pipelineState = context.createComputePipelineState("cast_float")
        val buffers = listOf(tensor.buffer, result.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((tensor.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Reshapes a tensor to a new shape.
     */
    fun reshape(tensor: MetalTensor, newShape: IntArray): MetalTensor {
        return tensor.reshape(newShape)
    }
    
    /**
     * Transposes a tensor.
     */
    fun transpose(tensor: MetalTensor, axes: IntArray?, context: MetalContext): MetalTensor {
        require(tensor.shape.size == 2) { "Transpose currently only supports 2D tensors" }
        
        val resultShape = intArrayOf(tensor.shape[1], tensor.shape[0])
        val result = MetalTensor.zeros(resultShape, tensor.dtype, context)
        
        // Create dimensions buffer
        val dimsData = intArrayOf(tensor.shape[0], tensor.shape[1])
        val dimsBuffer = context.createBuffer(dimsData.map { it.toFloat() }.toFloatArray())
        
        val pipelineState = context.createComputePipelineState("transpose_float")
        val buffers = listOf(tensor.buffer, result.buffer, dimsBuffer)
        
        val threadsPerThreadgroup = MetalSize(16, 16)
        val threadgroupsPerGrid = MetalSize(
            (tensor.shape[1] + 15) / 16,
            (tensor.shape[0] + 15) / 16
        )
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return result
    }
    
    /**
     * Copies data from one tensor to another.
     */
    fun copy(source: MetalTensor, destination: MetalTensor, context: MetalContext) {
        require(source.size == destination.size) { 
            "Source and destination tensors must have the same size" 
        }
        
        val pipelineState = context.createComputePipelineState("copy_float")
        val buffers = listOf(source.buffer, destination.buffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((source.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
    }
    
    /**
     * Computes the L2 norm of a tensor.
     */
    fun norm(tensor: MetalTensor, context: MetalContext): Float {
        val pipelineState = context.createComputePipelineState("norm_float")
        val resultBuffer = context.createBuffer(4) // Single float
        val buffers = listOf(tensor.buffer, resultBuffer)
        
        val threadsPerThreadgroup = MetalSize(256)
        val threadgroupsPerGrid = MetalSize((tensor.size + 255) / 256)
        
        context.executeKernel(pipelineState, buffers, threadsPerThreadgroup, threadgroupsPerGrid)
        
        return resultBuffer.toFloatArray()[0]
    }
    
    private fun requireSameShape(a: MetalTensor, b: MetalTensor) {
        require(a.shape.contentEquals(b.shape)) { 
            "Tensors must have the same shape: ${a.shape.contentToString()} vs ${b.shape.contentToString()}" 
        }
    }
}