# CUDA Kernel Insights from hyena-dna

## Introduction

The hyena-dna repository includes a sophisticated CUDA kernel implementation for FFT convolution. This document captures key insights from analyzing this implementation, which can inform our architecture design for Ember ML.

## Key Insights

### 1. FFT Implementation

The code uses cuFFTDx, a CUDA library for FFT operations, to implement the forward and inverse FFT operations. This provides high-performance FFT capabilities that are essential for efficient sequence processing.

```cpp
using FFT_base = decltype(cufftdx::Block() + cufftdx::Size<FFT_SIZE>() + cufftdx::Precision<float>() +
                        cufftdx::ElementsPerThread<EPT>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>()
                        + cufftdx::Type<cufftdx::fft_type::c2c>());

using FFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::forward>());
using IFFT = decltype(FFT_base() + cufftdx::Direction<cufftdx::fft_direction::inverse>());
```

### 2. Real FFT Optimization

The code implements a real FFT of size 2*N by calling a complex FFT of size N, which is an optimization technique. This is particularly useful for processing real-valued signals, which is common in many machine learning applications.

```cpp
template<typename FFT>
inline __device__ void rfft(c10::complex<float> (&thread_data)[FFT::elements_per_thread],
                            c10::complex<float> *shared_mem){
    // Implementation of real FFT using complex FFT
    // ...
}
```

### 3. Data Type Support

The implementation supports various data types including float, half-precision (fp16), and BFloat16. This flexibility allows for optimizing memory usage and computation speed based on the precision requirements of the task.

```cpp
template void fftconv_fwd_cuda_dispatch<float, float>(...);
template void fftconv_fwd_cuda_dispatch<float, at::Half>(...);
template void fftconv_fwd_cuda_dispatch<at::Half, at::Half>(...);
template void fftconv_fwd_cuda_dispatch<at::BFloat16, at::BFloat16>(...);
```

### 4. Activation Functions

The code includes implementations of GELU and its derivative for activation functions. These are implemented efficiently for GPU computation.

```cpp
template<int N>
inline __device__ void gelu(float (&output)[N], const float (&input)[N]) {
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        output[i] = input[i] * 0.5 * (1 + erff(input[i] * kAlpha));
    }
}

template<int N>
inline __device__ void dgelu(float (&grad_input)[N], const float (&grad_output)[N], const float (&input)[N]) {
    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    constexpr float kAlpha = M_SQRT1_2;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        const float cdf = 0.5 * (1 + erff(input[i] * kAlpha));
        const float pdf = expf(-0.5 * input[i] * input[i]) * kBeta;
        grad_input[i] = grad_output[i] * (cdf + input[i] * pdf);
    }
}
```

### 5. Batch Processing

The code is designed to handle batch processing efficiently, which is essential for training neural networks on large datasets.

```cpp
unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( batch_size / FPB ) ) };
unsigned int H_per_grid { static_cast<unsigned int>( std::ceil( H / FPB ) ) };
dim3 block(batch_size, H_per_grid / head_dim, head_dim);
```

### 6. Memory Management

The code carefully manages shared memory for efficient computation. This is crucial for GPU performance, as shared memory is much faster than global memory.

```cpp
extern __shared__ cfloat_t shared_mem[];
// ...
const auto shared_memory_size = std::max({FFT::shared_memory_size, IFFT::shared_memory_size, 8 * FFT_SIZE});
CUDA_RT_CALL( cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size ));
```

### 7. Forward and Backward Passes

The implementation includes both forward and backward passes for training. The backward pass computes gradients with respect to all inputs, which is necessary for backpropagation.

```cpp
template <typename input_t, typename output_t=input_t>
void fftconv_fwd_cuda_dispatch(...) {
    // Forward pass implementation
    // ...
}

template <typename input_t, typename output_t=input_t>
void fftconv_bwd_cuda_dispatch(...) {
    // Backward pass implementation
    // ...
}
```

### 8. Optimizations

The code includes various optimizations for performance, such as using lookup tables for twiddle factors, unrolling loops, and minimizing synchronization points.

```cpp
// Reading from lookup table is faster than computing the twiddle
int quadrant = i / (EPT / 4);
cfloat_t twiddle = twiddle_from_lut<N * 2>(quadrant, index);
```

### 9. Multi-Head Support

The code supports multi-head operations, which is important for transformer-like architectures. This allows for parallel processing of different attention heads.

```cpp
switch (head_dim) {
    case 1:
        // Single-head implementation
        // ...
        break;
    case 8:
        // Multi-head implementation
        // ...
        break;
    default:
        AT_ERROR("fftconv forward not implemented for this head_dim");
}
```

### 10. Parallelization

The code is designed to take advantage of CUDA's parallel processing capabilities, with careful attention to thread and block organization for optimal performance.

```cpp
__launch_bounds__( FFT::max_threads_per_block )
__global__ void fftconv_fwd_kernel(...) {
    // Kernel implementation
    // ...
}
```

## Implications for Ember ML

These insights have several implications for our Ember ML architecture:

1. **Backend-Specific Optimizations**: We should ensure our architecture allows for backend-specific optimizations like these CUDA kernels for PyTorch, while maintaining a clean frontend interface.

2. **Flexible Data Type Support**: Our architecture should support various data types and precision levels, with the ability to switch between them based on the task requirements.

3. **Efficient Memory Management**: We should design our components with efficient memory management in mind, particularly for GPU backends.

4. **Parallelization Support**: Our architecture should be designed to take advantage of parallel processing capabilities, whether on CPU or GPU.

5. **Activation Function Flexibility**: We should provide flexible options for activation functions, including efficient implementations of common functions like GELU.

6. **Forward and Backward Pass Integration**: Our architecture should seamlessly integrate forward and backward passes for training.

7. **Multi-Head Support**: We should ensure our architecture supports multi-head operations for transformer-like models.

8. **Optimization Techniques**: We should incorporate optimization techniques like lookup tables, loop unrolling, and minimizing synchronization points where appropriate.

## Implementation Strategy

To incorporate these insights into Ember ML, we propose the following strategy:

1. **Backend Interface**: Define a clear interface for backend-specific implementations, allowing for optimized CUDA kernels for PyTorch while maintaining a clean frontend.

2. **Data Type Abstraction**: Implement a data type abstraction layer that allows for switching between different precision levels.

3. **Memory Management Utilities**: Develop utilities for efficient memory management, particularly for GPU backends.

4. **Parallelization Utilities**: Create utilities for parallelizing operations across multiple cores or devices.

5. **Activation Function Library**: Build a library of efficient activation function implementations, including GELU and its derivatives.

6. **Training Integration**: Ensure seamless integration of forward and backward passes for training.

7. **Multi-Head Support**: Implement multi-head support for transformer-like models.

8. **Optimization Techniques**: Incorporate optimization techniques like lookup tables and loop unrolling where appropriate.

## Conclusion

The CUDA kernel implementation in hyena-dna provides valuable insights into efficient FFT convolution on GPUs. By incorporating these insights into our Ember ML architecture, we can ensure high performance while maintaining a clean and flexible interface for users.