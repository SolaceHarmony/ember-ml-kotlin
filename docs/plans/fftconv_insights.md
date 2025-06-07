# FFT Convolution Insights from hyena-dna

## Introduction

The hyena-dna repository implements a novel approach to fast attention using FFT (Fast Fourier Transform) convolution. This document captures insights from exploring their implementation, particularly the `fftconv` module in the `csrc/fftconv` directory.

## Overview

The FFT convolution implementation in hyena-dna is a CUDA-accelerated operation that enables efficient sequence mixing, which is a key component in modern sequence models like Hyena. By leveraging the properties of convolution in the frequency domain, this approach achieves linear complexity with respect to sequence length, making it more efficient than traditional attention mechanisms for long sequences.

## Implementation Details

The implementation consists of several key components:

1. **C++ Extension Interface**: The `fftconv.cpp` file defines a PyTorch C++ extension that serves as a bridge between Python and the CUDA implementation.

2. **CUDA Kernel**: The `fftconv_cuda.cu` file contains the actual CUDA implementation of the FFT convolution operation.

3. **Python Wrapper**: The `launch_fftconv.py` file provides a Python interface to the C++ extension.

### Key Features

1. **Data Type Support**: The implementation supports various data types:
   - Float32 (default)
   - Float16 (half-precision)
   - BFloat16

2. **Optimizations**:
   - Half-precision (fp16) support for improved performance (but only for head dimension 8)
   - Constraints on FFT size (must be at least 16, at most 16384, and a power of 2)
   - Sequence length must be at most half the FFT size

3. **Flexible Interface**:
   - Support for optional tensors (v, q, dropout_mask)
   - Control over activation functions (GELU can be applied to different inputs)
   - Output layout options (HBL - Head-Batch-Length)

## Function Signature

The main function for the forward pass has the following signature:

```cpp
torch::Tensor fftconv_fwd(
    torch::Tensor u,                    // Input tensor
    torch::Tensor filter,               // Complex filter tensor
    torch::Tensor D,                    // Diagonal matrix
    c10::optional<torch::Tensor> v,     // Optional input tensor
    int head_dim,                       // Dimension of attention heads
    c10::optional<torch::Tensor> q,     // Optional query tensor
    c10::optional<torch::Tensor> dropout_mask, // Optional dropout mask
    bool gelu,                          // Whether to apply GELU activation
    bool gelu_inp,                      // Whether to apply GELU to input
    bool gelu_q,                        // Whether to apply GELU to query
    int fft_size,                       // Size of the FFT
    bool force_fp16_output,             // Whether to force fp16 output
    bool output_hbl_layout,             // Whether to use HBL layout for output
    bool fftfp16                        // Whether to use fp16 for FFT
)
```

## How It Works

The FFT convolution operation works as follows:

1. **Input Preparation**: The input tensor `u` is prepared for convolution.

2. **FFT Transformation**: The input and filter are transformed to the frequency domain using FFT.

3. **Frequency Domain Operations**:
   - Multiplication with the filter in the frequency domain
   - Application of optional operations (GELU, dropout)

4. **Inverse FFT**: The result is transformed back to the time domain using inverse FFT.

5. **Post-Processing**: Additional operations like diagonal scaling and optional tensor operations are applied.

## Integration with Hyena

This FFT convolution operation is a core component of the Hyena architecture, enabling efficient sequence mixing with linear complexity. It replaces traditional attention mechanisms while maintaining or improving performance on long-sequence tasks.

## Relevance to Ember ML

For Ember ML, this FFT convolution approach offers several benefits:

1. **Efficiency**: Linear complexity with respect to sequence length, making it suitable for long-sequence tasks.

2. **Flexibility**: Support for various data types and operations, allowing for adaptation to different use cases.

3. **Performance**: CUDA acceleration for high-performance computation.

We should consider incorporating this approach into our sequence-mixing primitives, particularly for scenarios where traditional attention mechanisms might be too computationally expensive.

## Implementation Strategy

To incorporate FFT convolution into Ember ML:

1. **Create a Wrapper**: Develop a Python wrapper around the CUDA implementation that integrates with our tensor abstraction.

2. **Define a Sequence-Mixing Primitive**: Implement an FFTConv class as a sequence-mixing primitive in our architecture.

3. **Optimize for Different Backends**: Ensure the implementation works efficiently with different backends (PyTorch, NumPy, MLX).

4. **Provide Configuration Options**: Allow users to configure parameters like FFT size, precision, and activation functions.

## Conclusion

The FFT convolution implementation in hyena-dna represents a novel approach to efficient sequence mixing. By incorporating this approach into Ember ML, we can enhance our architecture with a powerful and efficient alternative to traditional attention mechanisms, particularly for long-sequence tasks.