# Liquid Structural State-Space Models (Liquid-S4) - Paper Summary

## Overview

This document summarizes the key concepts, findings, and hyperparameters from the paper "Liquid Structural State-Space Models" by Hasani et al. (2022) from MIT CSAIL, which forms the theoretical foundation for our Phonological Loop Neural Network implementation.

## Key Concepts

### Liquid Time-Constant Networks (LTCs)

LTCs are causal continuous-time neural networks with an input-dependent state transition module that enables them to adapt to incoming inputs at inference time. They are dynamic causal models that demonstrate strong generalization capabilities but have been limited by scalability issues.

### Structural State-Space Models (S4)

S4 models have shown impressive performance on sequence modeling tasks through:
1. High-order polynomial projection operators (HiPPO) for memorizing signal history
2. Diagonal plus low-rank parametrization of the state transition matrix
3. Efficient kernel computation in the frequency domain

### Liquid-S4: Combining LTCs and S4

Liquid-S4 combines the strengths of both approaches by using a linearized LTC state-space model:

```
ẋ(t) = [A + B·u(t)]·x(t) + B·u(t), y(t) = C·x(t)
```

This differs from standard S4 models which use:

```
ẋ(t) = A·x(t) + B·u(t), y(t) = C·x(t)
```

The key innovation is the input-dependent state transition module `[A + B·u(t)]` which allows the model to adapt to incoming inputs.

## Implementation Details

### Liquid-S4 Kernel

The Liquid-S4 kernel consists of two components:
1. The standard S4 convolution kernel
2. An additional "liquid kernel" that accounts for the auto-correlation of input signals

Two variants of the liquid kernel are described:
- **KB mode**: Full kernel that includes the state transition matrix A
- **PB mode**: Simplified kernel that sets A to identity for input correlation terms (generally performs better)

### Computational Complexity

The computational complexity of Liquid-S4 is O(N + L + p_max·L̃), where:
- N is the state size
- L is the sequence length
- p_max is the maximum liquid order
- L̃ is the number of terms used to compute the input correlation vector

## Performance Highlights

Liquid-S4 achieved state-of-the-art performance on:
- Long Range Arena benchmark: 87.32% average accuracy
- BIDMC vital signs dataset
- Speech Command recognition: 96.78% accuracy with 30% fewer parameters
- Sequential CIFAR image classification

## Hyperparameters

The paper provides the following hyperparameters for best performance:

| Task | Depth | Features | State Size | Normalization | Pre-norm | Dropout | Learning Rate | Batch Size | Epochs | Weight Decay | Liquid Order |
|------|-------|----------|------------|---------------|----------|---------|---------------|------------|--------|--------------|--------------|
| ListOps | 9 | 128 | 7 | BN | True | 0.01 | 0.002 | 12 | 30 | 0.03 | 5 |
| IMDB | 4 | 128 | 7 | BN | True | 0.1 | 0.003 | 8 | 50 | 0.01 | 6 |
| AAN | 6 | 256 | 64 | BN | False | 0.2 | 0.005 | 16 | 20 | 0.05 | 2 |
| CIFAR | 6 | 512 | 512 | LN | False | 0.1 | 0.01 | 16 | 200 | 0.03 | 3 |
| Pathfinder | 6 | 256 | 64 | BN | True | 0.0 | 0.0004 | 4 | 200 | 0.03 | 2 |
| Path-X | 6 | 320 | 64 | BN | True | 0.0 | 0.001 | 8 | 60 | 0.05 | 2 |
| Speech Commands | 6 | 128 | 7 | BN | True | 0.0 | 0.008 | 10 | 50 | 0.05 | 2 |
| BIDMC (HR) | 6 | 128 | 256 | LN | True | 0.0 | 0.005 | 32 | 500 | 0.01 | 3 |
| BIDMC (RR) | 6 | 128 | 256 | LN | True | 0.0 | 0.01 | 32 | 500 | 0.01 | 2 |
| BIDMC (SpO2) | 6 | 128 | 256 | LN | True | 0.0 | 0.01 | 32 | 500 | 0.01 | 4 |
| sCIFAR | 6 | 512 | 512 | LN | False | 0.1 | 0.01 | 50 | 200 | 0.03 | 3 |

## Implementation Recommendations

1. **Liquid Order**: Start with p=2 and increase if needed. Higher values consistently enhance performance.
2. **Kernel Choice**: Use the PB kernel mode for better performance and computational efficiency.
3. **State Size**: Liquid-S4 performs well with smaller state sizes (as low as 7 for some tasks).
4. **Learning Rate**: Liquid-S4 generally requires smaller learning rates compared to S4 and S4D.
5. **Causal vs. Bidirectional**: Liquid-S4 works better as a causal model without bidirectional configuration.
6. **Delta Parameters**: Set Δt_max to 0.2 and Δt_min proportional to 1/sequence_length.

## Theoretical Insights

The paper provides theoretical analysis showing that the liquid kernel accounts for the similarities between time-lagged signals, which helps the model better capture temporal dependencies. This is particularly valuable for tasks involving complex temporal patterns and noisy environments.

The input-dependent state transition mechanism allows Liquid-S4 to function as a dynamic causal model that can adapt to the characteristics of the input signal during inference, rather than having fixed dynamics like standard S4 models.

## Citation

```
@article{hasani2022liquid,
  title={Liquid Structural State-Space Models},
  author={Hasani, Ramin and Lechner, Mathias and Wang, Tsun-Hsuan and Chahine, Makram and Amini, Alexander and Rus, Daniela},
  journal={arXiv preprint arXiv:2209.12951},
  year={2022}
}