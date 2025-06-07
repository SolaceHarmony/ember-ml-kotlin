# Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture

**Daniel Y. Fu, Simran Arora, Jessica Grogan, Isys Johnson, Sabri Eyuboglu, Armin W. Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré**  
*Stanford University, University at Buffalo, SUNY*  
*Contact: danfu@cs.stanford.edu*  
*October 18, 2023*

---

## Abstract

Machine learning models are increasingly being scaled in both sequence length and model dimension to reach longer contexts and better performance. However, existing architectures such as Transformers scale quadratically along both these axes. We ask: are there performant architectures that can scale sub-quadratically along sequence length and model dimension? We introduce Monarch Mixer (M2), a new architecture that uses the same sub-quadratic primitive along both sequence length and model dimension: Monarch matrices, a simple class of expressive structured matrices that captures many linear transforms, achieves high hardware efficiency on GPUs, and scales sub-quadratically. As a proof of concept, we explore the performance of M2 in three domains: non-causal BERT-style language modeling, ViT-style image classification, and causal GPT-style language modeling. For non-causal BERT-style modeling, M2 matches BERT-base and BERT-large in downstream GLUE quality with up to 27% fewer parameters, and achieves up to 9.1× higher throughput at sequence length 4K. On ImageNet, M2 outperforms ViT-b by 1% in accuracy, with only half the parameters. Causal GPT-style models introduce a technical challenge: enforcing causality via masking introduces a quadratic bottleneck. To alleviate this bottleneck, we develop a novel theoretical view of Monarch matrices based on multivariate polynomial evaluation and interpolation, which lets us parameterize M2 to be causal while remaining sub-quadratic. Using this parameterization, M2 matches GPT-style Transformers at 360M parameters in pretraining perplexity on The PILE—showing for the first time that it may be possible to match Transformer quality without attention or MLPs.

---

## 1. Introduction

Modern machine learning models in NLP and vision are being stretched to longer sequences and higher-dimensional representations to enable longer context and higher quality. However, architectures like Transformers scale quadratically in both sequence length and model dimension, limiting context and making scaling expensive. We ask: can we find performant architectures that are sub-quadratic in both sequence length and model dimension?

Monarch Mixer (M2) is a new architecture that uses Monarch matrices—a class of expressive, structured matrices that generalize FFT, Hadamard, Toeplitz, and convolutions—to mix information along both sequence and model axes. M2 achieves sub-quadratic scaling, high hardware efficiency, and strong empirical performance across language and vision tasks.

---

## 2. Monarch Matrices and Sub-Quadratic Mixing

Monarch matrices are parameterized as products of block-diagonal matrices interleaved with permutations. For order-p Monarch matrices, the computational complexity is O(pN(p+1)/p) for input length N, interpolating between O(N log N) and O(N^1.5). Monarch matrices can represent a wide range of linear transforms, including FFT, Hadamard, Toeplitz, and convolutions.

---

## 3. Monarch Mixer (M2) Architecture

M2 layers use Monarch matrices to mix information along both sequence and model axes. Each layer consists of:

- **Sequence Mixer:** Applies a Monarch matrix along the sequence axis, optionally with gating and nonlinearity.
- **Dimension Mixer:** Applies a Monarch matrix along the model dimension, replacing dense MLPs.

### M2 Layer (Pseudocode)

```python
def M2_layer(X):
    # mix sequence
    Z = M @ (k * (M @ X))
    # mix channels
    Y = M @ σ(M @ Z.T)
    return Y
```
Where `M` is a Monarch matrix, `k` is a learnable kernel, and `σ` is a nonlinearity.

---

## 4. Empirical Results

- **M2-BERT:** Matches or outperforms BERT-base and BERT-large on GLUE with up to 27% fewer parameters and up to 9.1× higher throughput at sequence length 4K.
- **M2-ViT:** Outperforms ViT-b by 1% in accuracy on ImageNet with half the parameters.
- **M2-GPT:** Matches GPT-style Transformers in pretraining perplexity on The PILE at 360M parameters, without using attention or MLPs.

---

## 5. Theoretical Analysis

Monarch matrix multiplication is interpreted as multivariate polynomial evaluation and interpolation, enabling efficient, causal, and expressive mixing. The paper provides sufficient conditions for causality and efficient implementation using GEMMs.

---

## 6. Implementation and Optimization

- M2 layers are implemented in PyTorch using less than 40 lines of code, relying on efficient matrix operations.
- The architecture is highly hardware-efficient, achieving up to 41.4% FLOP utilization on RTX 4090 GPUs for 64K input size.
- Kernel fusion and memory layout optimizations further improve performance.

---

## 7. Discussion and Conclusion

Monarch Mixer (M2) demonstrates that performant, sub-quadratic architectures are possible for both sequence and model dimension mixing, providing a path toward more efficient and scalable models in language, vision, and beyond. M2-BERT, M2-ViT, and M2-GPT serve as proof-of-concept models, matching or exceeding Transformer baselines in quality and efficiency.

---

## References

Fu, D. Y., Arora, S., Grogan, J., Johnson, I., Eyuboglu, S., Thomas, A. W., Spector, B., Poli, M., Rudra, A., & Ré, C. (2023). Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture. arXiv:2310.12109.