# Phonological Loop Neural Network Documentation

This directory contains documentation for the Phonological Loop Neural Network (PLNN) project, a biomimetic architecture inspired by human working memory systems for robust signal classification in noisy environments. The implementation is based on the theoretical foundations of Liquid Structural State-Space Models (Liquid-S4) as described in Hasani et al., "Liquid Structural State-Space Models" (2022).

## Documentation Overview

### For General Audiences
- [**Concept Overview**](concept_overview.md): An accessible introduction to the key concepts, innovations, and potential applications of the Phonological Loop Neural Network.

### For Technical Audiences
- [**Technical Architecture**](technical_architecture.md): Detailed technical description of the architecture, implementation considerations, and computational flow.
- [**Paper Outline**](paper_outline.md): Comprehensive outline for a research paper documenting the architecture and its results.
- [**Future Research**](future_research.md): Exploration of potential research directions and extensions for the Phonological Loop Neural Network.
- [**PLNN vs. Traditional Deep Learning**](plnn_vs_traditional.md): Comparison between the Phonological Loop approach and traditional deep learning methods like CNNs.
- [**Liquid-S4 Paper Summary**](liquid_s4_paper_summary.md): Summary of the key concepts, findings, and hyperparameters from the Liquid-S4 paper that forms the theoretical foundation for this implementation.

## Additional Resources

### Code Documentation
The implementation of the Phonological Loop Neural Network is organized into several key modules:

- `phonological_loop/models/memory.py`: Implementation of the Phonological Loop Memory component
- `phonological_loop/models/noise_filter.py`: Implementation of the Log-Domain Noise Filter
- `phonological_loop/models/s4_layer.py`: Implementation of the Structured State Space Sequence (S4) layer
- `phonological_loop/models/phonological_loop_classifier.py`: Main model implementation combining all components
- `phonological_loop/features/aisbett_features.py`: Implementation of the Aisbett Feature Extractor
- `phonological_loop/utils/training.py`: Training utilities including the two-phase training approach

### Experimental Results

The Phonological Loop Neural Network has demonstrated remarkable performance on RF signal classification tasks:

- **Accuracy**: 100% on test set
- **Confidence**: 
  - AM signals: 0.57-0.89
  - FM signals: 0.94-0.98
  - Noise: 0.99-1.00
- **Training Time**: ~3 seconds for complete training
- **Convergence**: Significant learning by epoch 3-4

## Contributing

We welcome contributions to both the code and documentation. If you're interested in contributing:

1. For code contributions, please review the technical architecture document first
2. For documentation contributions, please maintain the existing style and organization
3. Submit pull requests with clear descriptions of changes and their rationale

## Citation

If you use this work in your research, please cite:

```
@article{phonological_loop_nn,
  title={Phonological Loop Neural Architecture: A Biomimetic Approach to Rapid Signal Classification in Noisy Environments},
  author={Bach, Sydney},
  journal={TBD},
  year={2025},
  publisher={TBD}
}
```

## License

This project is licensed under [LICENSE TBD] - see the LICENSE file for details.