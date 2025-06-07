# Phonological Loop Neural Network

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Phonological+Loop+Neural+Network" alt="Phonological Loop Neural Network" width="600">
</p>

## Overview

The Phonological Loop Neural Network (PLNN) is a biomimetic architecture inspired by human working memory systems, specifically designed for robust signal classification in noisy environments. By combining continuous-time dynamics with a specialized memory buffer that implements decay and rehearsal mechanisms, PLNN achieves remarkable classification accuracy with extraordinary learning efficiency.

This implementation is based on the theoretical foundations of Liquid Structural State-Space Models (Liquid-S4) as described in Hasani et al., "Liquid Structural State-Space Models" (2022), which combines the generalization capabilities of liquid time-constant networks (LTCs) with the memorization, efficiency, and scalability of S4 models.

### Key Features

- **Biomimetic Design**: Inspired by the phonological loop component of human working memory
- **Analytic Features**: Uses frame-based analytic signal features (A, θ', A², AA', A²θ') inspired by Aisbett (1988).
- **Temporal Integration**: Memory buffer with decay and rehearsal mechanisms for effective temporal processing.
- **S4 Integration**: Leverages a Structured State-Space Model (S4) layer for powerful sequence modeling.
- **Rapid Learning**: Achieves good accuracy with minimal training time and data on the included dataset.
- **Computational Efficiency**: Relatively fast inference despite sophisticated processing capabilities.

## Architecture

The PLNN consists of four main components arranged in a sequential pipeline:

```
Input Waveform → Analytic Feature Extraction → Memory Processing → S4 Temporal Mixing → Classification
```

1.  **Analytic Feature Extractor**: Computes 5 frame-based analytic signal features (`A, θ', A², AA', A²θ'`) derived from the complex analytic signal, inspired by Aisbett (1988).
2.  **Phonological Loop Memory**: Maintains a buffer of recent feature frames with decay and rehearsal mechanisms.
3.  **S4 Temporal Mixer**: Processes the memory state using a Structured State Space model for enhanced temporal pattern recognition.
4.  **Classifier**: Neural network that classifies the final state output by the S4 layer.

*Note: Key components like the S4 layer and the final classifier head are initialized lazily during the first forward pass to adapt to the input data's sequence length.*

For detailed technical information, see the [Technical Architecture](docs/technical_architecture.md) document.

## Performance

On RF modulation classification tasks (AM, FM, Noise) using the included simulated dataset:

- **Accuracy**: 100% (6/6 correct) on the included test set after 20 epochs of full training using the standard S4 configuration.
- **Confidence**: High for all classes in the test set.
- **Training Time**: ~10 seconds for complete training (10 epochs noise pretraining + 20 epochs full training) on Apple Silicon MPS.
- **Convergence**: Loss decreases significantly during training, leading to perfect classification on the test data.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+ (Note: Requires a version where `torch.atan2` is implemented for your device, e.g., MPS)
- torchaudio
- numpy
- matplotlib (for visualization)
- einops
- opt_einsum

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phonological-loop-nn.git
cd phonological-loop-nn

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from phonological_loop.models.phonological_loop_classifier import PhonologicalLoopClassifier
from phonological_loop.utils.audio import load_wav_to_torch
import torch
import torch.nn.functional as F

# --- Parameters ---
# These should match the parameters used during training for best results
# See main.py for the default training configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sample_rate = 16000
target_len_samples = sample_rate * 1 # Process 1 second chunks
label_map = {0: "AM", 1: "FM", 2: "Noise"}

# --- Initialize the model ---
# Note: The model uses lazy initialization. Some components (like S4, classifier)
# are fully built during the first forward pass.
model = PhonologicalLoopClassifier(
    # Feature Extractor Params (defaults used in main.py)
    hop_length=128,
    window_length=512,
    # Memory Params (defaults used in main.py)
    buffer_len=10,
    decay_factor=0.9,
    num_recent_windows=3,
    # S4 Layer Params (defaults used in main.py)
    s4_d_model=128, # Note: This is the target dimension *after* S4 processing
    s4_d_state=64,
    s4_mode='diag',
    s4_measure='diag-lin',
    s4_dropout=0.1,
    # Classifier Params (matching main.py)
    num_classes=len(label_map),
    classifier_type='deep', # Using the deep classifier
    classifier_hidden_dims=[256, 128, 64],
    classifier_dropout=0.2,
    classifier_hidden_dim=128 # Corresponds to s4_d_model
).to(device)

# --- Load a waveform ---
# Replace with the path to your audio file
# Ensure the audio file has the correct sample rate (16000 Hz)
try:
    # Example using a generated file (replace with your own)
    waveform, sr = load_wav_to_torch("simulated_data/FM_1.wav")
    if sr != sample_rate:
        print(f"Warning: Audio sample rate ({sr} Hz) doesn't match model expected rate ({sample_rate} Hz). Resampling needed.")
        # Add resampling logic here if necessary
        # waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    # Ensure waveform is on the correct device and has the target length
    waveform = waveform.to(device)
    current_len = waveform.shape[0]
    if current_len > target_len_samples:
        waveform = waveform[:target_len_samples]
    elif current_len < target_len_samples:
        waveform = F.pad(waveform, (0, target_len_samples - current_len))

    # --- Run inference ---
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Add batch dimension and run through the model
        # This first pass also finalizes lazy initialization
        logits = model(waveform.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=-1)

    predicted_class_name = label_map.get(predicted_class_idx.item(), "Unknown")

    print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx.item()})")
    print(f"Confidence: {confidence.item():.4f}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")

except FileNotFoundError:
    print("Error: Example audio file 'simulated_data/FM_1.wav' not found.")
    print("Please generate data first using 'python -m phonological_loop.main'")
except Exception as e:
    print(f"An error occurred: {e}")

```

### Running the Demo

```bash
# Run the main demo script (generates data if needed, then trains and evaluates)
python -m phonological_loop.main

# Skip data generation if you already have data in ./simulated_data/
python -m phonological_loop.main --skip_data_gen
```

## Documentation

Comprehensive documentation is available in the [docs](docs) directory:

- [**Concept Overview**](docs/concept_overview.md): Accessible introduction to key concepts
- [**Technical Architecture**](docs/technical_architecture.md): Detailed technical description
- [**Paper Outline**](docs/paper_outline.md): Research paper outline
- [**Future Research**](docs/future_research.md): Potential research directions

## Project Structure

```
phonological_loop/
├── docs/                  # Documentation
├── features/              # Feature extraction modules
│   └── analytic_features.py # Computes 5 analytic signal features
├── models/                # Neural network models
│   ├── classifiers.py
│   ├── memory.py          # Phonological loop memory implementation
│   # ├── noise_filter.py    # Original learned filter (currently unused)
│   # ├── statistical_noise_filter.py # Aisbett-style filter (currently unused)
│   ├── phonological_loop_classifier.py # Main model orchestrator
│   ├── s4_kernel.py       # S4 kernel implementation
│   └── s4_layer.py        # S4 layer implementation
├── utils/                 # Utility functions
│   ├── audio.py             # Audio loading/generation helpers
│   ├── signal_processing.py # General signal processing utilities
│   └── training.py        # Training and evaluation loops
└── main.py                # Main script for running demos
```

## Applications

The PLNN is particularly well-suited for:

- **Communications**: RF signal classification, modulation recognition
- **Audio Processing**: Speech recognition in noisy environments
- **Biomedical**: EEG/ECG signal analysis
- **Industrial**: Machinery fault detection, predictive maintenance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [LICENSE TBD] - see the LICENSE file for details.

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

This work is based on the Liquid-S4 model:

```
@article{hasani2022liquid,
  title={Liquid Structural State-Space Models},
  author={Hasani, Ramin and Lechner, Mathias and Wang, Tsun-Hsuan and Chahine, Makram and Amini, Alexander and Rus, Daniela},
  journal={arXiv preprint arXiv:2209.12951},
  year={2022}
}
```

## Acknowledgments

- This implementation is based on the [Liquid-S4 paper](https://arxiv.org/abs/2209.12951) by Hasani et al. from MIT CSAIL
- The S4 implementation draws inspiration from the [S4 paper](https://arxiv.org/abs/2111.00396) by Gu et al.
- The concept of the phonological loop is based on the working memory model by Baddeley and Hitch.