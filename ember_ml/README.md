# ember_ml Neural Network Library

A comprehensive library for neural networks with a focus on Liquid Time Constant (LTC) neurons, wave-based processing, attention mechanisms, and Neural Circuit Policies (NCPs).

## Architecture

```
ember_ml/
├── core/                 # Core neural implementations
│   ├── base.py          # Base neural classes
│   ├── ltc.py           # LTC neuron implementation
│   ├── geometric.py     # Geometric processing
│   ├── blocky.py        # Block-based neural processing
│   ├── hybrid.py        # Hybrid neural architectures
│   └── spherical_ltc.py # Spherical variants
│
├── attention/           # Attention mechanisms
│   ├── base.py          # Base attention classes
│   ├── temporal.py      # Time-based attention
│   ├── causal.py        # Causal attention
│   └── multiscale_ltc.py # Multiscale LTC attention
│
├── wave/                # Wave processing
│   ├── binary/          # Binary wave operations
│   ├── harmonic/        # Harmonic processing
│   ├── models/          # Wave-specific models
│   └── utils/           # Wave utilities
│
├── nn/                  # Neural network components
│   ├── wirings/         # Network connectivity patterns
│   │   ├── wiring.py    # Base wiring class
│   │   ├── ncp_wiring.py # NCP wiring implementation
│   │   └── auto_ncp.py  # Auto NCP wiring
│   ├── modules/         # Neural network modules
│   │   ├── ncp.py       # NCP implementation
│   │   └── auto_ncp.py  # Auto NCP implementation
│   └── backends/        # Backend-specific implementations
│
├── models/              # Pre-built models
│   ├── rbm.py           # Restricted Boltzmann Machine
│   ├── rbm_backend.py   # RBM backend implementations
│   └── rbm_anomaly_detector.py # RBM-based anomaly detector
│
├── backend/             # Backend implementations
│   ├── base.py          # Base backend interface
│   ├── numpy_backend.py # NumPy backend
│   ├── torch_backend.py # PyTorch backend
│   ├── mlx_backend.py   # MLX backend
│   └── torch/           # PyTorch-specific utilities
│
├── ops/                 # Neural network operations
│   ├── tensor.py        # Tensor operations
│   ├── math.py          # Mathematical operations
│   └── random.py        # Random number generation
│
├── keras_3_8/           # Keras integration
│   └── layers/          # Custom Keras layers
│
├── utils/               # Utility functions
│   ├── math_helpers.py  # Mathematical utilities
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Plotting tools
│
├── audio/               # Audio processing
│   ├── HarmonicWaveDemo.py # Harmonic wave demo
│   └── variablequantization.py # Variable quantization
│
└── wirings/             # Network connectivity utilities
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Components

### 1. LTC Neural Network
```python
from ember_ml.core.ltc import LTCNeuron

# Create an LTC neuron
neuron = LTCNeuron(tau=1.0)
output = neuron.forward(input_signal)
```

### 2. Attention Mechanisms
```python
from ember_ml.attention.temporal import TemporalAttention

# Create attention mechanism
attention = TemporalAttention(decay_rate=0.1)
weighted_output = attention.process(sequence_data)
```

### 3. Wave Processing
```python
from ember_ml.wave.binary.processor import BinaryWaveProcessor
from ember_ml.wave.harmonic.processor import HarmonicProcessor

# Process binary waves
wave_processor = BinaryWaveProcessor()
result = wave_processor.process(input_signal)

# Analyze harmonics
harmonic_processor = HarmonicProcessor()
frequencies = harmonic_processor.analyze(wave_pattern)
```

### 4. Neural Circuit Policies
```python
from ember_ml.nn.wirings import NCPWiring
from ember_ml.nn.modules import NCP

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)

# Create an NCP model
model = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True
)
```

### 5. Restricted Boltzmann Machines
```python
from ember_ml.models.rbm import RBM

# Create an RBM model
rbm = RBM(
    visible_size=784,
    hidden_size=256,
    learning_rate=0.01
)

# Train the model
rbm.train(training_data, epochs=10, batch_size=32)
```

### 6. Backend Agnosticism
```python
import ember_ml as eh

# Set the backend
eh.set_backend('torch')  # or 'numpy', 'mlx'

# Create random data using the current backend
x = eh.random_normal((32, 10))
```

## Features

1. **Core Neural Processing**
   - LTC neurons with variable time constants
   - Geometric and spherical variants
   - Hybrid architectures

2. **Attention Mechanisms**
   - Temporal attention with decay
   - Causal attention for prediction
   - Multiscale LTC attention

3. **Wave Processing**
   - Binary wave operations
   - Harmonic analysis
   - Wave-based models

4. **Neural Circuit Policies**
   - Biologically-inspired connectivity patterns
   - Interpretable and auditable neural networks
   - Custom wiring configurations

5. **Backend Agnosticism**
   - Support for NumPy, PyTorch, and MLX backends
   - Unified interface for tensor operations
   - Automatic backend selection

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Contributing
1. Follow the established module structure
2. Add comprehensive docstrings
3. Include unit tests
4. Update documentation

## Requirements

- Python 3.7+
- NumPy
- PyTorch (optional)
- MLX (optional)
- TensorFlow >= 2.8.0 (for Keras integration, optional)
- Matplotlib (for visualization)

## License

MIT