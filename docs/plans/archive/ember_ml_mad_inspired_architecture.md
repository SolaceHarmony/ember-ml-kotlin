# Ember ML Architecture: MAD-Inspired Design

## Introduction

After exploring the Mechanistic Architecture Design (MAD) approach from [MAD-Lab](https://github.com/athms/mad-lab), we propose an updated architecture for Ember ML that incorporates these insights along with our existing concepts. This architecture aims to provide a more integrated approach that combines:

1. NCP and AutoNCP wiring concepts from control theory
2. Traditional machine learning components (Dense, Sequential, etc.)
3. Data pipeline and feature extraction functionality
4. The concept of "blocks" from striped transformers
5. MAD's categorization of neural network primitives

## Core Architectural Principles

1. **Clear Frontend/Backend Separation**: The frontend should never expose backend-specific details to users.
2. **Mechanistic Design**: Components should have clear computational roles and interactions.
3. **Modular Architecture**: Components should be easily composable and reusable.
4. **Task-Oriented Pipelines**: Higher-level components should be organized around tasks or workflows.
5. **Asynchronous Processing**: The architecture should support asynchronous processing for better performance and scalability.

## Proposed Architecture

### 1. Layer Primitives

Following MAD's approach, we categorize neural network layers into two main types:

#### Channel-Mixing Primitives

These layers mix information across feature dimensions:

```
ember_ml/nn/layers/channel_mixing/
├── mlp.py                # Multi-layer perceptron
├── dense.py              # Dense/linear layer
├── gated_linear.py       # Gated linear units (GLU, SwiGLU)
├── moe.py                # Mixture of experts
└── batch_norm.py         # Batch normalization
```

#### Sequence-Mixing Primitives

These layers mix information across sequence positions:

```
ember_ml/nn/layers/sequence_mixing/
├── attention/            # Attention mechanisms
│   ├── base.py           # Base attention class
│   ├── self_attention.py # Self-attention
│   ├── causal.py         # Causal attention
│   └── sliding_window.py # Sliding window attention
├── recurrent/            # Recurrent mechanisms
│   ├── base.py           # Base recurrent class
│   ├── lstm.py           # LSTM
│   ├── gru.py            # GRU
│   └── ltc.py            # Liquid Time-Constant
├── hyena.py              # Hyena sequence mixer
├── mamba.py              # Mamba sequence mixer
└── rwkv.py               # RWKV sequence mixer
```

### 2. Block Architecture

Blocks are higher-level components that combine layer primitives:

```
ember_ml/nn/blocks/
├── base.py               # Base block class
├── transformer.py        # Transformer block
├── mamba_block.py        # Mamba block
├── hyena_block.py        # Hyena block
├── mlp_block.py          # MLP block
└── hybrid_block.py       # Hybrid block (combining different mixers)
```

A block would be defined as:

```python
class Block:
    """Base class for all blocks."""
    
    def __init__(self, name=None, config=None):
        self.name = name
        self.config = config or {}
        self.layers = []
        
    def add_layer(self, layer):
        """Add a layer to the block."""
        self.layers.append(layer)
        return self
    
    def forward(self, inputs):
        """Forward pass through the block."""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
```

### 3. Wiring Architecture

Wirings define how blocks are connected:

```
ember_ml/nn/wirings/
├── base.py               # Base wiring class
├── sequential.py         # Sequential wiring
├── residual.py           # Residual connections
├── ncp.py                # Neural Circuit Policies
└── auto_ncp.py           # Automatic NCP
```

A wiring would be defined as:

```python
class Wiring:
    """Base class for all wirings."""
    
    def __init__(self, name=None, config=None):
        self.name = name
        self.config = config or {}
        self.blocks = []
        self.connections = []
        
    def add_block(self, block):
        """Add a block to the wiring."""
        self.blocks.append(block)
        return self
    
    def connect(self, from_block, to_block):
        """Connect two blocks."""
        self.connections.append((from_block, to_block))
        return self
    
    def forward(self, inputs):
        """Forward pass through the wiring."""
        # Implementation depends on the specific wiring
        raise NotImplementedError("Subclasses must implement forward")
```

### 4. Pipeline Architecture

Pipelines orchestrate blocks and wirings for specific tasks:

```
ember_ml/nn/pipeline/
├── base.py               # Base pipeline class
├── feature/              # Feature extraction pipelines
│   ├── base.py           # Base feature extraction pipeline
│   ├── audio.py          # Audio feature extraction
│   └── text.py           # Text feature extraction
├── training/             # Training pipelines
├── inference/            # Inference pipelines
└── async/                # Asynchronous pipelines
```

A pipeline would be defined as:

```python
class Pipeline:
    """Base class for all pipelines."""
    
    def __init__(self, name=None, config=None):
        self.name = name
        self.config = config or {}
        self.stages = []
    
    def add_stage(self, stage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def run(self, input_data):
        """Run the pipeline on the input data."""
        result = input_data
        for stage in self.stages:
            result = stage(result)
        return result
```

### 5. Configuration System

Following MAD's approach, we separate configurations from implementations:

```
ember_ml/configs/
├── layers/               # Layer configurations
│   ├── channel_mixing/   # Channel-mixing layer configs
│   └── sequence_mixing/  # Sequence-mixing layer configs
├── blocks/               # Block configurations
├── wirings/              # Wiring configurations
└── pipelines/            # Pipeline configurations
```

A configuration would be defined as a simple dictionary or a configuration class:

```python
class Config:
    """Base class for all configurations."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary."""
        return cls(**config_dict)
```

### 6. Tensor Operations

Implement the EmberTensor frontend/backend separation as outlined in the [EmberTensor Frontend/Backend Separation](ember_tensor_frontend_backend_separation.md) plan:

```
ember_ml/nn/tensor/
├── common/               # Common tensor implementations
├── interfaces/           # Tensor interfaces
└── protocols/            # Python protocol implementations
```

## Complete Directory Structure

Putting it all together, the complete directory structure would be:

```
ember_ml/
├── nn/                   # Neural network components
│   ├── tensor/           # Tensor operations and classes
│   ├── layers/           # Layer primitives
│   │   ├── channel_mixing/  # Channel-mixing layers
│   │   └── sequence_mixing/ # Sequence-mixing layers
│   ├── blocks/           # Higher-level blocks
│   ├── wirings/          # Wiring components
│   ├── pipeline/         # Pipeline components
│   ├── activations/      # Activation functions
│   └── models/           # Complete model implementations
├── data/                 # Data processing
│   ├── audio/            # Audio processing
│   ├── text/             # Text processing
│   ├── vision/           # Vision processing
│   └── features/         # Feature extraction
├── configs/              # Configuration system
└── backend/              # Backend implementations
    ├── numpy/            # NumPy backend
    ├── torch/            # PyTorch backend
    └── mlx/              # MLX backend
```

## Example Usage

### Building a Transformer Block

```python
# Create a transformer block
config = TransformerConfig(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    activation='gelu'
)
block = TransformerBlock(config=config)

# Process input
output = block(input_tensor)
```

### Creating a Pipeline

```python
# Create a feature extraction pipeline
pipeline = Pipeline("feature_extraction")
pipeline.add_stage(Preprocessor())
pipeline.add_stage(FeatureExtractor())
pipeline.add_stage(Normalizer())

# Run the pipeline
features = pipeline.run(raw_data)
```

### Using NCP Wiring

```python
# Create an NCP wiring
wiring = NCPWiring(
    num_inputs=10,
    num_outputs=5,
    num_hidden=20
)

# Add blocks
wiring.add_block(InputBlock("input"))
wiring.add_block(HiddenBlock("hidden"))
wiring.add_block(OutputBlock("output"))

# Connect blocks according to NCP rules
wiring.connect_ncp()

# Process input
output = wiring(input_tensor)
```

## Implementation Strategy

1. **Phase 1**: Implement the EmberTensor frontend/backend separation
2. **Phase 2**: Implement the layer primitives (channel-mixing and sequence-mixing)
3. **Phase 3**: Implement the block architecture
4. **Phase 4**: Implement the wiring architecture
5. **Phase 5**: Implement the pipeline architecture
6. **Phase 6**: Implement the configuration system

## Migration Path

To minimize disruption, we'll implement these changes incrementally:

1. First, create the new directory structure
2. Implement the EmberTensor frontend/backend separation
3. Migrate existing components to the new architecture
4. Develop new components using the new architecture

## Conclusion

This MAD-inspired architecture for Ember ML provides a more integrated approach that combines concepts from control theory, machine learning, and data processing. By categorizing neural network components into channel-mixing and sequence-mixing primitives, and organizing them into blocks, wirings, and pipelines, we create a flexible and powerful architecture that can handle a wide range of machine learning tasks.

The mechanistic approach ensures that each component has a clear computational role and that interactions between components are well-defined. This makes the architecture more interpretable, maintainable, and extensible.