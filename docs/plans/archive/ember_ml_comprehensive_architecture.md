# Ember ML Comprehensive Architecture Plan

## Introduction

After exploring the [MAD-Lab](https://github.com/athms/mad-lab), [xLSTM](https://github.com/NX-AI/xlstm), [Striped Hyena](https://github.com/togethercomputer/stripedhyena), and [hyena-dna](https://github.com/HazyResearch/hyena-dna) repositories, we propose a comprehensive architecture for Ember ML that incorporates the best aspects of all four approaches. This architecture aims to provide a flexible, modular, and efficient framework for building neural networks.

## Core Architectural Principles

1. **Clear Frontend/Backend Separation**: The frontend should never expose backend-specific details to users.
2. **Mechanistic Design**: Components should have clear computational roles and interactions.
3. **Block-Based Architecture**: Higher-level components should be organized as blocks that combine multiple layers.
4. **Configuration-Driven Design**: Components should be configurable through dedicated configuration classes.
5. **Residual Connections**: Use residual connections to help with gradient flow in deep networks.
6. **Distributed Training Support**: Support for model and data parallelism.
7. **Registry System**: Use a registry system for instantiating components, allowing for flexible configuration.
8. **Sequential Processing**: Support both batch processing and sequential processing for recurrent models and autoregressive generation.

## Proposed Architecture

### 1. Registry System

Following hyena-dna, we implement a registry system for instantiating components:

```
ember_ml/registry/
├── base.py              # Base registry class
├── layers.py            # Layer registry
├── blocks.py            # Block registry
├── models.py            # Model registry
└── utils.py             # Registry utilities
```

A registry would be defined as:

```python
class Registry:
    """Base class for all registries."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name):
        """Register a component with the given name."""
        def decorator(component):
            cls._registry[name] = component
            return component
        return decorator
    
    @classmethod
    def get(cls, name):
        """Get a component by name."""
        if name not in cls._registry:
            raise ValueError(f"Component {name} not found in registry")
        return cls._registry[name]
    
    @classmethod
    def instantiate(cls, name, *args, **kwargs):
        """Instantiate a component by name."""
        component = cls.get(name)
        return component(*args, **kwargs)
```

### 2. Layer Primitives

Following MAD's approach, we categorize neural network layers into two main types:

#### Channel-Mixing Primitives

These layers mix information across feature dimensions:

```
ember_ml/nn/layers/channel_mixing/
├── base.py              # Base channel-mixing class
├── mlp.py               # Multi-layer perceptron
├── gated_mlp.py         # Gated MLP (like in Striped Hyena)
├── dense.py             # Dense/linear layer
├── gated_linear.py      # Gated linear units (GLU, SwiGLU)
├── moe.py               # Mixture of experts
└── norm.py              # Normalization layers (RMSNorm, LayerNorm)
```

#### Sequence-Mixing Primitives

These layers mix information across sequence positions:

```
ember_ml/nn/layers/sequence_mixing/
├── base.py              # Base sequence-mixing class
├── attention/           # Attention mechanisms
│   ├── base.py          # Base attention class
│   ├── self_attention.py # Self-attention
│   ├── causal.py        # Causal attention
│   └── linear.py        # Linear attention
├── recurrent/           # Recurrent mechanisms
│   ├── base.py          # Base recurrent class
│   ├── lstm.py          # LSTM
│   ├── gru.py           # GRU
│   └── ltc.py           # Liquid Time-Constant
├── hyena.py             # Hyena sequence mixer
├── mamba.py             # Mamba sequence mixer
└── rwkv.py              # RWKV sequence mixer
```

### 3. Block Architecture

Following xLSTM and hyena-dna, we define blocks as higher-level components that combine multiple layers:

```
ember_ml/nn/blocks/
├── base.py              # Base block class
├── residual.py          # Residual block (like in hyena-dna)
├── transformer.py       # Transformer block
├── lstm.py              # LSTM block
├── mamba_block.py       # Mamba block
├── hyena_block.py       # Hyena block
└── hybrid_block.py      # Hybrid block (combining different mixers)
```

A block would be defined as:

```python
class Block(nn.Module):
    """Base class for all blocks."""
    
    def __init__(self, d_input, layer=None, residual=None, norm=None, pool=None, prenorm=True, dropout=0.0):
        """Initialize a block with a configuration."""
        super().__init__()
        self.d_input = d_input
        self.prenorm = prenorm
        
        # Instantiate components from registry
        self.layer = registry.instantiate(layer, d_input) if layer else None
        self.residual = registry.instantiate(residual, d_input) if residual else None
        self.norm = registry.instantiate(norm, d_input) if norm else None
        self.pool = registry.instantiate(pool, d_input) if pool else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, state=None, **kwargs):
        """Forward pass through the block."""
        y = x
        
        # Pre-norm
        if self.norm and self.prenorm:
            y = self.norm(y)
        
        # Main layer
        if self.layer:
            y, state = self.layer(y, state=state, **kwargs)
        
        # Residual
        if self.residual:
            y = self.residual(x, y)
        else:
            y = x + y
        
        # Post-norm
        if self.norm and not self.prenorm:
            y = self.norm(y)
        
        # Pool
        if self.pool:
            y = self.pool(y)
        
        return y, state
    
    def step(self, x, state=None, **kwargs):
        """Step-by-step processing for sequential inputs."""
        y = x
        
        # Pre-norm
        if self.norm and self.prenorm:
            y = self.norm.step(y)
        
        # Main layer
        if self.layer:
            y, state = self.layer.step(y, state=state, **kwargs)
        
        # Residual
        if self.residual:
            y = self.residual(x, y)
        else:
            y = x + y
        
        # Post-norm
        if self.norm and not self.prenorm:
            y = self.norm.step(y)
        
        # Pool
        if self.pool:
            y = self.pool(y)
        
        return y, state
```

### 4. Configuration System

Following all four repositories, we use a configuration-driven design:

```
ember_ml/nn/configs/
├── base.py              # Base configuration classes
├── layers/              # Layer configurations
│   ├── channel_mixing/  # Channel-mixing layer configs
│   └── sequence_mixing/ # Sequence-mixing layer configs
├── blocks/              # Block configurations
└── models/              # Model configurations
```

A configuration would be defined as a data class:

```python
@dataclass
class BlockConfig:
    """Base configuration for all blocks."""
    
    d_input: int
    layer: Optional[str] = None
    residual: Optional[str] = None
    norm: Optional[str] = None
    pool: Optional[str] = None
    prenorm: bool = True
    dropout: float = 0.0
```

### 5. Model Architecture

Models combine multiple blocks:

```
ember_ml/nn/models/
├── base.py              # Base model class
├── transformer.py       # Transformer model
├── lstm.py              # LSTM model
├── mamba.py             # Mamba model
├── hyena.py             # Hyena model
└── hybrid.py            # Hybrid model
```

A model would be defined as:

```python
class Model(nn.Module):
    """Base class for all models."""
    
    def __init__(self, config):
        """Initialize a model with a configuration."""
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        
        # Create blocks based on configuration
        for block_config in config.blocks:
            self.blocks.append(Block(**block_config))
    
    def forward(self, x, state=None, **kwargs):
        """Forward pass through the model."""
        states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            x, block_state = block(x, state=block_state, **kwargs)
            states.append(block_state)
        return x, states
    
    def step(self, x, state=None, **kwargs):
        """Step-by-step processing for sequential inputs."""
        states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            x, block_state = block.step(x, state=block_state, **kwargs)
            states.append(block_state)
        return x, states
```

### 6. Tensor Operations

Implement the EmberTensor frontend/backend separation as outlined in the [EmberTensor Frontend/Backend Separation](ember_tensor_frontend_backend_separation.md) plan:

```
ember_ml/nn/tensor/
├── common/              # Common tensor implementations
├── interfaces/          # Tensor interfaces
└── protocols/           # Python protocol implementations
```

### 7. Pipeline Architecture

Pipelines orchestrate models for specific tasks:

```
ember_ml/nn/pipeline/
├── base.py              # Base pipeline class
├── feature/             # Feature extraction pipelines
├── training/            # Training pipelines
└── inference/           # Inference pipelines
```

A pipeline would be defined as:

```python
class Pipeline:
    """Base class for all pipelines."""
    
    def __init__(self, config):
        """Initialize a pipeline with a configuration."""
        self.config = config
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

### 8. Distributed Training Support

Following Striped Hyena, we add support for distributed training:

```
ember_ml/nn/distributed/
├── base.py              # Base distributed classes
├── model_parallel.py    # Model parallelism utilities
├── data_parallel.py     # Data parallelism utilities
└── mixed_parallel.py    # Mixed parallelism utilities
```

## Complete Directory Structure

Putting it all together, the complete directory structure would be:

```
ember_ml/
├── registry/            # Component registry
│   ├── base.py          # Base registry class
│   ├── layers.py        # Layer registry
│   ├── blocks.py        # Block registry
│   ├── models.py        # Model registry
│   └── utils.py         # Registry utilities
├── nn/                  # Neural network components
│   ├── tensor/          # Tensor operations and classes
│   ├── layers/          # Layer primitives
│   │   ├── channel_mixing/ # Channel-mixing layers
│   │   └── sequence_mixing/ # Sequence-mixing layers
│   ├── blocks/          # Higher-level blocks
│   ├── models/          # Complete model implementations
│   ├── configs/         # Configuration classes
│   ├── pipeline/        # Pipeline components
│   └── distributed/     # Distributed training utilities
├── data/                # Data processing
│   ├── audio/           # Audio processing
│   ├── text/            # Text processing
│   ├── vision/          # Vision processing
│   └── features/        # Feature extraction
└── backend/             # Backend implementations
    ├── numpy/           # NumPy backend
    ├── torch/           # PyTorch backend
    └── mlx/             # MLX backend
```

## Key Features from Each Repository

### From MAD-Lab

1. **Channel-Mixing vs. Sequence-Mixing**: Clear categorization of layers based on their computational role.
2. **Clean Separation**: Separation between layer implementations, operations, and configurations.
3. **Wrapper Approach**: Layers wrap specific implementations for flexibility.

### From xLSTM

1. **Block-Based Architecture**: Higher-level blocks that combine multiple layers.
2. **Configuration Classes**: Dedicated configuration classes for each component.
3. **Residual Connections**: Residual connections for better gradient flow.

### From Striped Hyena

1. **Parallel Implementations**: Support for model and data parallelism.
2. **Gated Architectures**: Gated MLP and other gated architectures.
3. **Efficient Implementations**: Focus on efficiency and performance.

### From hyena-dna

1. **Registry System**: Registry for instantiating components, allowing for flexible configuration.
2. **Sequential Processing**: Support for both batch processing and sequential processing.
3. **Configurable Block Structure**: Configurable options for normalization position, normalization type, residual connections, and pooling.
4. **Black Box Approach**: Core layer treated as a pluggable component.

## Example Usage

### Building a Block

```python
# Create a block configuration
config = BlockConfig(
    d_input=512,
    layer="hyena",
    residual="identity",
    norm="layer_norm",
    prenorm=True,
    dropout=0.1
)

# Create a block
block = Block(**config)

# Process input
output, state = block(input_tensor)

# Step-by-step processing
output, state = block.step(input_tensor, state=state)
```

### Creating a Model

```python
# Create a model configuration
config = ModelConfig(
    blocks=[
        BlockConfig(d_input=512, layer="hyena", norm="layer_norm"),
        BlockConfig(d_input=512, layer="hyena", norm="layer_norm"),
        BlockConfig(d_input=512, layer="hyena", norm="layer_norm"),
    ]
)

# Create a model
model = Model(config)

# Process input
output, states = model(input_tensor)

# Step-by-step processing
output, states = model.step(input_tensor, state=states)
```

### Creating a Pipeline

```python
# Create a pipeline configuration
config = PipelineConfig(
    model=ModelConfig(...),
    tokenizer=TokenizerConfig(...),
    optimizer=OptimizerConfig(...)
)

# Create a pipeline
pipeline = Pipeline(config)
pipeline.add_stage(Preprocessor())
pipeline.add_stage(Model())
pipeline.add_stage(Postprocessor())

# Run the pipeline
output = pipeline.run(input_data)
```

## Implementation Strategy

1. **Phase 1**: Implement the registry system
2. **Phase 2**: Implement the EmberTensor frontend/backend separation
3. **Phase 3**: Implement the layer primitives (channel-mixing and sequence-mixing)
4. **Phase 4**: Implement the block architecture
5. **Phase 5**: Implement the model architecture
6. **Phase 6**: Implement the pipeline architecture
7. **Phase 7**: Implement the configuration system
8. **Phase 8**: Implement distributed training support

## Migration Path

To minimize disruption, we'll implement these changes incrementally:

1. First, create the new directory structure
2. Implement the registry system
3. Implement the EmberTensor frontend/backend separation
4. Migrate existing components to the new architecture
5. Develop new components using the new architecture

## Conclusion

This comprehensive architecture for Ember ML incorporates the best aspects of MAD-Lab, xLSTM, Striped Hyena, and hyena-dna. By categorizing neural network components into channel-mixing and sequence-mixing primitives, organizing them into blocks, using a registry system for flexible configuration, supporting both batch and sequential processing, and enabling distributed training, we create a flexible and powerful architecture that can handle a wide range of machine learning tasks.

The block-based approach with residual connections ensures good gradient flow in deep networks, while the clean separation between layers, blocks, and models makes the architecture more interpretable, maintainable, and extensible.