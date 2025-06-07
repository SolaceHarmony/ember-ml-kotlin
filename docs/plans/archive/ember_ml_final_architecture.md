# Ember ML Final Architecture Plan

## Introduction

After exploring the [MAD-Lab](https://github.com/athms/mad-lab), [xLSTM](https://github.com/NX-AI/xlstm), and [Striped Hyena](https://github.com/togethercomputer/stripedhyena) repositories, we propose a comprehensive architecture for Ember ML that incorporates the best aspects of all three approaches. This architecture aims to provide a flexible, modular, and efficient framework for building neural networks.

## Core Architectural Principles

1. **Clear Frontend/Backend Separation**: The frontend should never expose backend-specific details to users.
2. **Mechanistic Design**: Components should have clear computational roles and interactions.
3. **Block-Based Architecture**: Higher-level components should be organized as blocks that combine multiple layers.
4. **Configuration-Driven Design**: Components should be configurable through dedicated configuration classes.
5. **Residual Connections**: Use residual connections to help with gradient flow in deep networks.
6. **Distributed Training Support**: Support for model and data parallelism.

## Proposed Architecture

### 1. Layer Primitives

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

### 2. Block Architecture

Following xLSTM and Striped Hyena, we define blocks as higher-level components that combine multiple layers:

```
ember_ml/nn/blocks/
├── base.py              # Base block class
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
    
    config_class = None  # Subclasses should specify their config class
    
    def __init__(self, config):
        """Initialize a block with a configuration."""
        super().__init__()
        self.config = config
        self.norm = None  # Normalization layer
        self.main_layer = None  # Main processing layer
        self.ffn = None  # Optional feed-forward network
        self.ffn_norm = None  # Normalization for FFN
    
    def forward(self, x, **kwargs):
        """Forward pass through the block."""
        # Apply normalization
        normalized_x = self.norm(x)
        
        # Apply main layer
        output = self.main_layer(normalized_x, **kwargs)
        
        # Add residual connection
        x = x + output
        
        # Apply FFN if present
        if self.ffn is not None:
            # Apply normalization
            normalized_x = self.ffn_norm(x)
            
            # Apply FFN
            output = self.ffn(normalized_x)
            
            # Add residual connection
            x = x + output
        
        return x
```

### 3. Configuration System

Following all three repositories, we use a configuration-driven design:

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
    
    embedding_dim: int
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0
    
    # Optional feed-forward network configuration
    ffn: Optional[FeedForwardConfig] = None
```

### 4. Model Architecture

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
    
    config_class = None  # Subclasses should specify their config class
    
    def __init__(self, config):
        """Initialize a model with a configuration."""
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        
        # Create blocks based on configuration
        for i in range(config.num_blocks):
            block_config = self._create_block_config(i)
            self.blocks.append(self._create_block(block_config))
    
    def _create_block_config(self, block_idx):
        """Create a configuration for a block."""
        raise NotImplementedError("Subclasses must implement _create_block_config")
    
    def _create_block(self, block_config):
        """Create a block from a configuration."""
        raise NotImplementedError("Subclasses must implement _create_block")
    
    def forward(self, x, **kwargs):
        """Forward pass through the model."""
        for block in self.blocks:
            x = block(x, **kwargs)
        return x
```

### 5. Tensor Operations

Implement the EmberTensor frontend/backend separation as outlined in the [EmberTensor Frontend/Backend Separation](ember_tensor_frontend_backend_separation.md) plan:

```
ember_ml/nn/tensor/
├── common/              # Common tensor implementations
├── interfaces/          # Tensor interfaces
└── protocols/           # Python protocol implementations
```

### 6. Pipeline Architecture

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

### 7. Distributed Training Support

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

## Example Usage

### Building a Transformer Block

```python
# Create a transformer block configuration
config = TransformerBlockConfig(
    embedding_dim=512,
    num_heads=8,
    dropout=0.1,
    ffn=FeedForwardConfig(
        hidden_dim=2048,
        activation="gelu"
    )
)

# Create a transformer block
block = TransformerBlock(config)

# Process input
output = block(input_tensor)
```

### Creating a Model

```python
# Create a model configuration
config = TransformerModelConfig(
    embedding_dim=512,
    num_blocks=6,
    num_heads=8,
    dropout=0.1,
    ffn=FeedForwardConfig(
        hidden_dim=2048,
        activation="gelu"
    )
)

# Create a model
model = TransformerModel(config)

# Process input
output = model(input_tensor)
```

### Creating a Pipeline

```python
# Create a pipeline configuration
config = PipelineConfig(
    model=TransformerModelConfig(...),
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

1. **Phase 1**: Implement the EmberTensor frontend/backend separation
2. **Phase 2**: Implement the layer primitives (channel-mixing and sequence-mixing)
3. **Phase 3**: Implement the block architecture
4. **Phase 4**: Implement the model architecture
5. **Phase 5**: Implement the pipeline architecture
6. **Phase 6**: Implement the configuration system
7. **Phase 7**: Implement distributed training support

## Migration Path

To minimize disruption, we'll implement these changes incrementally:

1. First, create the new directory structure
2. Implement the EmberTensor frontend/backend separation
3. Migrate existing components to the new architecture
4. Develop new components using the new architecture

## Conclusion

This final architecture for Ember ML incorporates the best aspects of MAD-Lab, xLSTM, and Striped Hyena. By categorizing neural network components into channel-mixing and sequence-mixing primitives, organizing them into blocks, using a configuration-driven design, and supporting distributed training, we create a flexible and powerful architecture that can handle a wide range of machine learning tasks.

The block-based approach with residual connections ensures good gradient flow in deep networks, while the clean separation between layers, blocks, and models makes the architecture more interpretable, maintainable, and extensible.