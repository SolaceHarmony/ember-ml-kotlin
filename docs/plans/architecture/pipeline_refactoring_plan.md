# Pipeline Refactoring Plan

This document outlines the plan for refactoring the existing pipeline implementation to use the ember_ml Module system, NCP, and other ember_ml components.

## Current Architecture

The current pipeline implementation in `tests/pipeline/pipeline_demo.py` consists of:

1. **Feature Extraction**: Using `TerabyteFeatureExtractor` and `TerabyteTemporalStrideProcessor`
2. **Feature Learning**: Using `OptimizedRBM`
3. **Neural Network Processing**: Using various liquid neural network implementations from `ember_ml.core.stride_aware_cfc`
4. **Motor Neuron Output**: For triggering deeper exploration

The implementation uses a mix of approaches, including direct TensorFlow usage, custom classes, and some ember_ml components.

## Refactoring Goals

1. **Modularize the Pipeline**: Split into distinct Module components
2. **Use ember_ml Module System**: Replace direct TensorFlow usage with ember_ml Modules
3. **Implement NCP-based Networks**: Replace custom implementations with NCP-based ones
4. **Separate Training Logic**: Create dedicated training scripts
5. **Backend Agnosticism**: Ensure the pipeline works with any backend (NumPy, PyTorch, MLX)

## New Architecture

### 1. Module-Based Components

#### Feature Extraction Module

```python
class FeatureExtractionModule(Module):
    """Feature extraction module using the ember_ml Module system."""
    
    def __init__(self, chunk_size=100000, max_memory_gb=16.0, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        # Additional initialization
        
    def forward(self, data, **kwargs):
        # Feature extraction logic
        return extracted_features
```

#### Temporal Processing Module

```python
class TemporalProcessingModule(Module):
    """Temporal processing module using the ember_ml Module system."""
    
    def __init__(self, window_size=10, stride_perspectives=[1, 3, 5], **kwargs):
        super().__init__()
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives
        # Additional initialization
        
    def forward(self, data, **kwargs):
        # Temporal processing logic
        return processed_features
```

#### RBM Module

```python
class RBMModule(Module):
    """RBM module using the ember_ml Module system."""
    
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, **kwargs):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Initialize weights and biases as Parameters
        self.weights = Parameter(ops.random_normal((n_visible, n_hidden), scale=0.01))
        self.visible_bias = Parameter(ops.zeros(n_visible))
        self.hidden_bias = Parameter(ops.zeros(n_hidden))
        
    def forward(self, visible_states, **kwargs):
        # Forward pass (transform)
        hidden_probs = self.compute_hidden_probabilities(visible_states)
        return hidden_probs
        
    def compute_hidden_probabilities(self, visible_states):
        # Compute hidden activations
        hidden_activations = ops.matmul(visible_states, self.weights) + self.hidden_bias
        return ops.sigmoid(hidden_activations)
        
    def reconstruct(self, visible_states):
        # Reconstruct visible states
        hidden_probs = self.compute_hidden_probabilities(visible_states)
        hidden_states = self.sample_hidden_states(hidden_probs)
        visible_probs = self.compute_visible_probabilities(hidden_states)
        return visible_probs
        
    # Additional methods for training, etc.
```

#### Liquid Network Module

```python
class LiquidNetworkModule(Module):
    """Liquid network module using the ember_ml Module system and NCP."""
    
    def __init__(self, input_dim, units=128, output_dim=1, **kwargs):
        super().__init__()
        
        # Create NCP wiring
        self.wiring = AutoNCP(
            units=units,
            output_size=output_dim,
            sparsity_level=kwargs.get('sparsity_level', 0.5)
        )
        
        # Create NCP cell
        self.ncp_cell = NCP(
            wiring=self.wiring,
            in_features=input_dim
        )
        
        # Input projection
        self.input_projection = ops.nn.Dense(
            units=input_dim,
            input_shape=(input_dim,)
        )
        
        # Output projection
        self.output_projection = ops.nn.Dense(
            units=output_dim,
            input_shape=(self.ncp_cell.output_size,)
        )
        
        # Motor neuron
        self.motor_neuron = MotorNeuronModule(
            input_dim=self.ncp_cell.output_size,
            threshold=kwargs.get('threshold', 0.5),
            adaptive_threshold=kwargs.get('adaptive_threshold', True)
        )
        
    def forward(self, inputs, states=None, **kwargs):
        # Process inputs through the liquid network
        batch_size = ops.shape(inputs)[0]
        
        if states is None:
            states = self.ncp_cell.get_initial_state(batch_size)
            
        # Process sequence
        outputs = []
        trigger_signals = []
        threshold_values = []
        
        for t in range(ops.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Project input
            projected_input = self.input_projection(x_t)
            
            # Process through NCP cell
            cell_output, states = self.ncp_cell(projected_input, states)
            
            # Generate motor neuron output and trigger
            motor_output, trigger, threshold = self.motor_neuron(cell_output)
            
            # Project output
            projected_output = self.output_projection(cell_output)
            
            # Store outputs
            outputs.append(projected_output)
            trigger_signals.append(trigger)
            threshold_values.append(threshold)
            
        # Stack outputs
        outputs = ops.stack(outputs, axis=1)
        trigger_signals = ops.stack(trigger_signals, axis=1)
        threshold_values = ops.stack(threshold_values, axis=1)
        
        return outputs, [trigger_signals, threshold_values]
```

#### Motor Neuron Module

```python
class MotorNeuronModule(Module):
    """Motor neuron module using the ember_ml Module system."""
    
    def __init__(self, input_dim, threshold=0.5, adaptive_threshold=True):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        
        # Output projection
        self.output_projection = ops.nn.Dense(
            units=1,
            input_shape=(input_dim,),
            activation="sigmoid"
        )
        
        # Adaptive threshold
        if adaptive_threshold:
            self.threshold_projection = ops.nn.Dense(
                units=1,
                input_shape=(input_dim,),
                activation="sigmoid",
                bias_initializer=ops.initializers.Constant(threshold)
            )
            
    def forward(self, inputs, **kwargs):
        # Generate output
        output = self.output_projection(inputs)
        
        # Generate threshold
        if self.adaptive_threshold:
            threshold = self.threshold_projection(inputs)
        else:
            threshold = ops.full_like(output, self.threshold)
            
        # Generate trigger
        trigger = ops.cast(output > threshold, ops.float32)
        
        return output, trigger, threshold
```

### 2. Pipeline Module

```python
class PipelineModule(Module):
    """Complete pipeline module using the ember_ml Module system."""
    
    def __init__(self, feature_dim, rbm_hidden_units=64, cfc_units=128, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.rbm_hidden_units = rbm_hidden_units
        self.cfc_units = cfc_units
        
        # Initialize components
        self.feature_extractor = None  # Will be initialized separately
        self.temporal_processor = None  # Will be initialized separately
        
        # Initialize RBM
        self.rbm = RBMModule(
            n_visible=feature_dim,
            n_hidden=rbm_hidden_units,
            learning_rate=kwargs.get('learning_rate', 0.01),
            momentum=kwargs.get('momentum', 0.5),
            weight_decay=kwargs.get('weight_decay', 0.0001)
        )
        
        # Initialize liquid network
        self.liquid_network = LiquidNetworkModule(
            input_dim=rbm_hidden_units,
            units=cfc_units,
            output_dim=kwargs.get('output_dim', 1),
            sparsity_level=kwargs.get('sparsity_level', 0.5),
            threshold=kwargs.get('threshold', 0.5),
            adaptive_threshold=kwargs.get('adaptive_threshold', True)
        )
        
    def forward(self, features, **kwargs):
        # Extract RBM features
        rbm_features = self.rbm(features)
        
        # Reshape for sequence input if needed
        if len(rbm_features.shape) == 2:
            rbm_features = rbm_features.reshape(rbm_features.shape[0], 1, rbm_features.shape[1])
            
        # Process through liquid network
        outputs, [trigger_signals, threshold_values] = self.liquid_network(rbm_features)
        
        if kwargs.get('return_triggers', True):
            return outputs, trigger_signals
        else:
            return outputs
```

### 3. Training Scripts

#### RBM Training Script

```python
def train_rbm(rbm_module, data_generator, epochs=10, batch_size=100, **kwargs):
    """Train the RBM module."""
    optimizer = ops.optimizers.SGD(
        learning_rate=rbm_module.learning_rate,
        momentum=rbm_module.momentum
    )
    
    training_errors = []
    
    for epoch in range(epochs):
        epoch_error = 0
        n_batches = 0
        
        for batch_data in data_generator:
            # Skip empty batches
            if len(batch_data) == 0:
                continue
                
            # Perform contrastive divergence
            with ops.GradientTape() as tape:
                # Positive phase
                pos_hidden_probs = rbm_module.compute_hidden_probabilities(batch_data)
                pos_hidden_states = rbm_module.sample_hidden_states(pos_hidden_probs)
                pos_associations = ops.matmul(ops.transpose(batch_data), pos_hidden_probs)
                
                # Negative phase
                neg_hidden_states = pos_hidden_states
                for _ in range(kwargs.get('k', 1)):
                    neg_visible_probs = rbm_module.compute_visible_probabilities(neg_hidden_states)
                    neg_visible_states = rbm_module.sample_visible_states(neg_visible_probs)
                    neg_hidden_probs = rbm_module.compute_hidden_probabilities(neg_visible_states)
                    neg_hidden_states = rbm_module.sample_hidden_states(neg_hidden_probs)
                    
                neg_associations = ops.matmul(ops.transpose(neg_visible_states), neg_hidden_probs)
                
                # Compute gradients
                batch_size = len(batch_data)
                weights_gradient = (pos_associations - neg_associations) / batch_size
                visible_bias_gradient = ops.mean(batch_data - neg_visible_states, axis=0)
                hidden_bias_gradient = ops.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
                
                # Compute reconstruction error
                reconstruction_error = ops.mean(ops.sum((batch_data - neg_visible_probs) ** 2, axis=1))
                
            # Apply gradients
            gradients = [weights_gradient, visible_bias_gradient, hidden_bias_gradient]
            optimizer.apply_gradients(zip(gradients, [rbm_module.weights, rbm_module.visible_bias, rbm_module.hidden_bias]))
            
            epoch_error += reconstruction_error
            n_batches += 1
            
        # Compute average epoch error
        avg_epoch_error = epoch_error / max(n_batches, 1)
        training_errors.append(avg_epoch_error)
        
    return training_errors
```

#### Liquid Network Training Script

```python
def train_liquid_network(liquid_network, features, targets, validation_data=None, epochs=100, batch_size=32, **kwargs):
    """Train the liquid network module."""
    optimizer = ops.optimizers.Adam(learning_rate=kwargs.get('learning_rate', 0.001))
    
    # Reshape features for sequence input if needed
    if len(features.shape) == 2:
        features = features.reshape(features.shape[0], 1, features.shape[1])
        
    # Reshape validation data if provided
    if validation_data is not None:
        val_features, val_targets = validation_data
        if len(val_features.shape) == 2:
            val_features = val_features.reshape(val_features.shape[0], 1, val_features.shape[1])
        validation_data = (val_features, val_targets)
        
    # Training loop
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Shuffle data
        indices = ops.random.permutation(len(features))
        features_shuffled = ops.gather(features, indices)
        targets_shuffled = ops.gather(targets, indices)
        
        # Train on batches
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(features), batch_size):
            batch_features = features_shuffled[i:i+batch_size]
            batch_targets = targets_shuffled[i:i+batch_size]
            
            with ops.GradientTape() as tape:
                # Forward pass
                outputs, _ = liquid_network(batch_features)
                
                # Compute loss
                loss = ops.mean(ops.square(outputs - batch_targets))
                
            # Compute gradients
            gradients = tape.gradient(loss, liquid_network.parameters())
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, liquid_network.parameters()))
            
            epoch_loss += loss
            n_batches += 1
            
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_epoch_loss)
        
        # Validation
        if validation_data is not None:
            val_features, val_targets = validation_data
            val_outputs, _ = liquid_network(val_features)
            val_loss = ops.mean(ops.square(val_outputs - val_targets))
            history["val_loss"].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= kwargs.get('patience', 10):
                    break
        
    return history
```

### 4. Feature Extraction Modules

#### Base Feature Extractor

```python
class BaseFeatureExtractor(Module):
    """Base feature extractor module."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, data, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
```

#### Terabyte Feature Extractor

```python
class TerabyteFeatureExtractorModule(BaseFeatureExtractor):
    """Terabyte feature extractor module."""
    
    def __init__(self, chunk_size=100000, max_memory_gb=16.0, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        # Additional initialization
        
    def forward(self, data, **kwargs):
        # Feature extraction logic
        return extracted_features
```

#### Temporal Stride Processor

```python
class TemporalStrideProcessorModule(BaseFeatureExtractor):
    """Temporal stride processor module."""
    
    def __init__(self, window_size=10, stride_perspectives=[1, 3, 5], **kwargs):
        super().__init__()
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives
        # Additional initialization
        
    def forward(self, data, **kwargs):
        # Temporal processing logic
        return processed_features
```

## Implementation Plan

1. **Create Base Modules**:
   - Implement the base Module classes for each component
   - Ensure they follow the ember_ml Module system design

2. **Implement RBM Module**:
   - Refactor the OptimizedRBM to use the Module system
   - Ensure backend-agnostic implementation using ops

3. **Implement Liquid Network Module**:
   - Use NCP and other ember_ml components
   - Replace TensorFlow-specific code with ops

4. **Create Training Scripts**:
   - Separate training logic from model definition
   - Implement backend-agnostic training procedures

5. **Implement Feature Extraction Modules**:
   - Refactor feature extraction to use the Module system
   - Ensure compatibility with the pipeline

6. **Create Pipeline Module**:
   - Integrate all components into a unified pipeline
   - Ensure proper data flow between components

7. **Testing**:
   - Create test scripts to verify functionality
   - Compare results with the original implementation

## Directory Structure

```
ember_ml/
├── features/
│   ├── __init__.py
│   ├── base_feature_extractor.py
│   ├── terabyte_feature_extractor.py
│   └── temporal_processor.py
├── models/
│   ├── __init__.py
│   ├── rbm/
│   │   ├── __init__.py
│   │   ├── rbm_module.py
│   │   └── training.py
│   └── liquid/
│       ├── __init__.py
│       ├── liquid_network_module.py
│       ├── motor_neuron_module.py
│       └── training.py
├── pipeline/
│   ├── __init__.py
│   ├── pipeline_module.py
│   └── training.py
└── examples/
    ├── __init__.py
    ├── pipeline_demo.py
    └── test_pipeline_with_sample_data.py
```

## Conclusion

This refactoring plan outlines a comprehensive approach to modernizing the pipeline implementation using the ember_ml Module system. The new architecture will be more modular, maintainable, and backend-agnostic, while preserving the functionality of the original implementation.

The use of NCP and other ember_ml components will ensure better integration with the rest of the ember_ml ecosystem, and the separation of training logic will make the code more flexible and easier to extend.