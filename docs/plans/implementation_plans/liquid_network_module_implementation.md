# Liquid Network Module Implementation Plan

This document provides a detailed implementation plan for refactoring the Liquid Neural Network components to use the ember_ml Module system and Neural Circuit Policies (NCP).

## Current Implementation

The current liquid neural network implementation in `ember_ml.core.stride_aware_cfc` includes:

1. Factory functions for creating different types of liquid networks:
   - `create_liquid_network_with_motor_neuron`
   - `create_lstm_gated_liquid_network`
   - `create_multi_stride_liquid_network`

2. Classes for different components:
   - `StrideAwareCfCCell`: A CfC cell with stride-aware processing
   - `LiquidNetworkWithMotorNeuron`: A liquid network with motor neuron output
   - `MotorNeuron`: A neuron that generates output values and trigger signals
   - `LSTMGatedLiquidNetwork`: A network using LSTM to gate CfC output
   - `MultiStrideLiquidNetwork`: A network processing inputs at multiple time scales

The implementation uses TensorFlow for some operations and doesn't fully leverage the ember_ml Module system or NCP.

## Refactoring Goals

1. **Module Integration**: Implement liquid networks as subclasses of `Module`
2. **NCP Integration**: Use Neural Circuit Policies for network connectivity
3. **Backend Agnosticism**: Use `ops` for all operations to support any backend
4. **Parameter Management**: Use the `Parameter` class for trainable parameters
5. **Training Separation**: Separate model definition from training logic
6. **Maintain Functionality**: Preserve all existing functionality

## Implementation Details

### 1. Base Liquid Network Module

```python
class BaseLiquidNetworkModule(Module):
    """
    Base class for liquid neural networks using the ember_ml Module system.
    
    This module provides a foundation for building liquid neural networks
    with different architectures and connectivity patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        **kwargs
    ):
        """
        Initialize the base liquid network module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, input_dim]
            states: Initial states (optional)
            **kwargs: Additional arguments
            
        Returns:
            Network outputs
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_initial_state(self, batch_size):
        """
        Get initial state for the network.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        raise NotImplementedError("Subclasses must implement get_initial_state method")
```

### 2. Motor Neuron Module

```python
class MotorNeuronModule(Module):
    """
    Motor neuron module that generates output values and trigger signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.5,
        adaptive_threshold: bool = True
    ):
        """
        Initialize the motor neuron module.
        
        Args:
            input_dim: Input dimension
            threshold: Threshold for activation
            adaptive_threshold: Whether to use adaptive thresholding
        """
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
        """
        Forward pass through the motor neuron.
        
        Args:
            inputs: Input tensor [batch_size, input_dim]
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, trigger, threshold)
        """
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

### 3. NCP-Based Liquid Network Module

```python
class NCPLiquidNetworkModule(BaseLiquidNetworkModule):
    """
    Liquid neural network module using Neural Circuit Policies.
    """
    
    def __init__(
        self,
        input_dim: int,
        units: int = 128,
        output_dim: int = 1,
        sparsity_level: float = 0.5,
        threshold: float = 0.5,
        adaptive_threshold: bool = True,
        mixed_memory: bool = True,
        **kwargs
    ):
        """
        Initialize the NCP-based liquid network module.
        
        Args:
            input_dim: Input dimension
            units: Number of units in the NCP
            output_dim: Output dimension
            sparsity_level: Sparsity level for the NCP
            threshold: Threshold for motor neuron activation
            adaptive_threshold: Whether to use adaptive thresholding
            mixed_memory: Whether to use mixed memory
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.units = units
        self.sparsity_level = sparsity_level
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.mixed_memory = mixed_memory
        
        # Create NCP wiring
        self.wiring = AutoNCP(
            units=units,
            output_size=units,  # Full output size for internal processing
            sparsity_level=sparsity_level
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
            input_shape=(units,)
        )
        
        # Motor neuron
        self.motor_neuron = MotorNeuronModule(
            input_dim=units,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold
        )
        
        # Mixed memory
        if mixed_memory:
            self.memory_gate = ops.nn.Dense(
                units=units,
                input_shape=(units,),
                activation="sigmoid"
            )
    
    def get_initial_state(self, batch_size):
        """
        Get initial state for the NCP cell.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state
        """
        return self.ncp_cell.get_initial_state(batch_size)
    
    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, input_dim]
            states: Initial states (optional)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (outputs, [trigger_signals, threshold_values])
        """
        # Initialize states if not provided
        if states is None:
            batch_size = ops.shape(inputs)[0]
            states = self.get_initial_state(batch_size)
        
        # Process each time step
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
            
            # Apply mixed memory if enabled
            if self.mixed_memory and t > 0:
                memory_gate = self.memory_gate(cell_output)
                cell_output = memory_gate * cell_output + (1 - memory_gate) * outputs[-1]
            
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

### 4. LSTM-Gated Liquid Network Module

```python
class LSTMGatedLiquidNetworkModule(BaseLiquidNetworkModule):
    """
    LSTM-gated liquid neural network module using Neural Circuit Policies.
    """
    
    def __init__(
        self,
        input_dim: int,
        ncp_units: int = 128,
        lstm_units: int = 32,
        output_dim: int = 1,
        sparsity_level: float = 0.5,
        **kwargs
    ):
        """
        Initialize the LSTM-gated liquid network module.
        
        Args:
            input_dim: Input dimension
            ncp_units: Number of units in the NCP
            lstm_units: Number of units in the LSTM
            output_dim: Output dimension
            sparsity_level: Sparsity level for the NCP
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.ncp_units = ncp_units
        self.lstm_units = lstm_units
        self.sparsity_level = sparsity_level
        
        # Create NCP wiring
        self.ncp_wiring = AutoNCP(
            units=ncp_units,
            output_size=ncp_units,
            sparsity_level=sparsity_level
        )
        
        # Create NCP cell
        self.ncp_cell = NCP(
            wiring=self.ncp_wiring,
            in_features=input_dim
        )
        
        # Create LSTM cell
        self.lstm_cell = LSTMCell(
            input_size=input_dim,
            hidden_size=lstm_units
        )
        
        # Input projection for NCP
        self.ncp_input_projection = ops.nn.Dense(
            units=input_dim,
            input_shape=(input_dim,)
        )
        
        # Input projection for LSTM
        self.lstm_input_projection = ops.nn.Dense(
            units=input_dim,
            input_shape=(input_dim,)
        )
        
        # Output projection
        self.output_projection = ops.nn.Dense(
            units=output_dim,
            input_shape=(ncp_units + lstm_units,)
        )
        
        # Gating mechanism
        self.gate = ops.nn.Dense(
            units=ncp_units,
            input_shape=(lstm_units,),
            activation="sigmoid"
        )
    
    def get_initial_state(self, batch_size):
        """
        Get initial states for the NCP and LSTM cells.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (ncp_states, lstm_states)
        """
        ncp_states = self.ncp_cell.get_initial_state(batch_size)
        lstm_states = self.lstm_cell.get_initial_state(batch_size)
        return [ncp_states, lstm_states]
    
    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, input_dim]
            states: Initial states (optional)
            **kwargs: Additional arguments
            
        Returns:
            Network outputs
        """
        # Initialize states if not provided
        if states is None:
            batch_size = ops.shape(inputs)[0]
            states = self.get_initial_state(batch_size)
        
        # Unpack states
        ncp_states, lstm_states = states
        
        # Process each time step
        outputs = []
        
        for t in range(ops.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Project input for NCP
            ncp_input = self.ncp_input_projection(x_t)
            
            # Project input for LSTM
            lstm_input = self.lstm_input_projection(x_t)
            
            # Process through NCP
            ncp_output, ncp_states = self.ncp_cell(ncp_input, ncp_states)
            
            # Process through LSTM
            lstm_output, lstm_states = self.lstm_cell(lstm_input, lstm_states)
            
            # Generate gate
            gate = self.gate(lstm_output)
            
            # Apply gate to NCP output
            gated_ncp_output = gate * ncp_output
            
            # Concatenate gated NCP output and LSTM output
            combined_output = ops.concatenate([gated_ncp_output, lstm_output], axis=-1)
            
            # Project output
            output = self.output_projection(combined_output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs
        outputs = ops.stack(outputs, axis=1)
        
        return outputs
```

### 5. Multi-Stride Liquid Network Module

```python
class MultiStrideLiquidNetworkModule(BaseLiquidNetworkModule):
    """
    Multi-stride liquid neural network module using Neural Circuit Policies.
    """
    
    def __init__(
        self,
        input_dim: int,
        stride_perspectives: List[int] = [1, 3, 5],
        units_per_stride: int = 32,
        output_dim: int = 1,
        sparsity_level: float = 0.5,
        **kwargs
    ):
        """
        Initialize the multi-stride liquid network module.
        
        Args:
            input_dim: Input dimension
            stride_perspectives: List of stride lengths
            units_per_stride: Number of units per stride
            output_dim: Output dimension
            sparsity_level: Sparsity level for the NCP
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.stride_perspectives = stride_perspectives
        self.units_per_stride = units_per_stride
        self.sparsity_level = sparsity_level
        
        # Create NCP cells for each stride
        self.ncp_cells = {}
        self.input_projections = {}
        
        for stride in stride_perspectives:
            # Create NCP wiring
            wiring = AutoNCP(
                units=units_per_stride,
                output_size=units_per_stride,
                sparsity_level=sparsity_level
            )
            
            # Create NCP cell
            self.ncp_cells[stride] = NCP(
                wiring=wiring,
                in_features=input_dim
            )
            
            # Input projection
            self.input_projections[stride] = ops.nn.Dense(
                units=input_dim,
                input_shape=(input_dim,)
            )
        
        # Output projection
        total_units = len(stride_perspectives) * units_per_stride
        self.output_projection = ops.nn.Dense(
            units=output_dim,
            input_shape=(total_units,)
        )
    
    def get_initial_state(self, batch_size):
        """
        Get initial states for all NCP cells.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of states for each stride
        """
        states = {}
        for stride, cell in self.ncp_cells.items():
            states[stride] = cell.get_initial_state(batch_size)
        return states
    
    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor [batch_size, time_steps, input_dim]
            states: Initial states (optional)
            **kwargs: Additional arguments
            
        Returns:
            Network outputs
        """
        # Initialize states if not provided
        if states is None:
            batch_size = ops.shape(inputs)[0]
            states = self.get_initial_state(batch_size)
        
        # Process each time step
        outputs = []
        
        for t in range(ops.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Process through each cell
            cell_outputs = []
            for stride, cell in self.ncp_cells.items():
                # Only process at appropriate stride steps
                if t % stride == 0:
                    # Project input
                    projected_input = self.input_projections[stride](x_t)
                    
                    # Process through cell
                    cell_output, states[stride] = cell(projected_input, states[stride])
                    
                    # Store cell output
                    cell_outputs.append(cell_output)
                elif stride in states and len(states[stride]) > 0:
                    # Use previous output for non-stride steps
                    cell_outputs.append(states[stride][0])
                else:
                    # Initialize with zeros if no previous output
                    cell_outputs.append(ops.zeros((ops.shape(x_t)[0], self.units_per_stride)))
            
            # Concatenate cell outputs
            combined_output = ops.concatenate(cell_outputs, axis=-1)
            
            # Project output
            output = self.output_projection(combined_output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs
        outputs = ops.stack(outputs, axis=1)
        
        return outputs
```

### 6. Factory Functions

```python
def create_ncp_liquid_network(
    input_dim: int,
    units: int = 128,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    threshold: float = 0.5,
    adaptive_threshold: bool = True,
    mixed_memory: bool = True,
    **kwargs
) -> NCPLiquidNetworkModule:
    """
    Create an NCP-based liquid neural network.
    
    Args:
        input_dim: Input dimension
        units: Number of units in the NCP
        output_dim: Output dimension
        sparsity_level: Sparsity level for the NCP
        threshold: Threshold for motor neuron activation
        adaptive_threshold: Whether to use adaptive thresholding
        mixed_memory: Whether to use mixed memory
        **kwargs: Additional arguments
        
    Returns:
        NCPLiquidNetworkModule
    """
    return NCPLiquidNetworkModule(
        input_dim=input_dim,
        units=units,
        output_dim=output_dim,
        sparsity_level=sparsity_level,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
        mixed_memory=mixed_memory,
        **kwargs
    )

def create_lstm_gated_liquid_network(
    input_dim: int,
    ncp_units: int = 128,
    lstm_units: int = 32,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    **kwargs
) -> LSTMGatedLiquidNetworkModule:
    """
    Create an LSTM-gated liquid neural network.
    
    Args:
        input_dim: Input dimension
        ncp_units: Number of units in the NCP
        lstm_units: Number of units in the LSTM
        output_dim: Output dimension
        sparsity_level: Sparsity level for the NCP
        **kwargs: Additional arguments
        
    Returns:
        LSTMGatedLiquidNetworkModule
    """
    return LSTMGatedLiquidNetworkModule(
        input_dim=input_dim,
        ncp_units=ncp_units,
        lstm_units=lstm_units,
        output_dim=output_dim,
        sparsity_level=sparsity_level,
        **kwargs
    )

def create_multi_stride_liquid_network(
    input_dim: int,
    stride_perspectives: List[int] = [1, 3, 5],
    units_per_stride: int = 32,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    **kwargs
) -> MultiStrideLiquidNetworkModule:
    """
    Create a multi-stride liquid neural network.
    
    Args:
        input_dim: Input dimension
        stride_perspectives: List of stride lengths
        units_per_stride: Number of units per stride
        output_dim: Output dimension
        sparsity_level: Sparsity level for the NCP
        **kwargs: Additional arguments
        
    Returns:
        MultiStrideLiquidNetworkModule
    """
    return MultiStrideLiquidNetworkModule(
        input_dim=input_dim,
        stride_perspectives=stride_perspectives,
        units_per_stride=units_per_stride,
        output_dim=output_dim,
        sparsity_level=sparsity_level,
        **kwargs
    )
```

### 7. Training Functions

```python
def train_liquid_network(
    liquid_network: BaseLiquidNetworkModule,
    features: np.ndarray,
    targets: np.ndarray,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Train a liquid neural network.
    
    Args:
        liquid_network: Liquid network module
        features: Input features [n_samples, time_steps, input_dim]
        targets: Target values [n_samples, output_dim]
        validation_data: Optional validation data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of training history
    """
    # Convert inputs to tensors
    features = tensor.convert_to_tensor(features, dtype=ops.float32)
    targets = tensor.convert_to_tensor(targets, dtype=ops.float32)
    
    # Convert validation data if provided
    if validation_data is not None:
        val_features, val_targets = validation_data
        val_features = tensor.convert_to_tensor(val_features, dtype=ops.float32)
        val_targets = tensor.convert_to_tensor(val_targets, dtype=ops.float32)
        validation_data = (val_features, val_targets)
    
    # Create optimizer
    optimizer = ops.optimizers.Adam(learning_rate=learning_rate)
    
    # Training loop
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = kwargs.get('patience', 10)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = ops.random.permutation(ops.shape(features)[0])
        features_shuffled = ops.gather(features, indices)
        targets_shuffled = ops.gather(targets, indices)
        
        # Train on batches
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, ops.shape(features)[0], batch_size):
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
                if patience_counter >= patience:
                    break
    
    return history
```

## Integration with Pipeline

The Liquid Network Modules will be integrated into the pipeline as follows:

```python
# In pipeline_module.py
from ember_ml.models.liquid.liquid_network_module import (
    create_ncp_liquid_network,
    create_lstm_gated_liquid_network,
    create_multi_stride_liquid_network
)
from ember_ml.models.liquid.training import train_liquid_network

class PipelineModule(Module):
    def __init__(self, feature_dim, rbm_hidden_units=64, ncp_units=128, **kwargs):
        super().__init__()
        # ...
        
        # Initialize liquid network
        network_type = kwargs.get('network_type', 'standard')
        
        if network_type == 'lstm_gated':
            self.liquid_network = create_lstm_gated_liquid_network(
                input_dim=rbm_hidden_units,
                ncp_units=ncp_units,
                lstm_units=kwargs.get('lstm_units', 32),
                output_dim=kwargs.get('output_dim', 1),
                sparsity_level=kwargs.get('sparsity_level', 0.5)
            )
        elif network_type == 'multi_stride':
            self.liquid_network = create_multi_stride_liquid_network(
                input_dim=rbm_hidden_units,
                stride_perspectives=kwargs.get('stride_perspectives', [1, 3, 5]),
                units_per_stride=ncp_units // len(kwargs.get('stride_perspectives', [1, 3, 5])),
                output_dim=kwargs.get('output_dim', 1),
                sparsity_level=kwargs.get('sparsity_level', 0.5)
            )
        else:  # standard
            self.liquid_network = create_ncp_liquid_network(
                input_dim=rbm_hidden_units,
                units=ncp_units,
                output_dim=kwargs.get('output_dim', 1),
                sparsity_level=kwargs.get('sparsity_level', 0.5),
                threshold=kwargs.get('threshold', 0.5),
                adaptive_threshold=kwargs.get('adaptive_threshold', True),
                mixed_memory=kwargs.get('mixed_memory', True)
            )
        
        # ...
    
    def train_liquid_network_component(self, features, targets, validation_data=None, **kwargs):
        """Train the liquid network component of the pipeline."""
        return train_liquid_network(
            self.liquid_network,
            features,
            targets,
            validation_data=validation_data,
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )
    
    def process_data(self, features, return_triggers=True):
        """Process data through the pipeline."""
        # Extract RBM features
        rbm_features = self.rbm(features)
        
        # Reshape for sequence input if needed
        if len(rbm_features.shape) == 2:
            rbm_features = rbm_features.reshape(rbm_features.shape[0], 1, rbm_features.shape[1])
        
        # Process through liquid network
        outputs = self.liquid_network(rbm_features)
        
        # Return outputs based on return_triggers
        if return_triggers:
            if isinstance(outputs, tuple) and len(outputs) > 1:
                motor_outputs, trigger_signals = outputs[0], outputs[1][0]
                return motor_outputs, trigger_signals
            else:
                motor_outputs = outputs
                trigger_signals = (motor_outputs > self.liquid_network.threshold).astype(float)
                return motor_outputs, trigger_signals
        else:
            if isinstance(outputs, tuple):
                return outputs[0]
            else:
                return outputs
```

## Testing Plan

1. **Unit Tests**:
   - Test initialization of each module
   - Test forward pass with different input shapes
   - Test state management
   - Test motor neuron functionality

2. **Integration Tests**:
   - Test training with small dataset
   - Test different network types
   - Test integration with pipeline

3. **Comparison Tests**:
   - Compare results with original implementation
   - Verify numerical stability and accuracy

## Implementation Timeline

1. **Day 1**: Implement base modules and motor neuron
2. **Day 2**: Implement NCP-based liquid network
3. **Day 3**: Implement LSTM-gated and multi-stride networks
4. **Day 4**: Implement training functions
5. **Day 5**: Write tests and debug
6. **Day 6**: Integrate with pipeline and finalize

## Conclusion

This implementation plan provides a detailed roadmap for refactoring the Liquid Neural Network components to use the ember_ml Module system and Neural Circuit Policies. The resulting implementation will be more modular, maintainable, and backend-agnostic, while preserving all the functionality of the original implementation.