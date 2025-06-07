# Function-First Design Pattern in Ember ML

This document provides a detailed explanation of the function-first design pattern used in Ember ML, focusing on how it enables memory optimization and efficient code organization.

## Overview

The function-first design pattern is a core architectural principle in Ember ML that separates function implementations from class definitions. This pattern offers significant advantages in terms of memory usage, code organization, and maintainability.

## Key Principles

### 1. Functions as Primary Units

In the function-first design pattern, functions are the primary units of implementation:

- **Standalone Functions**: Each operation is implemented first as a standalone function
- **Class Methods as Wrappers**: Class methods are thin wrappers around these standalone functions
- **Self as First Argument**: The class instance (`self`) is passed as the first argument to the function

### 2. Separation of Implementation from Interface

The pattern separates the implementation (functions) from the interface (class methods):

- **Implementation in Functions**: The actual implementation logic resides in standalone functions
- **Interface in Classes**: Classes provide a convenient interface for users
- **Lazy Loading**: Functions are only loaded when they're actually used

### 3. Consistent Parameter Ordering

The pattern maintains consistent parameter ordering between functions and methods:

- **Self as First Parameter**: Functions take the class instance as their first parameter
- **Identical Parameter Lists**: The remaining parameters are identical between functions and methods
- **Consistent Documentation**: Parameter documentation is consistent between functions and methods

## Implementation in Ember ML

### Function Implementation

In Ember ML, tensor operations are implemented as standalone functions in dedicated modules:

```python
# In ember_ml/backend/{backend_name}/tensor/ops/creation.py
def zeros(tensor_obj, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        tensor_obj: The tensor object (instance of the backend's tensor class)
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    # Implementation
    processed_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # Backend-specific implementation
    if isinstance(shape, int):
        shape = (shape,)
    
    # Create zeros tensor using the backend's API
    if processed_dtype is not None:
        return mx.zeros(shape, dtype=processed_dtype)
    else:
        return mx.zeros(shape)
```

### Method Implementation

The class methods are thin wrappers that call the standalone functions:

```python
# In ember_ml/backend/{backend_name}/tensor/tensor.py
def zeros(self, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    from ember_ml.backend.{backend_name}.tensor.ops.creation import zeros as zeros_func
    return zeros_func(self, shape, dtype, device)
```

### Frontend Implementation

The frontend EmberTensor class provides a consistent interface across all backends:

```python
# In ember_ml/nn/tensor/common/ember_tensor.py
def zeros(self, shape, dtype=None, device=None):
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    tensor = zeros(shape, dtype=dtype)
    return EmberTensor(tensor, dtype=dtype, device=device, requires_grad=self._requires_grad)
```

## Memory Optimization Benefits

The function-first design pattern provides several memory optimization benefits:

### 1. Lazy Loading

Functions are only loaded when they're actually used:

```python
def method_name(self, *args, **kwargs):
    # The function is only imported when the method is called
    from module.path import function_name
    return function_name(self, *args, **kwargs)
```

This approach:
- **Reduces Initial Memory Footprint**: Only the class definition is loaded initially
- **Loads Functions On-Demand**: Functions are loaded only when they're needed
- **Enables Partial Usage**: Users who only use a subset of functionality don't pay the memory cost for unused functions

### 2. Reduced Memory Overhead

The separation of functions from classes reduces memory overhead:

- **Smaller Class Definitions**: Class definitions are smaller since they don't contain implementation code
- **Shared Function Implementations**: Multiple instances of a class share the same function implementations
- **Efficient Method Dispatch**: Method calls have minimal overhead since they simply delegate to functions

### 3. Improved Garbage Collection

The pattern improves garbage collection efficiency:

- **Function-Level Garbage Collection**: Functions that aren't being used can be garbage collected
- **Reduced Reference Cycles**: Fewer reference cycles since functions don't hold references to class instances
- **Cleaner Memory Profile**: Memory usage more closely tracks actual usage patterns

## Implementation Example: Memory-Efficient RNN

Here's an example of how the function-first design pattern enables memory-efficient RNN implementations:

### Cell Implementation

```python
# In ember_ml/nn/modules/rnn/lstm_cell.py
def lstm_cell_forward(cell, inputs, state):
    """
    LSTM cell forward pass.
    
    Args:
        cell: The LSTM cell instance
        inputs: Input tensor
        state: Previous state (h, c)
        
    Returns:
        Tuple of (output, new_state)
    """
    # Unpack the state
    h_prev, c_prev = state
    
    # Concatenate inputs and previous hidden state
    combined = ops.concatenate([inputs, h_prev], axis=-1)
    
    # Apply the gate transformations
    gates = ops.matmul(combined, cell.weight) + cell.bias
    
    # Split the gates
    i, f, g, o = ops.split(gates, 4, axis=-1)
    
    # Apply the gate activations
    i = ops.sigmoid(i)  # Input gate
    f = ops.sigmoid(f)  # Forget gate
    g = ops.tanh(g)     # Cell gate
    o = ops.sigmoid(o)  # Output gate
    
    # Update the cell state
    c_next = f * c_prev + i * g
    
    # Compute the output
    h_next = o * ops.tanh(c_next)
    
    return h_next, (h_next, c_next)
```

### Cell Class

```python
# In ember_ml/nn/modules/rnn/lstm_cell.py
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.weight = EmberTensor.random_normal((input_size + hidden_size, 4 * hidden_size))
        self.bias = EmberTensor.zeros((4 * hidden_size,))
    
    def forward(self, inputs, state=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            state: Previous state (h, c)
            
        Returns:
            Tuple of (output, new_state)
        """
        # Initialize state if not provided
        if state is None:
            batch_size = inputs.shape[0]
            h = EmberTensor.zeros((batch_size, self.hidden_size))
            c = EmberTensor.zeros((batch_size, self.hidden_size))
            state = (h, c)
        
        # Call the function implementation
        from ember_ml.nn.modules.rnn.lstm_cell import lstm_cell_forward
        return lstm_cell_forward(self, inputs, state)
```

### Layer Implementation

```python
# In ember_ml/nn/modules/rnn/lstm.py
def lstm_forward(lstm, inputs, initial_state=None):
    """
    LSTM layer forward pass.
    
    Args:
        lstm: The LSTM layer instance
        inputs: Input sequence
        initial_state: Initial state for the LSTM
        
    Returns:
        Tuple of (outputs, final_state)
    """
    # Get sequence length and batch size
    seq_len = inputs.shape[1] if lstm.batch_first else inputs.shape[0]
    batch_size = inputs.shape[0] if lstm.batch_first else inputs.shape[1]
    
    # Initialize states
    if initial_state is None:
        h = EmberTensor.zeros((lstm.num_layers, batch_size, lstm.hidden_size))
        c = EmberTensor.zeros((lstm.num_layers, batch_size, lstm.hidden_size))
        states = [(h[i], c[i]) for i in range(lstm.num_layers)]
    else:
        states = initial_state
    
    # Process the sequence
    outputs = []
    for t in range(seq_len):
        # Get input at this time step
        if lstm.batch_first:
            x = inputs[:, t]
        else:
            x = inputs[t]
        
        # Process through each layer
        layer_input = x
        new_states = []
        for i, cell in enumerate(lstm.cells):
            # Process through the cell
            output, new_state = cell(layer_input, states[i])
            new_states.append(new_state)
            
            # Apply dropout except for the last layer
            if i < lstm.num_layers - 1 and lstm.dropout > 0:
                layer_input = ops.dropout(output, lstm.dropout, lstm.training)
            else:
                layer_input = output
        
        # Store the output
        outputs.append(layer_input)
        
        # Update states
        states = new_states
    
    # Stack outputs
    if lstm.batch_first:
        outputs = ops.stack(outputs, axis=1)
    else:
        outputs = ops.stack(outputs, axis=0)
    
    # Return outputs and final state
    return outputs, states
```

### Layer Class

```python
# In ember_ml/nn/modules/rnn/lstm.py
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Create cells
        self.cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size)
            self.cells.append(cell)
    
    def forward(self, inputs, initial_state=None):
        """
        Forward pass.
        
        Args:
            inputs: Input sequence
            initial_state: Initial state for the LSTM
            
        Returns:
            Tuple of (outputs, final_state)
        """
        # Call the function implementation
        from ember_ml.nn.modules.rnn.lstm import lstm_forward
        return lstm_forward(self, inputs, initial_state)
```

## Memory Usage Comparison

To illustrate the memory benefits of the function-first design pattern, here's a comparison of memory usage between traditional and function-first implementations:

### Traditional Implementation

```python
class TraditionalTensor:
    def __init__(self):
        # ...
    
    def zeros(self, shape, dtype=None, device=None):
        # Implementation directly in the method
        processed_dtype = self._validate_dtype(dtype)
        
        if isinstance(shape, int):
            shape = (shape,)
        
        if processed_dtype is not None:
            return backend_zeros(shape, dtype=processed_dtype)
        else:
            return backend_zeros(shape)
    
    def ones(self, shape, dtype=None, device=None):
        # Another implementation directly in the method
        processed_dtype = self._validate_dtype(dtype)
        
        if isinstance(shape, int):
            shape = (shape,)
        
        if processed_dtype is not None:
            return backend_ones(shape, dtype=processed_dtype)
        else:
            return backend_ones(shape)
    
    # Many more methods with implementations...
```

### Function-First Implementation

```python
# In tensor.py
class FunctionFirstTensor:
    def __init__(self):
        # ...
    
    def zeros(self, shape, dtype=None, device=None):
        from ops.creation import zeros as zeros_func
        return zeros_func(self, shape, dtype, device)
    
    def ones(self, shape, dtype=None, device=None):
        from ops.creation import ones as ones_func
        return ones_func(self, shape, dtype, device)
    
    # Many more methods that just delegate to functions...

# In ops/creation.py
def zeros(tensor_obj, shape, dtype=None, device=None):
    # Implementation in a separate function
    processed_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    if processed_dtype is not None:
        return backend_zeros(shape, dtype=processed_dtype)
    else:
        return backend_zeros(shape)

def ones(tensor_obj, shape, dtype=None, device=None):
    # Another implementation in a separate function
    processed_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    if isinstance(shape, int):
        shape = (shape,)
    
    if processed_dtype is not None:
        return backend_ones(shape, dtype=processed_dtype)
    else:
        return backend_ones(shape)
```

### Memory Usage Analysis

The function-first implementation has several memory advantages:

1. **Initial Memory Footprint**:
   - Traditional: All method implementations are loaded when the class is defined
   - Function-First: Only the class definition with thin method wrappers is loaded initially

2. **Partial Usage**:
   - Traditional: All method implementations consume memory even if only a few methods are used
   - Function-First: Only the functions for methods that are actually called are loaded

3. **Multiple Instances**:
   - Traditional: Each instance potentially duplicates method implementation code
   - Function-First: All instances share the same function implementations

4. **Garbage Collection**:
   - Traditional: Method implementations are tied to the class and can't be garbage collected separately
   - Function-First: Unused functions can be garbage collected independently of the class

## Conclusion

The function-first design pattern in Ember ML provides significant memory optimization benefits through lazy loading, reduced memory overhead, and improved garbage collection. This pattern enables efficient implementation of complex operations while maintaining a clean and consistent API.

By separating function implementations from class definitions, Ember ML achieves a balance between user-friendly interfaces and memory-efficient implementation, making it suitable for both small-scale and large-scale machine learning applications.