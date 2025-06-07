# Ember ML Frontend Usage Guide

This guide provides comprehensive documentation on how to use the Ember ML frontend, including tensor operations, neural network components, and backend selection.

## Table of Contents

1. [Backend Selection](#backend-selection)
2. [Tensor Operations](#tensor-operations)
   - [Creating Tensors](#creating-tensors)
   - [Tensor Properties](#tensor-properties)
   - [Basic Operations](#basic-operations)
   - [Reshaping and Manipulation](#reshaping-and-manipulation)
   - [Indexing and Slicing](#indexing-and-slicing)
   - [Random Operations](#random-operations)
3. [Neural Network Components](#neural-network-components)
   - [Basic Modules](#basic-modules)
   - [Recurrent Networks](#recurrent-networks)
   - [Neural Circuit Policies](#neural-circuit-policies)
   - [Restricted Boltzmann Machines](#restricted-boltzmann-machines)
4. [Advanced Usage](#advanced-usage)
   - [Custom Modules](#custom-modules)
   - [Backend-Specific Optimizations](#backend-specific-optimizations)
   - [Memory Management](#memory-management)
5. [Upcoming Features](#upcoming-features)
   - [Operator Overloading](#operator-overloading)
   - [Static Methods](#static-methods)
   - [Comparison Utilities](#comparison-utilities)

## Backend Selection

Ember ML supports multiple backends (NumPy, PyTorch, MLX) through a unified interface. You can select the backend using the `set_backend` function:

```python
from ember_ml.ops import set_backend, get_backend

# Use MLX (optimized for Apple Silicon)
set_backend('mlx')

# Or use PyTorch
# set_backend('torch')

# Or use NumPy
# set_backend('numpy')

# Check the current backend
current_backend = get_backend()
print(f"Current backend: {current_backend}")
```

You can switch backends at runtime, and tensors will be automatically converted to the new backend:

```python
import ember_ml
from ember_ml.nn.tensor import EmberTensor

# Create a tensor using the current backend
x = EmberTensor([1, 2, 3])

# Switch to a different backend
set_backend('torch')

# x is automatically converted to the new backend
y = x + 1  # Uses PyTorch operations
```

## Tensor Operations

### Creating Tensors

You can create tensors in several ways:

```python
from ember_ml.nn.tensor import EmberTensor, float32, zeros, ones, arange, linspace

# Create a tensor from a list
tensor1 = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Create a tensor with a specific data type
tensor2 = EmberTensor([[1, 2, 3], [4, 5, 6]], dtype=float32)

# Create a tensor on a specific device
tensor3 = EmberTensor([[1, 2, 3], [4, 5, 6]], device='cuda')  # For PyTorch backend

# Create a tensor of zeros
zeros_tensor = zeros((2, 3))

# Create a tensor of ones
ones_tensor = ones((2, 3))

# Create a tensor with evenly spaced values
range_tensor = arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Create a tensor with linearly spaced values
linear_tensor = linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Tensor Properties

You can access various properties of tensors:

```python
# Get the shape of a tensor
shape = tensor1.shape  # (2, 3)

# Get the data type of a tensor
dtype = tensor1.dtype  # int64

# Get the device of a tensor
device = tensor1.device  # 'cpu' or 'cuda' or 'mps'

# Check if a tensor requires gradients
requires_grad = tensor1.requires_grad  # False
```

### Basic Operations

Ember ML supports a wide range of tensor operations:

```python
from ember_ml.nn.tensor import EmberTensor
from ember_ml import ops

# Create tensors
a = EmberTensor([1, 2, 3])
b = EmberTensor([4, 5, 6])

# Addition
c = ops.add(a, b)  # [5, 7, 9]

# Subtraction
d = ops.subtract(a, b)  # [-3, -3, -3]

# Multiplication
e = ops.multiply(a, b)  # [4, 10, 18]

# Division
f = ops.divide(a, b)  # [0.25, 0.4, 0.5]

# Matrix multiplication
g = ops.matmul(EmberTensor([[1, 2], [3, 4]]), EmberTensor([[5, 6], [7, 8]]))  # [[19, 22], [43, 50]]

# Reduction operations
sum_a = ops.stats.sum(a)  # 6
mean_a = ops.stats.mean(a)  # 2.0
max_a = ops.stats.max(a)  # 3
min_a = ops.stats.min(a)  # 1
```

### Reshaping and Manipulation

You can reshape and manipulate tensors in various ways:

```python
from ember_ml.nn.tensor import EmberTensor, reshape, transpose, concatenate, stack, split

# Create a tensor
a = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Reshape a tensor
b = reshape(a, (3, 2))  # [[1, 2], [3, 4], [5, 6]]
b = a.reshape((3, 2))  # Same as above

# Transpose a tensor
c = transpose(a)  # [[1, 4], [2, 5], [3, 6]]
c = a.transpose()  # Same as above

# Concatenate tensors
d = concatenate([a, a], axis=0)  # [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
d = a.concatenate([a, a], axis=0)  # Same as above

# Stack tensors
e = stack([a, a], axis=0)  # [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
e = a.stack([a, a], axis=0)  # Same as above

# Split a tensor
f = split(a, 3, axis=1)  # [[[1], [4]], [[2], [5]], [[3], [6]]]
f = a.split(a, 3, axis=1)  # Same as above
```

### Indexing and Slicing

You can index and slice tensors:

```python
from ember_ml.nn.tensor import EmberTensor

# Create a tensor
a = EmberTensor([[1, 2, 3], [4, 5, 6]])

# Get a single element
element = a[0, 1]  # 2

# Get a row
row = a[0]  # [1, 2, 3]

# Get a column
column = a[:, 1]  # [2, 5]

# Get a slice
slice_a = a[0:1, 1:3]  # [[2, 3]]

# Set values
a[0, 1] = 10  # a is now [[1, 10, 3], [4, 5, 6]]
```

### Random Operations

Ember ML provides various random tensor operations:

```python
from ember_ml.nn.tensor import random_normal, random_uniform, random_binomial, set_seed

# Set random seed for reproducibility
set_seed(42)

# Create a tensor with random values from a normal distribution
normal_tensor = random_normal((2, 3), mean=0.0, stddev=1.0)

# Create a tensor with random values from a uniform distribution
uniform_tensor = random_uniform((2, 3), minval=0.0, maxval=1.0)

# Create a tensor with random values from a binomial distribution
binomial_tensor = random_binomial((2, 3), p=0.5)
```

## Neural Network Components

### Basic Modules

Ember ML provides basic neural network modules:

```python
from ember_ml.nn.modules import Linear, Activation, Sequential

# Create a linear layer
linear = Linear(input_size=10, output_size=5)

# Create an activation function
activation = Activation('relu')

# Create a sequential model
model = Sequential([
    Linear(10, 20),
    Activation('relu'),
    Linear(20, 5),
    Activation('sigmoid')
])

# Forward pass
input_tensor = EmberTensor.random_normal((32, 10))  # Batch of 32 samples with 10 features
output = model(input_tensor)  # Shape: (32, 5)
```

### Recurrent Networks

Ember ML supports various recurrent neural networks:

```python
from ember_ml.nn.modules.rnn import RNN, LSTM, GRU, LTC, CFC

# Create an RNN
rnn = RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Create an LSTM
lstm = LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Create a GRU
gru = GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Create a NeuronMap (e.g., FullyConnectedMap)
from ember_ml.nn.modules.wiring import FullyConnectedMap
neuron_map_ltc = FullyConnectedMap(units=20, input_dim=10, output_dim=20)
# Create an LTC (Liquid Time-Constant) network using the map
ltc = LTC(
    neuron_map=neuron_map_ltc,
    batch_first=True
)

# Create a CFC (Closed-form Continuous-time) network
from ember_ml.nn.modules.wiring import NCPMap
neuron_map_cfc = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10
)
cfc = CFC(
    neuron_map=neuron_map_cfc,
    batch_first=True
)

# Forward pass
input_sequence = EmberTensor.random_normal((32, 10, 10))  # Batch of 32 sequences, each with 10 time steps and 10 features
output = lstm(input_sequence)  # Shape: (32, 10, 20) if return_sequences=True, else (32, 20)
```

### Neural Circuit Policies

Ember ML provides Neural Circuit Policy (NCP) implementations:

```python
# Import the appropriate NeuronMap (e.g., NCPMap)
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.ncp import NCP

# Create a neuron map configuration
neuron_map_ncp = NCPMap(
    inter_neurons=30, # Example values
    command_neurons=24,
    motor_neurons=4,
    sensory_neurons=10 # Must match input features
)

# Create an NCP
model = NCP(neuron_map=neuron_map_ncp) # Use neuron_map argument

# Forward pass
input_tensor = EmberTensor.random_normal((32, 10)) # Batch of 32 samples, input features determined by build
output = model(input_tensor)  # Shape: (32, 4) (matches output_size)
```

### Restricted Boltzmann Machines

Ember ML includes Restricted Boltzmann Machine (RBM) implementations:

```python
from ember_ml.nn.modules.rbm import RestrictedBoltzmannMachine

# Create an RBM
rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=256)

# Train the RBM
training_data = EmberTensor.random_uniform((100, 784))  # 100 samples with 784 features
rbm.train(training_data, epochs=20)

# Extract features
features = rbm.transform(training_data)

# Generate samples
samples = rbm.generate(n_samples=10)

# Detect anomalies
anomaly_scores = rbm.anomaly_score(training_data)
is_anomaly = rbm.is_anomaly(training_data)
```

## Advanced Usage

### Custom Modules

You can create custom modules by extending the `Module` class:

```python
from ember_ml.nn.modules import Module
from ember_ml.nn.tensor import EmberTensor
from ember_ml import ops

class CustomModule(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = EmberTensor.random_normal((input_size, hidden_size))
        self.b1 = EmberTensor.zeros((hidden_size,))
        self.W2 = EmberTensor.random_normal((hidden_size, output_size))
        self.b2 = EmberTensor.zeros((output_size,))
        
    def forward(self, x):
        # First layer (use ops.add)
        z1 = ops.matmul(x, self.W1)
        h = ops.add(z1, self.b1)
        h_act = ops.relu(h) # Assuming relu is available via ops

        # Second layer (use ops.add)
        z2 = ops.matmul(h_act, self.W2)
        y = ops.add(z2, self.b2)
        y_act = ops.sigmoid(y) # Assuming sigmoid is available via ops
        
        return y_act
```

### Backend-Specific Optimizations

You can optimize your code for specific backends:

```python
from ember_ml.backend import get_backend

# Check the current backend
backend = get_backend()

if backend == 'torch':
    # PyTorch-specific optimizations
    # ...
elif backend == 'mlx':
    # MLX-specific optimizations
    # ...
elif backend == 'numpy':
    # NumPy-specific optimizations
    # ...
```

### Memory Management

Ember ML is designed to minimize memory usage through its function-first design pattern. Here are some tips for managing memory:

```python
from ember_ml.nn.tensor import EmberTensor

# Create a tensor
a = EmberTensor.random_normal((1000, 1000))

# Use in-place operations where possible
a = ops.add(a, 1) # Use ops.add for operations

# Reuse tensors instead of creating new ones
b = EmberTensor.zeros_like(a)
for i in range(10):
    # Update b in-place instead of creating a new tensor
    b = ops.add(b, a) # Use ops.add

# Release tensors when they're no longer needed
del a
```

## Upcoming Features

### Operator Overloading

In a future release, Ember ML will support operator overloading for EmberTensor to enable intuitive arithmetic and comparison operations:

```python
# Arithmetic operations
c = EmberTensor([1, 2]) + EmberTensor([2, 3])  # [3, 5]
c = EmberTensor([5, 6]) - EmberTensor([2, 3])  # [3, 3]
c = EmberTensor([1, 2]) * EmberTensor([2, 3])  # [2, 6]
c = EmberTensor([4, 6]) / EmberTensor([2, 3])  # [2, 2]

# Comparison operations
result = EmberTensor([1, 1]) == EmberTensor([1, 1])  # [True, True]
result = EmberTensor([1, 2]) != EmberTensor([1, 1])  # [False, True]
result = EmberTensor([1, 2]) > EmberTensor([2, 1])   # [False, True]
result = EmberTensor([1, 2]) < EmberTensor([2, 1])   # [True, False]

# Scalar operations
c = EmberTensor([1, 2]) + 2  # [3, 4]
c = 3 * EmberTensor([1, 2])  # [3, 6]
```

Currently, these operations need to be performed using the ops module:

```python
from ember_ml import ops

# Arithmetic operations (Examples already correctly use ops)
c = ops.add(EmberTensor([1, 2]), EmberTensor([2, 3]))
c = ops.subtract(EmberTensor([5, 6]), EmberTensor([2, 3]))
c = ops.multiply(EmberTensor([1, 2]), EmberTensor([2, 3]))
c = ops.divide(EmberTensor([4, 6]), EmberTensor([2, 3]))
```

### Static Methods

Future releases will include static methods on the EmberTensor class for common operations:

```python
# Current approach
tensor_obj = EmberTensor([0])
a = tensor_obj.zeros((2, 3))

# Future approach
a = EmberTensor.zeros((2, 3))
b = EmberTensor.ones((2, 3))
c = EmberTensor.random_normal((2, 3))
```

### Comparison Utilities

Future releases will include intuitive array comparison functions:

```python
# Future utilities
from ember_ml.nn.tensor import array_equal, allclose

assert array_equal(tensor1, tensor2)
assert allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)

# Testing helpers
from ember_ml.nn.tensor.testing import assert_tensor_equal, assert_tensor_allclose

assert_tensor_equal(tensor1, tensor2)
assert_tensor_allclose(tensor1, tensor2)
```

## Example: Building and Training a Model

Here's a complete example of building and training a model:

```python
import ember_ml
from ember_ml.nn.tensor import EmberTensor
from ember_ml.nn.modules import Linear, Activation, Sequential
from ember_ml import ops

# Set the backend
ember_ml.backend.set_backend('mlx')

# Create a dataset
X = EmberTensor.random_normal((1000, 10))
y = EmberTensor.random_binomial((1000, 1), p=0.5)

# Split into train and test sets
train_X = X[:800]
train_y = y[:800]
test_X = X[800:]
test_y = y[800:]

# Create a model
model = Sequential([
    Linear(10, 20),
    Activation('relu'),
    Linear(20, 1),
    Activation('sigmoid')
])

# Training parameters
learning_rate = 0.01
batch_size = 32
epochs = 10

# Training loop
for epoch in range(epochs):
    total_loss = 0
    
    # Process in batches
    for i in range(0, len(train_X), batch_size):
        # Get batch
        batch_X = train_X[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]
        
        # Forward pass
        predictions = model(batch_X)
        
        # Compute loss
        loss = ops.binary_cross_entropy(predictions, batch_y)
        total_loss += loss.item()
        
        # Backward pass and update weights
        # (This would depend on the backend's autograd capabilities)
        # ...
    
    # Print epoch results
    avg_loss = total_loss / (len(train_X) // batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
# Evaluate on test set
test_predictions = model(test_X)
test_loss = ops.binary_cross_entropy(test_predictions, test_y).item()
test_accuracy = ops.stats.mean(tensor.cast(ops.round(test_predictions) == test_y, 'float32')).item()
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
```

## Conclusion

This guide provides a comprehensive overview of the Ember ML frontend. For more detailed information on specific components, refer to the following documentation:

- [Tensor Architecture](tensor_architecture.md): Detailed explanation of the tensor operations architecture
- [Neural Network Modules](../architecture/nn_modules_architecture.md): Documentation for neural network modules
- [Backend Implementation Guide](../plans/tensor_implementation/tensor_implementation_guide.md): Guide for implementing backend-specific tensor operations