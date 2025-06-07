# Ember ML Architecture Summary

## Overview

Ember ML is a modern machine learning library designed for hardware-optimized neural networks with multi-backend support. The project implements various cutting-edge neural network architectures and provides a flexible, backend-agnostic tensor operations framework that can run efficiently on different hardware platforms (CUDA, Apple Metal, and other platforms).

## Core Design Principles

1. **Function-First Design Pattern**
   - Standalone functions as primary implementation units
   - Class methods as thin wrappers around these functions
   - Self as first argument to functions
   - Consistent API between functions and methods
   - Memory optimization through lazy loading

2. **Backend Abstraction**
   - Complete backend agnosticism in frontend code
   - Backend purity with no direct imports of backend-specific libraries
   - Runtime backend switching with automatic tensor conversion
   - Consistent API across all backends
   - NumPy as the universal intermediate format for cross-backend conversions

3. **Modular Component Architecture**
   - Base module system for building neural network components
   - Cell-based recurrent networks with consistent interfaces
   - Neural Circuit Policies with custom wiring
   - Comprehensive tensor operations with consistent API

4. **Memory Optimization**
   - Separation of functions from class implementations
   - Lazy loading of functions
   - Reduced memory footprint
   - Efficient garbage collection

5. **Strong Typing**
   - Explicit type annotations for tensor operations
   - Rejection of incompatible tensor types
   - Clear error messages for type mismatches
   - Consistent type conversion between backends via NumPy

## Architecture Components

### 1. Tensor Operations Framework

The tensor operations framework is organized into three main layers:

#### Frontend Layer
- **EmberTensor**: Main tensor class that users interact with
- **EmberDType**: Backend-agnostic data type class
- **Standalone Functions**: Functions for tensor operations that can be called directly

#### Backend Abstraction Layer
- **Backend Selection**: Functions for selecting and getting the current backend
- **Backend Registration**: Mechanism for registering new backends
- **Device Management**: Functions for managing devices (CPU, GPU, etc.)

#### Backend Implementations
Each backend implements the same API through a consistent structure:
```
ember_ml/backend/{backend_name}/tensor/
  ├── __init__.py           # Exports the tensor class and operations
  ├── tensor.py             # Contains the tensor class with method interfaces
  ├── dtype.py              # Contains the data type class
  ├── ops/                  # Directory for operation modules
  │   ├── __init__.py       # Exports all operations
  │   ├── casting.py        # Contains cast() and related functions
  │   ├── creation.py       # Contains zeros(), ones(), etc.
  │   ├── manipulation.py   # Contains reshape(), transpose(), etc.
  │   ├── indexing.py       # Contains slice(), gather(), etc.
  │   ├── utility.py        # Contains convert_to_tensor(), to_numpy(), etc.
  │   └── random.py         # Contains random_normal(), random_uniform(), etc.
```

### 2. Strong Typing Implementation

The MLX backend implements strong typing for tensor operations, which provides several benefits:

1. **Type Safety**: Ensures that operations are performed on compatible tensor types
2. **Clear Error Messages**: Provides informative error messages when type mismatches occur
3. **Explicit Conversions**: Requires explicit conversion between different tensor types
4. **Performance Optimization**: Avoids unnecessary type conversions

#### Key Components of Strong Typing

1. **Explicit Type Annotations**:
   ```python
   def convert_to_tensor(data: Union[int,float,list,tuple,numpy.ndarray,mx.array,MLXTensor], 
                         dtype: Optional[DType] = None, 
                         device: Optional[str] = None) -> mx.array:
   ```

2. **Type Validation and Conversion**:
   ```python
   def _convert_input(x: Any) -> mx.array:
       # Already an MLX array - check by type and module
       if isinstance(x, mx.array) or (hasattr(x, '__class__') and
                                     hasattr(x.__class__, '__module__') and
                                     x.__class__.__module__ == 'mlx.core' and
                                     x.__class__.__name__ == 'array'):
           return x
           
       # Handle EmberTensor and MLXTensor objects
       if hasattr(x, '__class__') and hasattr(x.__class__, '__name__') and x.__class__.__name__ in ['EmberTensor', 'MLXTensor']:
           # For EmberTensor, extract the underlying data and convert to numpy first
           if hasattr(x, 'to_numpy'):
               return mx.array(x.to_numpy())
           # If it has a _tensor attribute, use that
           elif hasattr(x, '_tensor'):
               return _convert_input(x._tensor)
               
       # Check for NumPy arrays by type name rather than direct import
       if hasattr(x, '__class__') and x.__class__.__module__ == 'numpy' and x.__class__.__name__ == 'ndarray':
           return mx.array(x)
           
       # Handle Python scalars and sequences
       if isinstance(x, (int, float, bool, list, tuple)):
           try:
               return mx.array(x)
           except Exception as e:
               raise ValueError(f"Cannot convert {type(x)} to MLX array: {e}")
       
       # Check for PyTorch tensors and reject them explicitly
       if hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'torch':
           raise ValueError(f"Cannot convert {type(x)} to MLX array. Use tensor.to_numpy() first.")
   ```

3. **Explicit Rejection of Incompatible Types**:
   ```python
   # Check for PyTorch tensors and reject them explicitly
   if hasattr(x, '__class__') and hasattr(x.__class__, '__module__') and x.__class__.__module__ == 'torch':
       raise ValueError(f"Cannot convert {type(x)} to MLX array. Use tensor.to_numpy() first.")
   ```

4. **Type-Specific Function Signatures**:
   ```python
   def slice_tensor(tensor: Union[MLXTensor,mx.array], starts: Sequence[int], sizes: Sequence[int]) -> mx.array:
   ```

5. **Consistent Type Conversion**:
   ```python
   # Convert inputs to MLX arrays
   tensor_array = Tensor.convert_to_tensor(tensor)
   indices_array = Tensor.convert_to_tensor(indices)
   ```

#### Allowed Conversion Paths

The strong typing implementation enforces specific conversion paths between different tensor types:

1. **Same-Backend Conversions** (always allowed):
   - NumPy to NumPy
   - Torch to Torch
   - MLX to MLX

2. **Python Primitives to Any Backend** (always allowed):
   - Python floats, integers, booleans to any backend tensor

3. **NumPy as Universal Intermediate**:
   - NumPy arrays to any backend tensor
   - Any backend tensor to NumPy array

4. **Python Data Structure Conversions**:
   - Any backend tensor to Python list via `tolist()`
   - Any backend tensor to Python scalar via `item()`

5. **Cross-Backend Conversions** (must go through NumPy):
   ```python
   # Convert from MLX to PyTorch
   x_mlx = tensor.ones((3, 4))  # MLX tensor
   x_np = tensor.to_numpy(x_mlx)  # Convert to NumPy
   x_torch = tensor.convert_to_tensor(x_np)  # Convert to PyTorch
   ```

This approach ensures that tensor conversions are explicit and intentional, reducing the risk of unexpected behavior due to implicit conversions. While this slightly compromises absolute backend purity, it's a deliberate design choice that leverages NumPy's universal compatibility for data conversion purposes only.

#### EmberTensor Python Operator Support

EmberTensor implements numerous dunder methods to allow seamless operations with Python operators:

```python
# Example of operator overloading in EmberTensor
def __add__(self, other):
    return ops.add(self, other)

def __mul__(self, other):
    return ops.multiply(self, other)

def __getitem__(self, key):
    return ops.slice(self, key)
```

This allows for intuitive, Pythonic code when working with EmberTensor objects:

```python
# Using Python operators with EmberTensor
x = EmberTensor([1, 2, 3])
y = EmberTensor([4, 5, 6])
z = x + y  # Uses __add__ which calls ops.add
w = x * 2  # Uses __mul__ which calls ops.multiply
element = x[0]  # Uses __getitem__ which calls ops.slice
```

The operator support is continuously expanding as more operations are tested and implemented, making the EmberTensor API increasingly Pythonic and user-friendly.

### 3. Mathematical Operations

The framework provides a comprehensive set of mathematical operations that are implemented for each backend:

#### Basic Math Operations
- **Arithmetic Operations**: add, subtract, multiply, divide
- **Linear Algebra**: dot, matmul
- **Statistical Functions**: mean, sum, var
- **Exponential and Logarithmic**: exp, log, log10, log2
- **Trigonometric Functions**: sin, cos, tan, sinh, cosh
- **Activation Functions**: sigmoid, softplus, tanh, relu

#### Advanced Math Operations
- **Reduction Operations**: min, max, softmax
- **Manipulation**: clip, abs, negative, sign, square
- **Calculus**: gradient

#### Solver Operations
- **Linear System Solvers**: solve, inv
- **Matrix Decompositions**: svd, eig, qr, cholesky
- **Eigenvalue Problems**: eigvals
- **Matrix Properties**: det, norm
- **Least Squares**: lstsq

These operations are implemented in a backend-specific manner while maintaining a consistent API across all backends. The MLX backend, for example, implements these operations using MLX's native functions, with custom implementations for operations not directly available in MLX.

### 4. Pipeline Architecture

Ember ML is evolving towards a comprehensive pipeline architecture that integrates both neural and non-neural components in a flexible, modular system. This approach allows for complex data processing workflows that can leverage different backends and processing paradigms.

#### Current Pipeline Implementation

The current implementation, exemplified by the `IntegratedPipeline` class, provides a sequential processing pipeline that includes:

1. **Feature Extraction**: Processing raw data into features using the `TerabyteFeatureExtractor`
2. **Temporal Processing**: Creating multi-stride temporal representations with `TerabyteTemporalStrideProcessor`
3. **Feature Learning**: Unsupervised feature learning with Restricted Boltzmann Machines
4. **Neural Network Processing**: Processing through liquid neural networks with motor neuron outputs

```python
class IntegratedPipeline:
    """
    Integrated pipeline for processing terabyte-scale data through
    feature extraction, RBM, and liquid neural network components.
    """
    
    def __init__(self, project_id=None, location="US", chunk_size=100000, ...):
        # Initialize components
        self.feature_extractor = None
        self.temporal_processor = None
        self.rbm = None
        self.liquid_network = None
        
    def extract_features(self, table_id, target_column=None, limit=None):
        # Extract features from data source
        
    def apply_temporal_processing(self, features_df):
        # Apply temporal processing to features
        
    def train_rbm(self, features, epochs=10):
        # Train RBM on features
        
    def extract_rbm_features(self, features):
        # Extract features from trained RBM
        
    def train_liquid_network(self, features, targets, ...):
        # Train liquid neural network
        
    def process_data(self, features, return_triggers=True):
        # Process data through the complete pipeline
```

#### Future Pipeline Evolution: Task-Based Architecture

The pipeline architecture is evolving from a sequential model to a more flexible "Tasks with handoffs" approach. This evolution will enable:

1. **Integration of NLP and Data Processing Blocks**:
   - Incorporating NLP tasks and autoencoder components into the pipeline
   - Allowing specialized transformations at different points in the feed-forward process
   - Enabling these non-neural components to participate in the layering and auto-wiring system
   - Supporting novel transformations of inputs or outputs at different stages of processing

2. **Asynchronous Communication and Actor Model**:
   - Transition from synchronous to asynchronous communication between components
   - Implementation of actor-based architecture for parallel processing
   - Leveraging coroutines for efficient task scheduling and execution
   - Potential integration with frameworks like Ray for distributed processing
   - Moving away from waiting for one channel while sending or receiving from another

3. **Cross-Language Support**:
   - Future expansion to Swift and Kotlin implementations
   - Ensuring compatibility across different programming languages
   - Leveraging platform-specific optimizations while maintaining a consistent API

This task-based architecture will allow for more complex processing flows where components can operate independently and in parallel, with well-defined interfaces for communication between tasks. The actor model provides a natural way to handle concurrency and distribution, with each actor responsible for a specific part of the processing pipeline.

#### Auto-Wiring for Non-Neural Components

The auto-wiring capabilities currently used in Neural Circuit Policies will be extended to non-neural tasks:

1. **Current Approach**:
   - Formulaic scenarios implemented through the Wiring class or Module classes
   - Manual configuration of connectivity patterns

2. **Future Direction**:
   - Integration of NLP or data processing blocks into the auto-wiring system
   - Dynamic measurement of success and optimization of connections
   - Drawing inspiration from Liquid Foundation Models' approach to connectivity
   - Using liquid kernels and sigmoids at layer boundaries to determine connections

This extension of auto-wiring to non-neural components will enable more flexible and powerful processing pipelines that can adapt to different data types and processing requirements. The integration with the actor model will allow for efficient parallel processing of these components, with dynamic routing of data between them.

#### Backend Integration in Pipelines

The pipeline architecture will support mixing different backend implementations:

1. **Backend-Agnostic Components**:
   - Each component in the pipeline can use the most appropriate backend
   - Seamless conversion between backends at component boundaries
   - Leveraging the strengths of each backend for specific tasks

2. **Mixed Backend Backbones**:
   - Supporting different backends for different parts of the pipeline
   - Enabling specialized hardware acceleration for specific components
   - Optimizing performance across heterogeneous computing environments

This approach will allow for optimal performance across different hardware platforms and processing requirements, with each component using the most appropriate backend for its specific task.

### 5. Feature Extraction Framework

Ember ML includes a comprehensive feature extraction framework designed to handle various data types and scales, from small datasets to terabyte-scale data processing. The framework is built with backend-agnostic principles, allowing it to leverage different computational backends for optimal performance.

#### Core Feature Extraction Components

1. **Column-Based Feature Extraction**:
   - `ColumnFeatureExtractor`: Processes data on a per-column basis with specialized handling for different data types
   - `ColumnPCAFeatureExtractor`: Extends the base extractor with PCA-based dimensionality reduction
   - `TemporalColumnFeatureExtractor`: Adds temporal processing capabilities for time series data

2. **Terabyte-Scale Processing**:
   - `TerabyteFeatureExtractor`: Optimized for processing terabyte-sized datasets with efficient chunking and memory management
   - `TerabyteTemporalStrideProcessor`: Creates multi-stride temporal representations for large-scale time series data

3. **Interfaces and Abstractions**:
   - `TensorFeaturesInterface`: Defines abstract interfaces for tensor operations used in feature extraction
   - `TensorOpsInterface`: Provides backend-agnostic tensor operations for feature extraction

#### Feature Extraction Capabilities

The feature extraction framework provides specialized processing for different data types:

1. **Numeric Data**:
   - Scaling (standard, robust, minmax)
   - Imputation of missing values
   - PCA-based dimensionality reduction

2. **Categorical Data**:
   - One-hot encoding
   - Ordinal encoding
   - Target encoding

3. **Datetime Data**:
   - Cyclical encoding (sine/cosine transformations)
   - Component extraction (year, month, day, etc.)
   - Time-based features

4. **Text Data**:
   - Basic text features (length, word count, etc.)
   - TF-IDF vectorization
   - Count vectorization

5. **Temporal Data**:
   - Multi-stride window creation
   - PCA-based temporal blending
   - Incremental processing for large datasets

#### Backend-Agnostic Implementation

The feature extraction framework leverages Ember ML's backend abstraction to provide optimal performance across different hardware:

```python
# Example of backend-agnostic datetime feature creation
def _create_datetime_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Extract datetime components
    hours = df[col].dt.hour / 23.0
    days_of_week = df[col].dt.dayofweek / 6.0
    
    # Apply sine and cosine transformations using backend-agnostic functions
    hours_sin, hours_cos = backend_utils.sin_cos_transform(hours, period=1.0)
    dow_sin, dow_cos = backend_utils.sin_cos_transform(days_of_week, period=1.0)
    
    # Convert to numpy arrays for pandas
    df[f'{col}_sin_hour'] = backend_utils.tensor_to_numpy_safe(hours_sin)
    df[f'{col}_cos_hour'] = backend_utils.tensor_to_numpy_safe(hours_cos)
    
    return df
```

This approach allows the feature extraction code to run efficiently on different hardware platforms (CPU, GPU, Apple Silicon) by leveraging the appropriate backend for tensor operations.

#### Terabyte-Scale Processing

The `TerabyteFeatureExtractor` is designed to handle extremely large datasets by:

1. **Chunked Processing**: Processing data in manageable chunks to avoid memory issues
2. **Memory Optimization**: Monitoring and managing memory usage during processing
3. **BigQuery Integration**: Optimized queries for terabyte-scale BigQuery tables
4. **Distributed Processing**: Support for distributed processing of large datasets
5. **Progress Tracking**: Detailed logging and progress tracking for long-running operations

```python
def process_bigquery_in_chunks(
    self,
    table_id: str,
    processing_fn: Optional[callable] = None,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    max_chunks: Optional[int] = None
) -> Union[List[Any], pd.DataFrame]:
    # Get total row count
    total_rows = self.get_table_row_count(table_id, where_clause)
    
    # Calculate number of chunks
    num_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
    
    # Process each chunk
    results = []
    for i in range(num_chunks):
        # Create query for this chunk
        offset = i * self.chunk_size
        chunk_query = self.optimize_bigquery_query(
            table_id=table_id,
            columns=columns,
            where_clause=where_clause,
            limit=self.chunk_size,
            offset=offset
        )
        
        # Load and process chunk
        chunk_df = bf.read_gbq(chunk_query)
        if processing_fn:
            result = processing_fn(chunk_df)
            results.append(result)
        else:
            results.append(chunk_df)
        
        # Monitor memory and force garbage collection
        self._monitor_memory()
        gc.collect()
    
    # Combine results
    return pd.concat(results, ignore_index=True)
```

#### Multi-Stride Temporal Processing

The `TerabyteTemporalStrideProcessor` enables multi-scale temporal analysis by:

1. **Stride Perspectives**: Creating windows with different stride lengths to capture patterns at different time scales
2. **PCA-Based Dimensionality Reduction**: Applying PCA to reduce the dimensionality of the windowed data
3. **Incremental Processing**: Supporting incremental PCA for large datasets
4. **Stateful Processing**: Maintaining state between batches for continuous processing

```python
def process_large_dataset(
    self,
    data_generator: Generator,
    maintain_state: bool = True
) -> Dict[int, Any]:
    results = {stride: [] for stride in self.stride_perspectives}
    
    for batch_idx, batch_data in enumerate(data_generator):
        # If state buffer exists and maintain_state is True, prepend to current batch
        if maintain_state and self.state_buffer is not None:
            state_buffer_tensor = backend_utils.convert_to_tensor_safe(self.state_buffer)
            batch_tensor = backend_utils.vstack_safe([state_buffer_tensor, batch_tensor])
        
        # Process batch for each stride perspective
        batch_results = self.process_batch(batch_tensor)
        
        # Append results
        for stride, data in batch_results.items():
            results[stride].append(data)
    
    # Combine results for each stride
    combined_results = {}
    for stride, data_list in results.items():
        tensors = [backend_utils.convert_to_tensor_safe(data) for data in data_list]
        combined_results[stride] = backend_utils.vstack_safe(tensors)
    
    return combined_results
```

### 6. Neural Network Framework

#### Module Hierarchy

The Ember ML neural network framework is built around a hierarchical module system that provides increasing levels of specialization:

1. **BaseModule (Module)**: The foundation for all neural network components
   - Parameter management (register_parameter, named_parameters, parameters)
   - Buffer management (register_buffer, named_buffers, buffers)
   - Submodule management (add_module, named_modules, modules)
   - Training/evaluation modes (train, eval)
   - Device and dtype handling (to)
   - Gradient management (zero_grad)

2. **ModuleCell**: Specialized for recurrent neural networks
   - Extends BaseModule with recurrent cell-specific functionality
   - Adds input_size, hidden_size, and activation parameters
   - Provides state_size and output_size properties
   - Implements reset_state method for initializing cell state

3. **ModuleWiredCell**: Specialized for wired connectivity patterns
   - Extends BaseModule with wiring-specific functionality
   - Integrates with Wiring configurations
   - Provides properties for accessing wiring information (state_size, layer_sizes, num_layers, sensory_size, motor_size)
   - Supports synapse counting and management

This hierarchical approach allows for increasing specialization while maintaining a consistent interface. The framework is evolving towards a more control-theory friendly architecture with automatic wiring capabilities and support for hybrid modules.

#### Evolution Towards Control-Theory Friendly Autowiring

The Ember ML project is moving towards a more integrated approach to wiring and connectivity:

1. **Current Architecture**:
   - Separate NCP and AutoNCP implementations
   - Wiring configurations as separate objects
   - Manual integration between modules and wiring

2. **Evolving Architecture**:
   - Integration of wiring capabilities into base neurons
   - Automatic wiring schemes for common connectivity patterns
   - Support for dynamic wiring for advanced hybrid State Space Models (SSMs)
   - More direct support for control-theory principles

This evolution will enable more flexible and powerful neural network architectures, particularly for control systems and time-series modeling. The integration of wiring capabilities into base neurons will simplify the creation of complex connectivity patterns and enable more advanced neural architectures.

#### Base Module System
The `BaseModule` class provides the foundation for building neural network components:

- **Parameter Management**: Tracking and updating parameters through the `Parameter` class
- **Module Composition**: Building complex modules from simpler ones using `add_module`
- **Training/Evaluation Modes**: Switching between training and evaluation modes with `train()` and `eval()`
- **Device and Dtype Handling**: Managing devices and data types with `to()`
- **Gradient Management**: Zeroing gradients with `zero_grad()`

The `BaseModule` class follows a design similar to PyTorch's `nn.Module`, with methods for registering parameters, buffers, and submodules, and for iterating over them. This design allows for building complex neural network architectures by composing simpler components.

#### Cell-Based Recurrent Networks
- **Basic RNN**: Simple recurrent cells
- **LSTM**: Long Short-Term Memory cells
- **GRU**: Gated Recurrent Units
- **CFC**: Closed-form Continuous-time cells
- **LTC**: Liquid Time-Constant cells
- **Stride-Aware Cells**: Cells for multi-scale temporal processing

#### Neural Circuit Policies (NCPs)

The Neural Circuit Policy (NCP) implementation in Ember ML is based on the original work by Mathias Lechner et al. in their paper "Neural Circuit Policies Enabling Auditable Autonomy" (Nature Machine Intelligence, 2020). It follows a biologically-inspired approach to neural network design, with a focus on control theory principles.

The NCP architecture consists of three main components:

1. **Wiring**: Defines the connectivity pattern between neurons
   - `Wiring`: Base class for all wiring configurations
   - `NCPWiring`: Specific implementation for Neural Circuit Policies
   - Divides neurons into sensory, inter, and motor groups
   - Configurable sparsity levels for different connection types

2. **NCP Module**: Implements the neural network using the wiring configuration
   - Applies masks from the wiring to enforce the connectivity pattern
   - Uses masked matrix multiplications for efficient computation
   - Supports both forward pass and step-by-step processing

3. **AutoNCP**: Convenience wrapper for automatic wiring configuration
   - Automatically configures the wiring based on the number of units and outputs
   - Calculates appropriate numbers of inter and command neurons
   - Simplifies the creation of NCPs for common use cases

Key aspects of the NCP implementation include:

- **Neuron Groups**: Neurons are divided into three groups:
  - **Sensory Neurons**: Receive input from the environment
  - **Inter Neurons**: Process information internally
  - **Motor Neurons**: Produce output to the environment

- **Connectivity Pattern**: The connectivity between neuron groups is defined by:
  - **Sparsity Level**: Controls the density of connections
  - **Group-Specific Sparsity**: Different sparsity levels for different connection types
  - **Random Connectivity**: Connections are randomly generated based on sparsity

- **Masking Mechanism**: Three types of masks enforce the connectivity pattern:
  - **Input Mask**: Determines which input dimensions connect to which neurons
  - **Recurrent Mask**: Determines which neurons connect to which other neurons
  - **Output Mask**: Determines which neurons contribute to the output

- **Forward Pass**: The forward pass applies these masks to ensure that only the connections defined in the wiring are used:
  ```python
  # Apply input mask
  masked_inputs = ops.multiply(inputs, self.input_mask)
  
  # Apply recurrent mask
  masked_state = ops.matmul(state, self.recurrent_mask)
  
  # Compute new state
  new_state = ops.matmul(masked_inputs, self.kernel)
  if self.use_bias:
      new_state = ops.add(new_state, self.bias)
  new_state = ops.add(new_state, ops.matmul(masked_state, self.recurrent_kernel))
  new_state = self.activation(new_state)
  
  # Compute output - only include motor neurons
  masked_output = ops.multiply(new_state, self.output_mask)
  ```

The NCP implementation is designed to be flexible and configurable, allowing for a wide range of connectivity patterns and neural network architectures. The `AutoNCP` class provides a convenient way to create NCPs with sensible defaults, making it easy to use NCPs in practice.

#### Restricted Boltzmann Machines (RBMs)
The Ember ML project includes two implementations of Restricted Boltzmann Machines:

1. **CPU-Friendly Implementation** (`RestrictedBoltzmannMachine`):
   - Optimized for computational efficiency on CPU with minimal requirements
   - Uses NumPy for efficient matrix computations
   - Implements mini-batch training and contrastive divergence
   - Provides methods for anomaly detection and generative capabilities

2. **PyTorch Implementation** (`RBM`):
   - Leverages PyTorch tensors and operations for potential GPU acceleration
   - Provides a more concise API compared to the CPU implementation
   - Can use CUDA or MPS for accelerated computation

RBMs in Ember ML can be used for:
- **Feature Extraction**: Extracting meaningful features from raw data
- **Anomaly Detection**: Detecting anomalies through reconstruction error or free energy
- **Generative Modeling**: Generating new samples from learned distributions

### 7. Future Architecture Components (Planned)

#### Registry System
- **Component Registration**: Mechanism for registering and retrieving components
- **Instantiation**: Dynamic instantiation of components based on configuration

#### Layer Primitives
- **Channel-Mixing Primitives**: Layers that mix information across feature dimensions
- **Sequence-Mixing Primitives**: Layers that mix information across sequence positions

#### Mixture of Experts (MoE)
- **Expert Networks**: Specialized neural networks for specific subsets of inputs
- **Routing Mechanism**: Determines which expert processes an input
- **Combination**: Methods for combining expert outputs

#### FFT Convolution
- **Efficient Implementation**: Fast Fourier Transform-based convolution for efficient attention
- **CUDA Optimization**: Low-level optimizations for performance

#### Self-Training and Self-Tuning
- **Self-Training Pipeline**: Semi-supervised learning with unlabeled data
- **Self-Tuning Module**: Dynamic hyperparameter optimization

#### Block Architecture
- **Higher-Level Components**: Blocks that combine multiple layers
- **Residual Connections**: Improved gradient flow in deep networks
- **Configuration-Driven Design**: Flexible configuration of blocks

## Implementation Patterns

### 1. Function-First Implementation
```python
# In ember_ml/backend/{backend_name}/tensor/ops/{module}.py
def operation_name(tensor_obj, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        tensor_obj: The tensor object (instance of the backend's tensor class)
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    # Implementation
    return result
```

### 2. Method as Passthrough
```python
# In ember_ml/backend/{backend_name}/tensor/tensor.py
def operation_name(self, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    from ember_ml.backend.{backend_name}.tensor.ops.{module} import operation_name as op_func
    return op_func(self, *args, **kwargs)
```

### 3. Frontend Implementation
```python
# In ember_ml/nn/tensor/common/ember_tensor.py
def operation_name(self, *args, **kwargs):
    """
    Operation documentation.
    
    Args:
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of the operation
    """
    # Convert EmberTensor arguments to backend tensors if needed
    backend_args = [arg.to_backend_tensor() if isinstance(arg, EmberTensor) else arg for arg in args]
    backend_kwargs = {k: v.to_backend_tensor() if isinstance(v, EmberTensor) else v for k, v in kwargs.items()}
    
    # Call the backend operation
    result = operation_name(self._tensor, *backend_args, **backend_kwargs)
    
    # Wrap the result in an EmberTensor if needed
    return EmberTensor(result, device=self.device, requires_grad=self._requires_grad)
```

### 4. Backend-Specific Implementation Pattern

The MLX backend implementation demonstrates how operations are implemented in a backend-specific manner:

```python
# In ember_ml/backend/mlx/math_ops.py
def add(x: Optional[Union[int, float, list, tuple, np.ndarray, mx.array, MLXTensor]],
        y: Optional[Union[int, float, list, tuple, np.ndarray, mx.array, MLXTensor]]) -> mx.array:
    """
    Add two MLX arrays element-wise.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Element-wise sum
    """
    return mx.add(Tensor.convert_to_tensor(x), Tensor.convert_to_tensor(y))
```

Key aspects of backend-specific implementations:
- **Type Conversion**: Converting various input types to backend-specific tensors
- **Backend-Specific Operations**: Using the backend's native operations
- **Error Handling**: Handling edge cases and providing informative error messages
- **Custom Implementations**: Implementing operations not directly available in the backend

### 5. Strong Typing Pattern

The strong typing pattern in the MLX backend ensures type safety and explicit conversions:

```python
# In ember_ml/backend/mlx/tensor/ops/indexing.py
def slice_tensor(tensor: Union[MLXTensor,mx.array], starts: Sequence[int], sizes: Sequence[int]) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension
        
    Returns:
        Sliced tensor
    """
    # Convert input to MLX array
    tensor_array = Tensor.convert_to_tensor(tensor)
    
    # Convert starts to MLX array
    if isinstance(starts, (list, tuple)):
        starts_mx = mx.array(starts, dtype=mx.int32)
    else:
        # Handle scalar case
        starts_mx = mx.array([starts], dtype=mx.int32)
    
    # Create axes as a tuple of integers
    axes = tuple(range(len(starts_mx)))
    
    # Use MLX's slice function
    return mx.slice(tensor_array, starts_mx, axes, sizes)
```

Key aspects of the strong typing pattern:
- **Explicit Type Annotations**: Clear type annotations for function parameters and return values
- **Type Validation**: Validation of input types before operations
- **Explicit Rejection**: Explicit rejection of incompatible types with informative error messages
- **Consistent Conversion**: Consistent conversion of inputs to the appropriate backend tensor type

### 6. Cell-Layer Pattern
```python
# Cell implementation
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Initialize parameters
        
    def forward(self, input, state):
        # Implement single-step computation
        return output, new_state

# Layer implementation
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # Create cells
        self.cells = [LSTMCell(input_size, hidden_size) for _ in range(num_layers)]
        
    def forward(self, input_sequence):
        # Process sequence through cells
        return sequence_output
```

### 7. Wiring Configuration Pattern
```python
# Create wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5
)

# Create NCP with wiring
model = NCP(wiring=wiring)
```

### 8. Module Hierarchy Pattern

The module hierarchy pattern establishes a clear inheritance structure for specialized modules:

```python
# Base module for all neural network components
class BaseModule:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
    
    # ... parameter management, device handling, etc.

# Specialized module for recurrent cells
class ModuleCell(BaseModule):
    def __init__(self, input_size, hidden_size, activation="tanh", use_bias=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        self.activation = ops.get_activation(activation)
        self.use_bias = use_bias
    
    @property
    def state_size(self):
        return self.hidden_size
    
    @property
    def output_size(self):
        return self.hidden_size
    
    def reset_state(self, batch_size=1):
        # Initialize cell state
        return tensor.zeros((batch_size, self.state_size))

# Specialized module for wired connectivity
class ModuleWiredCell(BaseModule):
    def __init__(self, input_size, wiring, mode="default", **kwargs):
        super().__init__()
        self.wiring = wiring
        # Build the wiring if needed
        if input_size is not None:
            wiring.build(input_size)
        # ... wiring-specific initialization
    
    @property
    def state_size(self):
        return self.wiring.units
    
    # ... wiring-specific properties and methods
```

This pattern allows for specialized functionality while maintaining a consistent interface across all module types.

### 9. Feature Extraction Pattern

The feature extraction pattern provides a consistent approach to processing different data types:

```python
# Column-based feature extraction
class ColumnFeatureExtractor:
    def __init__(self, numeric_strategy='standard', categorical_strategy='onehot', 
                 datetime_strategy='cyclical', text_strategy='basic'):
        # Initialize strategies
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.text_strategy = text_strategy
        
        # Initialize processors
        self.column_processors = {}
        self.column_types = {}
    
    def fit(self, df, target_column=None):
        # Detect column types
        self._detect_column_types(df)
        
        # Fit processors for each column type
        for column, col_type in self.column_types.items():
            if col_type == 'numeric':
                self._fit_numeric_processor(df, column)
            elif col_type == 'categorical':
                self._fit_categorical_processor(df, column, target_column)
            # ... other column types
        
        return self
    
    def transform(self, df):
        # Transform each column using the fitted processors
        result_dfs = []
        for column, processor in self.column_processors.items():
            if column in df.columns:
                col_type = self.column_types[column]
                transformed = self._transform_column(df, column, col_type, processor)
                result_dfs.append(transformed)
        
        # Combine results
        return pd.concat(result_dfs, axis=1)
```

Key aspects of the feature extraction pattern:
- **Column Type Detection**: Automatic detection of column types
- **Type-Specific Processing**: Specialized processing for each data type
- **Fit/Transform API**: Consistent fit/transform API similar to scikit-learn
- **Backend-Agnostic Operations**: Using backend-agnostic operations for tensor manipulations

### 10. Asynchronous Task Pattern

The emerging asynchronous task pattern will enable parallel processing and efficient resource utilization:

```python
# Asynchronous task implementation (conceptual)
class AsyncTask:
    def __init__(self, name, processor, inputs=None):
        self.name = name
        self.processor = processor
        self.inputs = inputs or []
        self.outputs = []
        self.state = "pending"
    
    async def execute(self, input_data=None):
        # Set state to running
        self.state = "running"
        
        # Process inputs
        if input_data is not None:
            result = await self.processor(input_data)
        else:
            result = await self.processor()
        
        # Set state to completed
        self.state = "completed"
        
        # Return result
        return result

# Pipeline with asynchronous tasks
class AsyncPipeline:
    def __init__(self):
        self.tasks = {}
        self.connections = {}
    
    def add_task(self, task):
        self.tasks[task.name] = task
    
    def connect(self, source_task, target_task):
        if source_task not in self.connections:
            self.connections[source_task] = []
        self.connections[source_task].append(target_task)
    
    async def execute(self, initial_inputs=None):
        # Execute tasks in parallel based on dependencies
        # ...
```

This pattern will enable efficient parallel processing of tasks, with each task operating independently and communicating through well-defined interfaces. The actor model provides a natural way to handle concurrency and distribution, with each actor responsible for a specific part of the processing pipeline.

## Backend Selection and Switching

Ember ML allows for dynamic backend selection and switching:

```python
from ember_ml.backend import set_backend, get_backend

# Set the backend to PyTorch
set_backend('torch')

# Create a tensor using PyTorch backend
x = EmberTensor([1, 2, 3])

# Switch to MLX backend
set_backend('mlx')

# x is automatically converted to MLX backend via NumPy
y = x + 1  # Uses MLX operations
```

## Supported Neural Network Architectures

1. **Liquid Neural Networks (LNN)**: Dynamic networks with adaptive connectivity
2. **Neural Circuit Policies (NCP)**: Biologically-inspired neural architectures
3. **Stride-Aware Continuous-time Fully Connected (CfC)** networks
4. **Restricted Boltzmann Machines (RBM)**: Energy-based models for unsupervised learning
5. **Specialized attention mechanisms and temporal processing units**

## Supported Backends

1. **MLX**: Optimized for Apple Silicon
2. **PyTorch**: For CUDA and other GPU platforms
3. **NumPy**: For CPU computation
4. **Future support for additional backends**

## Future Enhancements

1. **Operator Overloading**: Implementation of operator overloading for EmberTensor
2. **Static Methods**: Implementation of static methods for common tensor operations
3. **Registry System**: Flexible component registration and instantiation
4. **Mixture of Experts (MoE)**: Specialized processing of different parts of input data
5. **FFT Convolution**: Efficient alternative to traditional attention mechanisms
6. **Self-Training and Self-Tuning**: Semi-supervised learning and hyperparameter optimization
7. **Block Architecture**: Higher-level components that combine multiple layers
8. **Configuration System**: Configuration-driven design for flexible component configuration
9. **Distributed Training Support**: Support for model and data parallelism
10. **Cross-Language Support**: Swift and Kotlin implementations for platform-specific optimizations

## Conclusion

The Ember ML architecture is designed for flexibility, consistency, and performance. The function-first design pattern, backend abstraction, and modular component architecture enable a clean, maintainable codebase that can easily adapt to new requirements and backends. The separation of functions from class implementations and the careful design of abstract classes contribute to memory efficiency, allowing the framework to handle large models and datasets without excessive memory usage.

The strong typing implementation in the MLX backend demonstrates a commitment to type safety and explicit conversions, reducing the risk of unexpected behavior due to implicit conversions. This approach ensures that tensor operations are performed on compatible tensor types and provides clear error messages when type mismatches occur.

The feature extraction framework provides powerful tools for processing various data types and scales, from small datasets to terabyte-scale data. The backend-agnostic implementation allows for optimal performance across different hardware platforms, while the specialized processing for different data types enables comprehensive feature engineering.

The module hierarchy (BaseModule, ModuleCell, ModuleWiredCell) provides a foundation for specialized neural network components while maintaining a consistent interface. The evolution towards control-theory friendly autowiring and the integration of wiring capabilities into base neurons will enable more flexible and powerful neural network architectures, particularly for control systems and time-series modeling.

The pipeline architecture is evolving towards a more flexible task-based system with asynchronous communication and actor-based parallelism. This evolution will enable more complex processing flows where components can operate independently and in parallel, with well-defined interfaces for communication between tasks. The integration of NLP and data processing blocks into the auto-wiring system will further enhance the flexibility and power of the framework.

The comprehensive set of mathematical operations and solver functions, implemented in a backend-specific manner while maintaining a consistent API, provides a solid foundation for implementing advanced neural network architectures. The Neural Circuit Policies implementation demonstrates how the framework can be used to create biologically-inspired neural networks with custom connectivity patterns, while the Restricted Boltzmann Machines implementation shows how the framework can support both CPU-optimized and GPU-accelerated implementations of the same algorithm.

The planned enhancements, including cross-language support and distributed processing capabilities, will further improve the framework's capabilities, making it more powerful and flexible for a wide range of machine learning tasks.