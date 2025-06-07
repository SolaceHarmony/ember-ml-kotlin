# Using Initializers in Ember ML: Best Practices

## Overview

This guide focuses on the proper way to use initializers in Ember ML, particularly in the context of Parameter attributes. It addresses the issues we discovered with Parameter attribute assignment and provides best practices for using initializers in a way that works consistently across all backends.

## Initializers in Ember ML

Ember ML provides several initializers for neural network parameters:

1. `glorot_uniform`: Glorot/Xavier uniform initialization
2. `glorot_normal`: Glorot/Xavier normal initialization
3. `orthogonal`: Orthogonal matrix initialization
4. `zeros`: Zeros initialization

These initializers can be used in two main ways:

### 1. Direct Usage

```python
from ember_ml.nn import initializers
from ember_ml.nn.modules import Parameter

# Initialize tensor directly
tensor_data = initializers.glorot_uniform((input_dim, output_dim))
parameter = Parameter(tensor_data)
```

### 2. String-Based Approach with Helper Function

```python
from ember_ml.nn import initializers

# Get initializer by name
initializer_fn = initializers.get_initializer('glorot_uniform')
tensor_data = initializer_fn(shape)
parameter = Parameter(tensor_data)
```

## Best Practices for Using Initializers with Parameters

### In Module Initialization

When initializing parameters in a module, follow this pattern:

```python
class MyModule(BaseModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Initialize attributes to None
        self._kernel = None
        self._bias = None
        
        # Build parameters
        self.build((input_dim,))
    
    def build(self, input_shape):
        if self.built:
            return
            
        input_dim = input_shape[-1]
        
        # Initialize parameters using initializers
        kernel_data = initializers.glorot_uniform((input_dim, self.output_dim))
        self._kernel = Parameter(kernel_data)
        
        bias_data = initializers.zeros((self.output_dim,))
        self._bias = Parameter(bias_data)
        
        self.built = True
```

### In Deferred Build

For modules that use deferred building (initialized on first call):

```python
class LazyModule(BaseModule):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
        # Initialize attributes to None
        self._kernel = None
        self._bias = None
    
    def build(self, input_shape):
        if self.built:
            return
            
        input_dim = input_shape[-1]
        
        # Initialize parameters using initializers
        kernel_data = initializers.glorot_uniform((input_dim, self.output_dim))
        self._kernel = Parameter(kernel_data)
        
        bias_data = initializers.zeros((self.output_dim,))
        self._bias = Parameter(bias_data)
        
        self.built = True
```

## Using Initializers with the NCP Module

The NCP module uses a more sophisticated approach with a helper method and string-based configuration:

```python
def _initialize_tensor(self, shape, initializer):
    """Initialize a tensor with the specified shape and initializer."""
    # Get initializer by name
    initializer_fn = initializers.get_initializer(initializer)
    # Call the initializer function
    return initializer_fn(shape, dtype=self.dtype)

# In build method:
tensor_data = self._initialize_tensor(
    (self.neuron_map.input_dim, self.neuron_map.units),
    self.kernel_initializer
)
self._kernel = Parameter(tensor_data)
```

## Implementing the Parameter Attribute Fix

After applying the fix to `BaseModule.__setattr__`, you should be able to initialize parameters more naturally:

```python
# Before fix (workaround required):
tensor_data = initializers.glorot_uniform((input_dim, output_dim))
param = Parameter(tensor_data)
object.__setattr__(self, '_kernel', param)  # Bypass __setattr__

# After fix (natural assignment works):
tensor_data = initializers.glorot_uniform((input_dim, output_dim))
param = Parameter(tensor_data)
self._kernel = param  # Works correctly with all backends
```

## Testing Your Initializers

When testing initializers, verify both the statistical properties and the parameter assignment:

```python
def test_initializer_with_parameter():
    # Test initializer creates proper tensor
    data = initializers.glorot_uniform((10, 10))
    assert tensor.shape(data) == (10, 10)
    
    # Test parameter assignment works
    module = TestModule()
    param = Parameter(data)
    module.param = param
    
    # Verify both access methods work
    assert module.param is not None
    param_dict = dict(module.named_parameters())
    assert param_dict['param'] is module.param
```

## Common Pitfalls to Avoid

1. **Direct Tensor Assignment**: Don't assign tensors directly to attributes that should be parameters:
   ```python
   # Wrong
   self._kernel = initializers.glorot_uniform((input_dim, output_dim))
   
   # Right
   self._kernel = Parameter(initializers.glorot_uniform((input_dim, output_dim)))
   ```

2. **Missing Parameter Wrapper**: Always wrap initialized tensors in Parameter:
   ```python
   # Wrong
   tensor_data = initializers.glorot_uniform((input_dim, output_dim))
   self._kernel = tensor_data
   
   # Right
   tensor_data = initializers.glorot_uniform((input_dim, output_dim))
   self._kernel = Parameter(tensor_data)
   ```

3. **Inconsistent Backend Usage**: Ensure initializers are used with the current backend:
   ```python
   # Set backend first
   set_backend('mlx')
   
   # Then use initializers
   data = initializers.glorot_uniform((10, 10))
   ```

## Conclusion

With the fix to the Parameter attribute assignment issue, using initializers in Ember ML becomes more straightforward and consistent across all backends. Follow the patterns outlined in this guide to ensure your parameters are properly initialized and accessible both directly and via parameter collection.

Remember the key steps:
1. Create tensor data using initializers
2. Wrap in Parameter
3. Assign to module attribute
4. Verify parameter is accessible both directly and via collection

This approach will work reliably across all backends, including MLX.