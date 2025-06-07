# EmberTensor Design Improvement Plan

## Current Issues

1. **Inconsistent Dtype Handling**:
   - EmberTensor doesn't store dtype explicitly as a member variable (`_dtype`)
   - Instead relies on extracting dtype from the underlying tensor each time it's accessed
   - This can lead to inconsistencies if the underlying tensor's dtype changes

2. **Tensor Consistency Challenges**:
   - No clear mechanism to ensure consistent handling of `_tensor` across operations
   - New tensors created during operations may not maintain the same properties

3. **Missing Serialization Support**:
   - No explicit serialization methods (`__getstate__`, `__setstate__`, `__iter__`)
   - Makes saving/loading tensors difficult, especially with NumPy's pickle-based approach

4. **Lack of Backend Metadata**:
   - No mechanism to track which backend the tensor data originated from
   - This information would be valuable for save/load operations

## Proposed Solutions

### 1. Explicit Dtype Storage

```python
def __init__(
    self,
    data: Optional[Any] = None,
    *,
    dtype: Optional[Union[DType, str, Callable[[], Any]]] = None,
    device: Optional[str] = None,
    requires_grad: bool = False
) -> None:
    # Process dtype
    self._processed_dtype = None
    if dtype is not None:
        if callable(dtype):
            self._processed_dtype = dtype()
        elif isinstance(dtype, str):
            self._processed_dtype = dtype
        else:
            self._processed_dtype = dtype
    
    # Create tensor with processed dtype
    self._tensor = convert_to_tensor(data, dtype=self._processed_dtype)
    
    # Store the dtype explicitly
    backend_dtype = dtype_func(self._tensor)
    if isinstance(backend_dtype, DType):
        self._dtype = backend_dtype
    else:
        dtype_name = str(backend_dtype).split('.')[-1]
        self._dtype = DType(dtype_name)
    
    # Store device and requires_grad
    from ember_ml.backend import get_device
    self._device = device if device is not None else get_device()
    self._requires_grad = requires_grad
    
    # Store backend information
    self._backend = get_backend()
```

### 2. Consistent Property Access

```python
@property
def dtype(self) -> DType:
    """Get the dtype of the tensor."""
    return self._dtype

@property
def backend(self) -> str:
    """Get the backend the tensor was created with."""
    return self._backend
```

### 3. Serialization Support

```python
def __getstate__(self):
    """Get the state for pickling."""
    return {
        'tensor_data': to_numpy(self._tensor),
        'dtype': str(self._dtype),
        'device': self._device,
        'requires_grad': self._requires_grad,
        'backend': self._backend
    }

def __setstate__(self, state):
    """Set the state during unpickling."""
    self._dtype = EmberDType(state['dtype'])
    self._device = state['device']
    self._requires_grad = state['requires_grad']
    self._backend = state['backend']
    self._tensor = convert_to_tensor(state['tensor_data'], dtype=self._dtype)

def __iter__(self):
    """Make the tensor iterable."""
    # Convert to numpy and iterate
    for item in to_numpy(self._tensor):
        yield item
```

### 4. Consistent Operation Results

Update all methods that return new tensors to ensure they properly propagate dtype, device, and backend information:

```python
def some_operation(self, x: Any, ...) -> 'EmberTensor':
    # Process inputs
    if isinstance(x, EmberTensor):
        x = x.to_backend_tensor()
    
    # Perform operation
    result = operation_func(x, ...)
    
    # Return new tensor with consistent properties
    return EmberTensor(
        result,
        dtype=self._dtype,  # Use stored dtype
        device=self._device,
        requires_grad=self._requires_grad
    )
```

## Implementation Strategy

1. **Phase 1: Core Changes**
   - Add explicit dtype storage in `__init__`
   - Add backend tracking
   - Update property accessors

2. **Phase 2: Serialization Support**
   - Implement `__getstate__` and `__setstate__`
   - Implement `__iter__`
   - Test serialization with different backends

3. **Phase 3: Operation Consistency**
   - Update all methods that return new tensors
   - Ensure consistent property propagation

4. **Phase 4: Testing**
   - Create comprehensive tests for dtype consistency
   - Test serialization across backends
   - Verify backward compatibility

## Benefits

1. **Improved Consistency**: Explicit storage of dtype and backend ensures consistent behavior
2. **Better Serialization**: Proper serialization methods enable reliable save/load operations
3. **Enhanced Debugging**: Backend tracking makes it easier to diagnose issues
4. **Future Extensibility**: Cleaner architecture allows for easier addition of new features